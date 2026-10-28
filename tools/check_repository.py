"""Check source and the planning contract using only Python 3.11+ stdlib."""
import ast
import hashlib
import json
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("draft_k", "ddpo_is", "flow_grpo", "lambda_grpo", "offline_d3po")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def check_plan():
    plan = read_json(ROOT / "configs/experiment-plan.json")
    require(plan["schema"] == "ogo-experiment-plan-v2", "Wrong planning schema")
    require(plan["purpose"] == "planning_only_not_an_executable_campaign", "Plan must not be executable")
    require(tuple(plan["common_baselines"]) == BASELINES, "Common five changed")
    require(plan["deployment"]["steps"] == [16, 32], "Deployment horizons changed")
    require(plan["deployment"]["solver"] == "deterministic_uniform_euler", "Deployment solver changed")
    tasks = plan["tasks"]
    expected_ids = {"synthetic", "imagenet64", "imagenet256", "text_to_image_sd35", "text_to_image_sana", "text_to_video"}
    require(len(tasks) == len(expected_ids) and {t["id"] for t in tasks} == expected_ids, "Task coverage changed")
    for task in tasks:
        if task["kind"] == "policy_optimization":
            require(tuple(task["baselines"]) == BASELINES, f"Five-baseline coverage: {task['id']}")
        else:
            require(task["id"] == "synthetic" and not task["baselines"], "Synthetic uses mechanism controls")
        require(task["scope"] == ("optional" if task["id"] == "text_to_video" else "core"), "Core/optional scope changed")
        if task["production_config"] is not None:
            require(task["id"] == "imagenet256" and (ROOT / task["production_config"]).is_file(), "Unsupported production task")
        if task.get("task") == "text_to_image" or task["id"] == "text_to_video":
            require(task["status"] == "planned_not_implemented" and task["budget_gpu_hours"] is None
                    and task["production_config"] is None and task["launch_gates"], "Planned task cannot bypass launch gates")
    policy_tasks = [t for t in tasks if t["kind"] == "policy_optimization"]
    t2i = [t for t in tasks if t.get("task") == "text_to_image"]
    require({t["id"] for t in t2i} == {"text_to_image_sd35", "text_to_image_sana"}, "T2I backbone coverage changed")
    require(all(t["resolution_planning_target"] == [512, 512] for t in t2i), "T2I resolution changed")
    sana = next(t for t in t2i if t["id"] == "text_to_image_sana")
    require(sana["backbone"] == "SANA-1.6B" and sana["checkpoint"] == "Efficient-Large-Model/Sana_1600M_512px_diffusers", "Selected SANA variant changed")
    per_backbone = (1 + len(BASELINES)) * len(plan["deployment"]["steps"])
    counts = plan["counts_per_training_seed"]
    require(counts["trained_configurations_per_backbone"] == per_backbone, "Per-backbone count mismatch")
    require(counts["text_to_image_configurations"] == per_backbone * len(t2i), "T2I count mismatch")
    require(counts["core_nonsynthetic_configurations"] == per_backbone * sum(t["scope"] == "core" for t in policy_tasks), "Core count mismatch")
    require(counts["including_optional_video"] == per_backbone * len(policy_tasks), "Optional count mismatch")
    for name, expected in plan["frozen_contract_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f"Frozen contract bytes changed: {name}")
    campaign = read_json(ROOT / "configs/imagenet256-baselines-512.json")
    profile = read_json(ROOT / "configs/ogo-profile.json")
    require(tuple(campaign["methods"]) == BASELINES and campaign["optimizer_updates"] == 512, "Production scope changed")
    require(campaign["methods"] == {m: profile["methods"][m] for m in BASELINES}, "Baseline settings differ")
    require(campaign["thresholds"] == profile["thresholds"], "Numerical thresholds differ")
    require(campaign["baseline_contract_sha256"] == plan["frozen_contract_sha256"]["configs/ogo-profile.json"], "Contract identity differs")
    return counts


def check_source():
    package = ROOT / "src/ogo"
    require(not (ROOT / "src/fct").exists(), "Retired namespace remains")
    sources = sorted(package.glob("*.py")) + sorted((ROOT / "tests").glob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    trees = {p: ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in sources}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = None
            if node.level and path.parent == package:
                require(node.level == 1, f"Unexpected relative import in {path.name}")
                module = node.module
            elif node.module and node.module.startswith("ogo."):
                module = node.module.removeprefix("ogo.")
            if module:
                target = package.joinpath(*module.split("."))
                require(target.with_suffix(".py").is_file() or (target / "__init__.py").is_file(), f"Missing import {module} in {path.name}")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for name, value in project["scripts"].items():
        module, function = value.split(":")
        require(module.startswith("ogo."), f"Stale entry point: {name}")
        path = ROOT / "src" / (module.replace(".", "/") + ".py")
        require(path in trees and any(isinstance(n, ast.FunctionDef) and n.name == function for n in trees[path].body), f"Missing entry point: {name}")
    version = next(n.value.value for n in trees[package / "__init__.py"].body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets))
    require(version == project["version"], "Package version mismatch")
    return len(sources)


def check_documents():
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted((ROOT / "experiments").rglob("*.md"))]
    links = 0
    for path in documents:
        for target in re.findall(r"\[[^\]\n]+\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            require((path.parent / target.split("#", 1)[0]).exists(), f"Broken link in {path.name}: {target}")
            links += 1
    for folder in ("configs", "docs", "experiments"):
        for path in (ROOT / folder).rglob("*.json"):
            read_json(path)
    return links


def main():
    result = {"planning_counts": check_plan(), "python_sources": check_source(),
              "local_document_links": check_documents(), "passed": True,
              "scope": "static_source_and_plan_only_no_model_execution"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
