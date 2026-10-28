"""Run the predeclared finite-step gradient study; never train a model."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .verification_cases import (CONTROLS, PARAMETERS, REGIMES, autograd_gradient, discrete_adjoint,
                   frozen_rollout, loop_objective, objective, rollout, scalar_adjoint,
                   scalar_oracle, scalar_rollout, terminal_covector, terminal_objective)


RAW_FIELDS = ["fixture", "setting", "device", "seed", "steps", "ensemble", "state",
              "regime", "direction", "epsilon_relative", "epsilon_absolute", "method",
              "derivative", "finite_difference", "absolute_error", "relative_error",
              "tolerance", "within_tolerance"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json", ".sh", ".txt")
            and "__pycache__" not in p.parts}


def close(a, b, atol, rtol):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                and np.all(np.abs(a - b) <= atol + rtol * np.maximum(np.abs(a), np.abs(b))))


def stable_window(ratios, length):
    """Smallest worst error/tolerance over a consecutive epsilon window."""
    ratios = np.asarray(ratios)
    return float(min(np.max(ratios[i:i + length]) for i in range(len(ratios) - length + 1)))


def epsilon_grid(config):
    return np.logspace(config["epsilon_log10_start"], config["epsilon_log10_stop"],
                       config["epsilon_count"])


def make_fixture(seed, config):
    # Separate generators encode independence, even though every draw is fixed for FD.
    student_rng, teacher_rng, model_rng, direction_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    contexts = np.array([-1.0, 1.0])
    initial = student_rng.normal(size=(2, max(config["ensembles"]), 2))
    teacher_initial = teacher_rng.normal(size=(2, 1, 2))
    base = np.zeros(PARAMETERS)
    base[:32] = model_rng.normal(scale=0.5, size=32)
    nonzero = base.copy()
    nonzero[40:] = model_rng.normal(scale=0.1, size=18)
    directions = direction_rng.normal(size=(config["directions"], PARAMETERS))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    with torch.no_grad():
        target = frozen_rollout(torch.tensor(teacher_initial), torch.tensor(contexts),
                                config["teacher_steps"])[:, 0].numpy()
    return dict(contexts=contexts, initial=initial, teacher_initial=teacher_initial,
                target=target, zero=base, nonzero=nonzero, directions=directions)


def finite_differences(theta, initial, target, contexts, steps, regime, directions, relative_eps):
    eps = relative_eps * max(1.0, float(torch.linalg.vector_norm(theta)))
    offsets = directions[:, None, :] * theta.new_tensor(eps)[None, :, None]
    plus, minus = theta + offsets, theta - offsets
    parameters = torch.stack((plus, minus), dim=2).reshape(-1, PARAMETERS)
    with torch.no_grad():
        values = torch.vmap(lambda p: objective(p, initial, target, contexts, steps, regime))(parameters)
    values = values.reshape(len(directions), len(eps), 2)
    fd = (values[..., 0] - values[..., 1]) / theta.new_tensor(2.0 * eps)[None, :]
    # Verify batching does not change the scalar being differentiated.
    with torch.no_grad():
        index = len(eps) // 2
        sequential = objective(plus[0, index], initial, target, contexts, steps, regime)
    batch_delta = abs(float(sequential - values[0, index, 0]))
    return fd.cpu().numpy(), eps, batch_delta


def record_curve(writer, meta, method, projections, fd, relative_eps, eps, config):
    projections = np.asarray(projections)
    windows = []
    for direction, projection in enumerate(projections):
        ratios = []
        for j, e in enumerate(eps):
            other = float(fd[direction, j])
            error = abs(float(projection) - other)
            tolerance = config["fd_atol"] + config["fd_rtol"] * max(abs(projection), abs(other))
            ratio = error / tolerance
            ratios.append(ratio if np.isfinite(ratio) else float("inf"))
            writer.writerow(dict(meta, direction=direction, epsilon_relative=relative_eps[j],
                                 epsilon_absolute=e, method=method, derivative=projection,
                                 finite_difference=other, absolute_error=error,
                                 relative_error=error / max(abs(projection), abs(other), 1e-12),
                                 tolerance=tolerance, within_tolerance=bool(ratio <= 1)))
        windows.append(stable_window(ratios, config["consecutive_passes"]))
    worst = max(windows)
    return {"passed": bool(worst <= 1), "passed_directions": sum(x <= 1 for x in windows),
            "directions": len(windows), "worst_window_ratio": worst if np.isfinite(worst) else None}


def applicable(control, steps, state, regime):
    if state != "nonzero" or steps <= 1:
        return False
    return control != "one_sided_kernel" or regime != "utility"


def nonlinear_case(writer, fixture, seed, state, steps, ensemble, regime, device, config, output):
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device=device)

    theta, contexts = tensor(fixture[state]), tensor(fixture["contexts"])
    initial, target = tensor(fixture["initial"][:, :ensemble]), tensor(fixture["target"])
    directions = tensor(fixture["directions"])
    name = f"s{seed}-{state}-n{steps}-k{ensemble}-{regime}"
    meta = dict(fixture=name, setting="nonlinear2d", device=device, seed=seed,
                steps=steps, ensemble=ensemble, state=state, regime=regime)
    loss, full = autograd_gradient(theta, initial, target, contexts, steps, regime)
    adjoint = discrete_adjoint(theta, initial, target, contexts, steps, regime)
    with torch.no_grad():
        endpoint = rollout(theta, initial, contexts, steps)
        independent_value = loop_objective(endpoint, target, contexts, regime)
        frozen = frozen_rollout(initial, contexts, steps)
        zero_endpoint = rollout(tensor(fixture["zero"]), initial, contexts, steps)
        analytic_terminal = terminal_covector(endpoint, target, contexts, regime)
    terminal_input = endpoint.detach().requires_grad_(True)
    terminal_gradient, = torch.autograd.grad(terminal_objective(terminal_input, target, contexts, regime),
                                            terminal_input)
    relative_eps = epsilon_grid(config)
    fd, eps, batch_delta = finite_differences(theta, initial, target, contexts, steps, regime,
                                             directions, relative_eps)
    gradient = full.cpu().numpy()
    adjoint_array = adjoint.cpu().numpy()
    checks = {
        "adjoint_agreement": close(gradient, adjoint_array, config["gradient_atol"], config["gradient_rtol"]),
        "loop_objective_agreement": close(float(loss), independent_value, config["value_atol"], config["value_rtol"]),
        "terminal_covector_agreement": close(terminal_gradient.detach().cpu(), analytic_terminal.cpu(),
                                             config["gradient_atol"], config["gradient_rtol"]),
        "zero_correction_agreement": close(zero_endpoint.cpu(), frozen.cpu(), config["value_atol"], config["value_rtol"]),
        "batched_objective_agreement": batch_delta <= config["value_atol"] + config["value_rtol"] * abs(float(loss)),
    }
    methods = {"full": full}
    forwards = {"full": float(loss)}
    for control in CONTROLS:
        control_loss, control_gradient = autograd_gradient(theta, initial, target, contexts,
                                                          steps, regime, control)
        methods[control], forwards[control] = control_gradient, float(control_loss)
        checks[control + "_forward_equal"] = bool(torch.equal(loss, control_loss))
    outcomes = {}
    for method, grad in methods.items():
        outcomes[method] = record_curve(writer, meta, method, (directions @ grad).cpu().numpy(),
                                        fd, relative_eps, eps, config)
        outcomes[method]["applicable_control"] = method != "full" and applicable(method, steps, state, regime)
        outcomes[method]["gradient_max_abs_difference"] = float(torch.max(torch.abs(grad - full)))
    np.savez_compressed(output / "gradients" / f"{device}-{name}.npz", full=gradient,
                        adjoint=adjoint_array, endpoint=endpoint.cpu().numpy(),
                        fd=fd, epsilon_absolute=eps,
                        **{key: value.cpu().numpy() for key, value in methods.items() if key != "full"})
    return {**meta, "objective": float(loss), "loop_objective": independent_value,
            "adjoint_max_abs_error": float(np.max(np.abs(gradient - adjoint_array))),
            "objective_abs_error": abs(float(loss) - independent_value),
            "terminal_covector_max_abs_error": float(torch.max(torch.abs(terminal_gradient.detach() - analytic_terminal))),
            "batch_scalar_abs_error": batch_delta, "checks": checks, "methods": outcomes}


def scalar_cases(writer, device, config):
    cases = []
    relative_eps = epsilon_grid(config)
    for steps, value in itertools.product(config["steps"], config["scalar_parameters"]):
        phi = torch.tensor(value, device=device, dtype=torch.float64, requires_grad=True)
        endpoint = scalar_rollout(phi, steps)
        loss = 0.5 * (endpoint - 2.0).square()
        grad, = torch.autograd.grad(loss, phi)
        analytic_endpoint, oracle = scalar_oracle(value, steps)
        adjoint = scalar_adjoint(value, steps)
        eps = relative_eps * max(1.0, abs(value))
        with torch.no_grad():
            plus = scalar_rollout(phi + phi.new_tensor(eps), steps)
            minus = scalar_rollout(phi - phi.new_tensor(eps), steps)
            fd = ((0.5 * (plus - 2.0).square() - 0.5 * (minus - 2.0).square()) / phi.new_tensor(2.0 * eps)).cpu().numpy()[None]
        meta = dict(fixture=f"scalar-n{steps}-phi{value:g}", setting="scalar", device=device,
                    seed="", steps=steps, ensemble=1, state=value, regime="utility")
        methods = {key: record_curve(writer, meta, key, [derivative], fd, relative_eps, eps, config)
                   for key, derivative in (("full", float(grad)), ("adjoint", adjoint), ("oracle", oracle))}
        checks = {"closed_form_gradient": close(float(grad), oracle, config["gradient_atol"], config["gradient_rtol"]),
                  "adjoint_agreement": close(float(grad), adjoint, config["gradient_atol"], config["gradient_rtol"]),
                  "closed_form_endpoint": close(float(endpoint.detach()), analytic_endpoint, config["value_atol"], config["value_rtol"])}
        cases.append({**meta, "gradient": float(grad), "oracle_gradient": oracle,
                      "adjoint_max_abs_error": abs(float(grad) - adjoint),
                      "oracle_abs_error": abs(float(grad) - oracle), "checks": checks, "methods": methods})
    return cases


def validate_config(config):
    if config["schema"] != "fct-gradient-correctness-v1":
        raise ValueError("Unsupported configuration schema")
    if any(n < 1 or not isinstance(n, int) for n in config["steps"]):
        raise ValueError("Euler budgets must be positive integers")
    if any(k < 2 for k in config["ensembles"]):
        raise ValueError("Study ensembles must have K >= 2")
    if not set(config["regimes"]) <= set(REGIMES) or not set(config["states"]) <= {"zero", "nonzero"}:
        raise ValueError("Unknown objective regime or initialization")
    if not 1 <= config["consecutive_passes"] <= config["epsilon_count"]:
        raise ValueError("Invalid consecutive epsilon acceptance window")
    if any(config[key] <= 0 for key in ("directions", "teacher_steps", "fd_atol", "fd_rtol",
                                       "gradient_atol", "gradient_rtol", "value_atol", "value_rtol")):
        raise ValueError("Counts and tolerances must be positive")
    for gpu, cpu in (("gpu_seeds", "seeds"), ("gpu_steps", "steps"), ("gpu_ensembles", "ensembles")):
        if not set(config[gpu]) <= set(config[cpu]):
            raise ValueError("GPU cases must be a subset of CPU cases for parity checks")


def run(config, output, devices):
    validate_config(config)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if "cuda" in devices and not torch.cuda.is_available():
        raise RuntimeError("The approved GPU parity checks require an available CUDA device")
    for part in ("fixtures", "gradients"):
        (output / part).mkdir()
    write_json(output / "config.json", config)
    manifest = {"study": config["study"], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
                "architecture": platform.machine(), "python": sys.version, "numpy": np.__version__,
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if "cuda" in devices else None,
                "devices": devices, "dtype": "float64", "source_sha256": source_hashes(),
                "config_sha256": hashlib.sha256((output / "config.json").read_bytes()).hexdigest(),
                "sampling": "Independent SeedSequence child streams: students, teacher, parameters, directions",
                "status": "RUNNING"}
    write_json(output / "manifest.json", manifest)
    fixtures = {seed: make_fixture(seed, config) for seed in config["seeds"]}
    for seed, fixture in fixtures.items():
        np.savez_compressed(output / "fixtures" / f"seed-{seed}.npz", **fixture)
    cases = []
    started = time.monotonic()
    with (output / "directional-errors.csv").open("w", newline="", encoding="utf-8") as raw:
        writer = csv.DictWriter(raw, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for device in devices:
            cases.extend(scalar_cases(writer, device, config))
            seeds = config["gpu_seeds"] if device == "cuda" else config["seeds"]
            steps = config["gpu_steps"] if device == "cuda" else config["steps"]
            ensembles = config["gpu_ensembles"] if device == "cuda" else config["ensembles"]
            combinations = list(itertools.product(seeds, config["states"], steps, ensembles, config["regimes"]))
            for index, (seed, state, n, k, regime) in enumerate(combinations):
                case = nonlinear_case(writer, fixtures[seed], seed, state, n, k, regime, device, config, output)
                cases.append(case)
                with (output / "cases.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(case, allow_nan=False) + "\n")
                if index % 15 == 0 or index == len(combinations) - 1:
                    raw.flush()
                    print(f"{device}: {index + 1}/{len(combinations)} nonlinear fixtures; "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    parity = []
    cpu_cases = {c["fixture"]: c for c in cases if c["device"] == "cpu"}
    for c in cases:
        if c["device"] == "cuda" and c["setting"] == "nonlinear2d" and "cpu" in devices:
            key = c["fixture"]
            with np.load(output / "gradients" / f"cpu-{key}.npz") as a, np.load(output / "gradients" / f"cuda-{key}.npz") as b:
                parity.append({"fixture": key,
                               "gradient_agreement": close(a["full"], b["full"], config["gradient_atol"], config["gradient_rtol"]),
                               "objective_agreement": close(cpu_cases[key]["objective"], c["objective"], config["value_atol"], config["value_rtol"]),
                               "gradient_max_abs_error": float(np.max(np.abs(a["full"] - b["full"])))})
    bad_cases = [c["device"] + ":" + c["fixture"] for c in cases
                 if not all(c["checks"].values()) or not c["methods"]["full"]["passed"]
                 or (c["setting"] == "scalar" and not all(m["passed"] for m in c["methods"].values()))]
    controls = {}
    for control in CONTROLS:
        eligible = [c for c in cases if c["setting"] == "nonlinear2d"
                    and c["methods"][control]["applicable_control"]]
        detected = [c for c in eligible if not c["methods"][control]["passed"]]
        controls[control] = {"applicable_fixtures": len(eligible), "detected_fixtures": len(detected),
                             "detected": bool(detected),
                             "example": detected[0]["device"] + ":" + detected[0]["fixture"] if detected else None}
    parity_pass = all(p["gradient_agreement"] and p["objective_agreement"] for p in parity)
    status = "FAIL" if bad_cases or not parity_pass else "PASS" if all(c["detected"] for c in controls.values()) else "INCONCLUSIVE"
    summary = {"study": config["study"], "status": status, "elapsed_seconds": time.monotonic() - started,
               "counts": {"scalar": sum(c["setting"] == "scalar" for c in cases),
                          "nonlinear2d": sum(c["setting"] == "nonlinear2d" for c in cases),
                          "cpu_gpu_pairs": len(parity)},
               "failed_full_fixtures": bad_cases, "controls": controls, "cpu_gpu_parity": parity,
               "cases": cases,
               "claim_limit": "Correct differentiation of tested finite sample objectives; no population unbiasedness, convergence, or quality claim."}
    write_json(output / "summary.json", summary)
    manifest.update(status=status, elapsed_seconds=summary["elapsed_seconds"])
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k: summary[k] for k in ("status", "counts", "controls", "elapsed_seconds")}, indent=2), flush=True)
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; existing study evidence must not be overwritten")
    args.output.mkdir(parents=True)
    try:
        return run(json.loads(args.config.read_text(encoding="utf-8")), args.output, args.devices)
    except Exception:
        (args.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
