"""Independently audit saved CSV acceptance decisions, counts, and source hashes.

This reads recorded evidence only; it performs no rollout, training, or evaluation.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


def audit(run, source):
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    cases = {(c["device"], c["fixture"]): c for c in summary["cases"]}
    if len(cases) != len(summary["cases"]):
        raise AssertionError("Duplicate fixtures in summary")
    verified, rows_count, seen = {}, 0, set()
    with (run / "directional-errors.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        key = lambda r: (r["device"], r["fixture"], r["method"], int(r["direction"]))
        for group_key, records in itertools.groupby(reader, key=key):
            if group_key in seen:
                raise AssertionError(f"Duplicate CSV direction group: {group_key}")
            seen.add(group_key)
            records = list(records)
            if len(records) != config["epsilon_count"]:
                raise AssertionError(f"Incomplete epsilon sweep: {group_key}")
            rows_count += len(records)
            flags = []
            previous_eps = float("inf")
            for row in records:
                g, fd, e = [float(row[k]) for k in ("derivative", "finite_difference", "epsilon_relative")]
                error = abs(g - fd)
                tolerance = config["fd_atol"] + config["fd_rtol"] * max(abs(g), abs(fd))
                if not all(math.isfinite(v) for v in (g, fd, e, error, tolerance)) or not 0 < e < previous_eps:
                    raise AssertionError(f"Nonfinite values or unordered perturbations: {group_key}")
                previous_eps = e
                for field, expected in (("absolute_error", error), ("tolerance", tolerance)):
                    if not math.isclose(float(row[field]), expected, rel_tol=1e-13, abs_tol=1e-25):
                        raise AssertionError(f"Recorded {field} differs from recomputation: {group_key}")
                flag = error <= tolerance
                if flag != (row["within_tolerance"] == "True"):
                    raise AssertionError(f"Incorrect per-epsilon pass flag: {group_key}")
                flags.append(flag)
            width = config["consecutive_passes"]
            passed = any(all(flags[i:i + width]) for i in range(len(flags) - width + 1))
            verified.setdefault(group_key[:3], []).append(passed)
    for (device, fixture), case in cases.items():
        for method, expected in case["methods"].items():
            flags = verified[(device, fixture, method)]
            if len(flags) != expected["directions"] or sum(flags) != expected["passed_directions"] or all(flags) != expected["passed"]:
                raise AssertionError(f"CSV/summary acceptance disagreement: {device}:{fixture}:{method}")
    if len(verified) != sum(len(c["methods"]) for c in cases.values()):
        raise AssertionError("Unexpected method groups in CSV")
    for device in manifest["devices"]:
        prefix = "gpu_" if device == "cuda" else ""
        expected = len(config[prefix + "seeds"]) * len(config[prefix + "steps"]) * len(config[prefix + "ensembles"]) * len(config["states"]) * len(config["regimes"])
        observed = sum(c["device"] == device and c["setting"] == "nonlinear2d" for c in cases.values())
        if observed != expected:
            raise AssertionError(f"Wrong {device} fixture count: {observed} != {expected}")
    for relative, digest in manifest["source_sha256"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Source hash mismatch: {relative}")
    if hashlib.sha256((run / "config.json").read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise AssertionError("Configuration hash mismatch")
    result = {"status": "PASS", "raw_rows": rows_count, "direction_groups": len(seen),
              "method_groups": len(verified), "fixtures": len(cases),
              "source_files_verified": len(manifest["source_sha256"]),
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "raw_csv_sha256": hashlib.sha256((run / "directional-errors.csv").read_bytes()).hexdigest(),
              "summary_sha256": hashlib.sha256((run / "summary.json").read_bytes()).hexdigest(),
              "scope": "Saved evidence counts, numeric row arithmetic, consecutive-epsilon acceptance, config and source hashes; no rerun."}
    (run / "evidence-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.source)


if __name__ == "__main__":
    main()
