"""Laplace extension, separate from the immutable historical Gaussian E0 study."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch

from . import laplace_cases as cases
from .verification_cases import CONTROLS, PARAMETERS, frozen_rollout, rollout
from .verify import (RAW_FIELDS, applicable, close, epsilon_grid, make_fixture,
                     record_curve, source_hashes, validate_config, write_json)


def run(config, output, devices):
    validate_config(config)
    if config["mixtures"] != [0., .5, 1.]:
        raise ValueError("The regression requires mixture endpoints and midpoint")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (output/"gradients").mkdir()
    (output/"fixtures").mkdir()
    write_json(output/"config.json", config)
    write_json(output/"manifest.json", {"source_sha256": source_hashes(), "torch": torch.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "devices": devices,
        "job_id": os.environ.get("SLURM_JOB_ID"), "kernel": "exp(-euclidean-distance/ell)",
        "ell": 1., "coincidence_convention": "zero subgradient, not classical differentiability"})
    fixtures = {seed: make_fixture(seed, config) for seed in config["seeds"]}
    for seed, fixture in fixtures.items():
        # Distinct roles from all historical streams, unequal target counts.
        rng_t, rng_g = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 91231]).spawn(2)]
        fixture["teacher"] = frozen_rollout(torch.tensor(rng_t.normal(size=(2,2,2))),
                                           torch.tensor(fixture["contexts"]), 64).numpy()
        fixture["ground_truth"] = rng_g.normal(size=(2,3,2)) + np.array([[-1,.5],[1,-.5]])[:,None]
        np.savez(output/"fixtures"/f"seed-{seed}.npz", **fixture)
    results, parity, started = [], [], time.monotonic()
    with (output/"directional-errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for device in devices:
            tensor = lambda a: torch.tensor(a, device=device, dtype=torch.float64)
            prefix = "gpu_" if device == "cuda" else ""
            combinations = list(itertools.product(config[prefix+"seeds"], config["states"],
                config[prefix+"steps"], config[prefix+"ensembles"], config["regimes"], config["mixtures"]))
            for index, (seed,state,n,k,regime,rho) in enumerate(combinations):
                f = fixtures[seed]
                theta, x, c, d = tensor(f[state]), tensor(f["initial"][:,:k]), tensor(f["contexts"]), tensor(f["directions"])
                targets = {name: tensor(f[name]) for name in ("teacher","ground_truth")}
                loss, full = cases.gradient(theta,x,targets,c,n,regime,rho)
                adjoint = cases.adjoint(theta,x,targets,c,n,regime,rho)
                with torch.no_grad():
                    endpoint = rollout(theta,x,c,n)
                    analytic = cases.covector(endpoint,targets,c,regime,rho)
                    independent = cases.loop_value(endpoint,targets,c,regime,rho)
                    zero = rollout(tensor(f["zero"]),x,c,n)
                    frozen = frozen_rollout(x,c,n)
                z = endpoint.detach().requires_grad_(True)
                terminal, = torch.autograd.grad(cases.terminal(z,targets,c,regime,rho), z)
                rel = epsilon_grid(config)
                eps = rel*max(1.,float(theta.norm()))
                offsets = d[:,None]*theta.new_tensor(eps)[None,:,None]
                inputs = torch.stack((theta+offsets,theta-offsets),2).reshape(-1,PARAMETERS)
                with torch.no_grad():
                    values = torch.vmap(lambda p: cases.objective(p,x,targets,c,n,regime,rho))(inputs).reshape(len(d),len(eps),2)
                    fd = ((values[...,0]-values[...,1])/theta.new_tensor(2*eps)[None]).cpu().numpy()
                    sequential = cases.objective(inputs[len(eps)],x,targets,c,n,regime,rho)
                name = f"s{seed}-{state}-n{n}-k{k}-{regime}-rho{rho:g}"
                meta = dict(fixture=name,setting="laplace2d",device=device,seed=seed,steps=n,ensemble=k,state=state,regime=regime)
                checks = {"adjoint": close(full.cpu(),adjoint.cpu(),config["gradient_atol"],config["gradient_rtol"]),
                    "terminal_covector": close(terminal.cpu(),analytic.cpu(),config["gradient_atol"],config["gradient_rtol"]),
                    "independent_value": close(float(loss),independent,config["value_atol"],config["value_rtol"]),
                    "zero_correction": torch.equal(zero,frozen),
                    "batched_value": close(float(sequential),float(values.reshape(-1)[len(eps)]),config["value_atol"],config["value_rtol"])}
                grads, outcomes = {"full": full}, {}
                for control in CONTROLS:
                    control_loss, grads[control] = cases.gradient(theta,x,targets,c,n,regime,rho,control)
                    checks[control+"_forward_equal"] = torch.equal(loss,control_loss)
                for method, grad in grads.items():
                    outcomes[method] = record_curve(writer,meta,method,(d@grad).cpu().numpy(),fd,rel,eps,config)
                    outcomes[method]["applicable_control"] = method != "full" and applicable(method,n,state,regime)
                record = {**meta,"rho":rho,"objective":float(loss),"checks":checks,"methods":outcomes,
                          "adjoint_max_abs_error":float((full-adjoint).abs().max())}
                results.append(record)
                np.savez_compressed(output/"gradients"/f"{device}-{name}.npz",full=full.cpu().numpy(),adjoint=adjoint.cpu().numpy())
                with (output/"cases.jsonl").open("a") as raw:
                    raw.write(json.dumps(record,allow_nan=False)+"\n")
                if index%60 == 0:
                    stream.flush()
                    print(f"{device}: {index+1}/{len(combinations)}",flush=True)
    if "cpu" in devices and "cuda" in devices:
        cpu = {r["fixture"]:r for r in results if r["device"] == "cpu"}
        for r in results:
            if r["device"] == "cuda":
                name = r["fixture"]
                a,b = [np.load(output/"gradients"/f"{device}-{name}.npz")["full"] for device in ("cpu","cuda")]
                parity.append({"fixture":name,"passed":close(a,b,config["gradient_atol"],config["gradient_rtol"]) and
                    close(cpu[name]["objective"],r["objective"],config["value_atol"],config["value_rtol"]),
                    "max_abs_error":float(np.max(np.abs(a-b)))})
    failures = [r["device"]+":"+r["fixture"] for r in results if not all(r["checks"].values()) or not r["methods"]["full"]["passed"]]
    controls = {}
    for control in CONTROLS:
        eligible = [r for r in results if r["methods"][control]["applicable_control"]]
        detected = [r for r in eligible if not r["methods"][control]["passed"]]
        controls[control] = {"applicable":len(eligible),"detected":len(detected)}
    status = "FAIL" if failures or not all(p["passed"] for p in parity) else "PASS" if all(c["detected"] for c in controls.values()) else "INCONCLUSIVE"
    summary = dict(status=status,counts={"fixtures":len(results),"cpu_gpu_pairs":len(parity)},
        failed_full_fixtures=failures,controls=controls,cpu_gpu_parity=parity,cases=results,elapsed_seconds=time.monotonic()-started)
    write_json(output/"summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("cases","cpu_gpu_parity")}),flush=True)
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--devices",nargs="+",choices=["cpu","cuda"],default=["cpu","cuda"])
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        code = run(json.loads(args.config.read_text()),args.output,args.devices)
    except BaseException:
        (args.output/"failure.txt").write_text(traceback.format_exc())
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    main()
