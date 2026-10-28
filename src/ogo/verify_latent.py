"""Latent-only Laplace FD and independently assembled discrete-adjoint regression."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .interfaces import TargetSource
from .objectives import FeatureKernel, LaplaceKernel, conditional_loss
from .synthetic import utility
from .verification_cases import rollout, euler_step
from .verify import make_fixture


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results, parity = [], {}
    for device in ("cpu", "cuda"):
        combinations = itertools.product(range(3) if device == "cpu" else [0], ("zero", "nonzero"),
            (1, 2, 4, 8, 16) if device == "cpu" else [8], (2, 4, 8) if device == "cpu" else [4], (0., .5, 1.))
        for seed, state, steps, k, rho in combinations:
            f = make_fixture(seed, {"ensembles": [8], "directions": 10, "teacher_steps": 64})
            tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
            theta, x, contexts, directions = (tensor(f[state]), tensor(f["initial"][:, :k]), tensor(f["contexts"]), tensor(f["directions"]))
            rng = np.random.default_rng(seed + 5731)
            targets = {"teacher": tensor(rng.normal(size=(2, 2, 2))), "ground_truth": tensor(rng.normal(size=(2, 3, 2)))}
            weights = {"teacher": 1-rho, "ground_truth": rho}
            branch = FeatureKernel("latent", lambda a, c: a, LaplaceKernel(1.), .5)
            def terminal(z):
                return torch.stack([conditional_loss(z[g], None, c, utility, (branch,),
                    target_sources={n: TargetSource({"latent": y[g]}, ()) for n,y in targets.items()},
                    target_weights=weights)[0] for g,c in enumerate(contexts)]).mean()
            def objective(p): return terminal(rollout(p, x, contexts, steps))
            p = theta.clone().requires_grad_(True)
            loss = objective(p)
            full, = torch.autograd.grad(loss, p)
            with torch.no_grad():
                z, states = rollout(theta, x, contexts, steps, return_states=True)
                desired = torch.stack((contexts, -.5*contexts), -1)
                force = z-desired[:, None]
                def dk(delta):
                    r = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                    return -torch.exp(-r)*delta/torch.where(r > 0, r, 1.)
                mask = 1-torch.eye(k, device=device, dtype=torch.float64)
                matching = (dk(z[:, :, None]-z[:, None])*mask[None, :, :, None]).sum(-2)/(k-1)
                for name,w in weights.items(): matching -= w*dk(z[:, :, None]-targets[name][:, None]).mean(-2)
                a = (force+.5*matching)/(2*k)
            adjoint = torch.zeros_like(theta)
            for i in reversed(range(steps)):
                state_tensor = states[i].detach().requires_grad_(True)
                updated = euler_step(p, state_tensor, i/steps, 1/steps, contexts)
                a, contribution = torch.autograd.grad(updated, (state_tensor, p), grad_outputs=a)
                adjoint += contribution
            eps = tensor(np.logspace(-1, -8, 15))*max(1., float(theta.norm()))
            offsets = directions[:, None]*eps[None, :, None]
            with torch.no_grad():
                values = torch.vmap(objective)(torch.stack((theta+offsets, theta-offsets), 2).reshape(-1, len(theta))).reshape(10, 15, 2)
            fd = (values[..., 0]-values[..., 1])/(2*eps[None])
            projections = directions @ full
            ok = (fd-projections[:, None]).abs() <= 1e-9+1e-6*torch.maximum(fd.abs(), projections[:, None].abs())
            passed = bool((ok[:, :-2] & ok[:, 1:-1] & ok[:, 2:]).any(-1).all())
            agree = bool(((full-adjoint).abs() <= 1e-10+1e-8*torch.maximum(full.abs(), adjoint.abs())).all())
            key = f"s{seed}-{state}-n{steps}-k{k}-rho{rho}"
            if device == "cpu": parity[key] = full.cpu()
            else: agree &= bool(((full.cpu()-parity[key]).abs() <= 1e-10+1e-8*full.cpu().abs()).all())
            result = {"fixture": key, "device": device, "fd": passed, "adjoint": agree,
                "max_adjoint_error": float((full-adjoint).abs().max()), "eps": eps.cpu().tolist(),
                "projections": projections.cpu().tolist(), "finite_differences": fd.cpu().tolist(),
                "objective": float(loss.detach())}
            results.append(result)
            with (output / "raw.jsonl").open("a") as stream: stream.write(json.dumps(result)+"\n")
    summary = {"status": "PASS" if all(r["fd"] and r["adjoint"] for r in results) else "FAIL",
        "fixtures": len(results), "failed": [r for r in results if not r["fd"] or not r["adjoint"]],
        "thresholds": {"fd_atol": 1e-9, "fd_rtol": 1e-6, "consecutive": 3, "adjoint_atol": 1e-10, "adjoint_rtol": 1e-8}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if summary["status"] != "PASS": raise AssertionError(summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
