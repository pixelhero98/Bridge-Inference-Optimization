"""Independent equations, derivative checks and exact optimizer replay."""
import copy
import math
from pathlib import Path

import torch
from torch import nn

from .baselines import advantages, ppo_loss, lambda_statistics, lambda_loss, trajectory_dpo_loss
from .policy import euler, gaussian_log_prob, install_lora, reference_policy, transition_mean
from .study_io import write


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def verify(output, thresholds):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(561)
    results = []
    def check(name, fn):
        fn()
        results.append({'name': name, 'passed': True})
    def close(a, b):
        torch.testing.assert_close(a, b, rtol=thresholds['fd_rtol'], atol=thresholds['fd_atol'])
    def fd(fn, x):
        step = 1e-5
        return (fn(x+step)-fn(x-step))/(2*step)

    def gaussian():
        x, mu = torch.randn(3, 4), torch.randn(3, 4, requires_grad=True)
        std = .37
        actual = gaussian_log_prob(x, mu, std)
        expected = torch.distributions.Normal(mu, std).log_prob(x).mean(1)
        close(actual, expected)
        grad = torch.autograd.grad(actual.sum(), mu)[0]
        close(grad, (x-mu.detach())/(4*std**2))
        try:
            gaussian_log_prob(x, mu, 0.)
        except ValueError:
            pass
        else:
            raise AssertionError('Deterministic likelihood accepted')
    check('Gaussian transition density and analytic score; zero variance rejected', gaussian)

    def sde():
        x, v = torch.randn(3, 4), torch.randn(3, 4)
        for n in (16, 32):
            for k in (0, 1, n-1):
                t = k/n
                m, std = transition_mean(x, v, t, n, 'sit256', .7)
                sigma = 1-t
                # Independent evaluation in the official descending-sigma convention.
                dt = -1/n
                diffusion = math.sqrt(sigma/(1-(1-1/n if sigma == 1 else sigma)))*.7
                official = x*(1+diffusion**2/(2*sigma)*dt)+(-v)*(1+diffusion**2*(1-sigma)/(2*sigma))*dt
                close(m, official)
                assert abs(std-diffusion*math.sqrt(-dt)) < 1e-14
                for backbone in ('sit256', 'edm64'):
                    m0, s0 = transition_mean(x, v, t, n, backbone, 0.)
                    close(m0, x+v/n)
                    assert s0 == 0
            t = .3
            a, b, speed = (1+80*t)/81, 80*(1-t)/81, 80/81
            denoised = torch.randn_like(x)
            velocity = (denoised-x)/(1-t)
            score1 = (a*velocity/speed-x)/b
            score2 = (a*denoised-x)/b**2
            close(score1, score2)
    check('SiT Flow-GRPO forward parity; EDM score conversion; zero-noise Euler equivalence', sde)

    def ppo():
        x = torch.tensor([.02, -.03, .5, -.5], requires_grad=True)
        adv = torch.tensor([1., -1., 2., -2.])
        expected = torch.maximum(-adv*x.exp(), -adv*x.exp().clamp(.9, 1.1)).mean()
        actual = ppo_loss(x, torch.zeros_like(x), adv, .1)
        close(actual, expected)
        close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])
        assert torch.equal(advantages(torch.ones(8), 4), torch.zeros(8))
    check('DDPO/GRPO clipping forward and gradient parity; zero reward variance', ppo)

    def lambdas():
        old = torch.randn(4, 5)
        new = old+.03*torch.randn(4,5)
        c, v = lambda_statistics(new, old, .2)
        close(c, ((new-old)/.2).square().sum(1)/5)
        close(v, c/5)
        y = torch.tensor([.001, -.001, .002, -.002], requires_grad=True)
        adv = torch.tensor([1., -1., 1., -1.])
        loss, _ = lambda_loss(y, y.detach()*0, new, old, .2, adv,
                             clip=.1, retained_fraction=.99, late_steps=8, floor=1e-8)
        z = (y+.5*c.detach())/(v.detach()+1e-8).sqrt()
        ratio = (.1*z-.005).exp()
        w = min(1., math.sqrt((-math.log(.99)/8)/(float(c.mean())+1e-8)))
        expected = -w*torch.minimum(ratio*adv, ratio.clamp(.9,1.1)*adv).mean()
        close(loss, expected)
        close(torch.autograd.grad(loss, y)[0], torch.autograd.grad(expected, y)[0])
        same, _ = lambda_loss(y*0, y.detach()*0, old, old, .2, adv,
                             clip=.1, retained_fraction=.99, late_steps=8, floor=1e-8)
        assert torch.isfinite(same)
    check('LambdaNorm-T two-scale law, detached statistics, published loss/gradient and zero shift', lambdas)

    def dpo():
        x = torch.tensor([.01,-.02,.4,-.4], requires_grad=True)
        pref = torch.tensor([1., -1.])
        actual = trajectory_dpo_loss(x, torch.zeros_like(x), pref, .1, .1)
        logratio = torch.exp(x).clamp(.9,1.1).log().reshape(2,2)
        expected = -torch.log(torch.sigmoid(.1*pref*(logratio[:,0]-logratio[:,1]))).mean()
        close(actual, expected)
        close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])
    check('D3PO released ratio clipping and preference gradient parity', dpo)

    def truncation():
        theta = torch.tensor(.2, requires_grad=True)
        x = torch.ones(2,1)
        labels = torch.zeros(2, dtype=torch.long)
        endpoint = euler(lambda z,t,c: theta*z, x, labels, 16, backward_steps=4, rematerialize=False)
        close(endpoint, x*(1+theta/16)**16)
        gradient = torch.autograd.grad(endpoint.sum(), theta)[0]
        close(gradient, 2*4/16*(1+theta.detach()/16)**15)
        frozen = (x*(1+theta.detach()/16)**12).detach()
        exact = lambda p: (frozen*(1+p/16)**4).sum()
        close(gradient, fd(exact, theta.detach()))
    check('DRaFT-K complete forward, last-K gradient, frozen-prefix finite differences', truncation)

    def lora():
        m = nn.Sequential(nn.Linear(3,4), nn.Tanh(), nn.Linear(4,2)).double()
        x = torch.randn(2,3)
        before = m(x).detach()
        install_lora(m, 2, 'all_linear_conv', 2.)
        close(m(x), before)
        trainable = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=.003)
        def step():
            opt.zero_grad();m(x).square().mean().backward();opt.step()
        step()
        s, o = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
        step(); final, optim = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
        m.load_state_dict(s);opt.load_state_dict(o);step()
        assert equal_tree(m.state_dict(), final) and equal_tree(opt.state_dict(), optim)
        with reference_policy(m):
            close(m(x), before)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    check('Low-rank parameterization, frozen-reference disabling and exact AdamW resume', lora)
    torch.set_default_dtype(torch.float32)
    value = {'passed': True, 'checks': results, 'thresholds': thresholds}
    write(Path(output)/'numerical-verification.json', value)
    return value
