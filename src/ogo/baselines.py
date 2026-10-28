"""Published optimization rules; backbone/sampler adaptations are explicit."""
import math
import torch
from torch.nn import functional as F

METHODS = ('ogo', 'draft_k', 'ddpo_is', 'flow_grpo', 'lambda_grpo', 'offline_d3po')


def advantages(rewards, group_size=None):
    value = rewards.detach()
    if group_size is not None:
        if group_size < 2 or value.numel() % group_size:
            raise ValueError('Whole reward groups required')
        value = value.reshape(-1, group_size)
    return ((value-value.mean(-1, keepdim=True))/(value.std(-1, unbiased=False, keepdim=True)+1e-8)).reshape(-1)


def ppo_loss(logp, old_logp, advantage, clip):
    ratio = (logp-old_logp.detach()).exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError('Nonfinite PPO ratio')
    return -torch.minimum(ratio*advantage.detach(), ratio.clamp(1-clip, 1+clip)*advantage.detach()).mean()


def lambda_statistics(mean, old_mean, std):
    shift = ((mean-old_mean.detach())/std).flatten(1)
    center = shift.square().mean(1)
    return center, center/shift.shape[1]


def lambda_loss(logp, old_logp, mean, old_mean, std, advantage, *, clip, retained_fraction, late_steps, floor):
    center, variance = (v.detach() for v in lambda_statistics(mean, old_mean, std))
    nu = clip**2  # Main text Eq. 3.11, not the alternative appendix approximation.
    z = (logp-old_logp.detach()+.5*center)/(variance+floor).sqrt()
    ratio = (math.sqrt(nu)*z-.5*nu).exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError('Nonfinite LambdaNorm-T proposal')
    tau = -math.log(retained_fraction)/late_steps
    weight = torch.minimum(center.new_tensor(1.), (tau/(center.mean()+floor)).sqrt()).detach()
    loss = -weight*torch.minimum(ratio*advantage.detach(), ratio.clamp(1-clip, 1+clip)*advantage.detach()).mean()
    return loss, {'center': center.mean(), 'variance': variance.mean(), 'weight': weight}


def trajectory_dpo_loss(logp, reference_logp, preference, beta, q_clip=.1):
    """D3PO Eq. 12 applied per corresponding transition, then averaged.

    Dataset construction fixes pairs with the same class and initial noise;
    only their stochastic transition noises differ. winner is fixed feedback.
    """
    ratio = (logp-reference_logp.detach()).reshape(-1, 2)
    ratio = ratio.clamp(math.log(1-q_clip), math.log(1+q_clip))
    if preference.shape != (len(ratio),):
        raise ValueError('One frozen preference per trajectory pair required')
    return -F.logsigmoid(beta*preference.detach().to(ratio)*(ratio[:, 0]-ratio[:, 1])).mean()
