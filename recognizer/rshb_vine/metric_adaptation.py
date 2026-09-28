"""Bounded SigLIP vision adaptation primitives; this module grants no data admission."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Zero-initialized residual, with an exact inference merge into the base layer."""

    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0 or alpha <= 0:
            raise ValueError('Expected Linear and positive rank/alpha')
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale

    def merged(self):
        layer = nn.Linear(self.base.in_features, self.base.out_features,
                          bias=self.base.bias is not None,
                          device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            layer.weight.copy_(self.base.weight + (self.b @ self.a) * self.scale)
            if layer.bias is not None:
                layer.bias.copy_(self.base.bias)
        return layer.requires_grad_(False).eval()


def install_vision_lora(model):
    """Fixed first recipe: q/v, last four blocks, rank8/alpha16; text stays frozen."""
    layers = model.vision_model.encoder.layers
    if len(layers) < 4:
        raise ValueError('Recipe requires at least four vision blocks')
    targets = [(i, name, getattr(layers[i].self_attn, name))
               for i in range(len(layers) - 4, len(layers))
               for name in ('q_proj', 'v_proj')]
    if any(not isinstance(layer, nn.Linear) for _, _, layer in targets):
        raise ValueError('Unexpected or already adapted q/v projection')
    model.requires_grad_(False)
    names = []
    for i, name, layer in targets:
        setattr(layers[i].self_attn, name, LoRALinear(layer))
        names.append(f'vision_model.encoder.layers.{i}.self_attn.{name}')
    return {'rank': 8, 'alpha': 16, 'targets': names,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def merge_vision_lora(model):
    """Mutate an adapted model into one frozen encoder; no extra inference stage."""
    count = sum(isinstance(getattr(layer.self_attn, name), LoRALinear)
                for layer in model.vision_model.encoder.layers for name in ('q_proj', 'v_proj'))
    if count != 8:
        raise ValueError('Expected exactly eight adapted projections')
    merged = []
    for i, layer in enumerate(model.vision_model.encoder.layers):
        for name in ('q_proj', 'v_proj'):
            projection = getattr(layer.self_attn, name)
            if isinstance(projection, LoRALinear):
                setattr(layer.self_attn, name, projection.merged())
                merged.append(f'vision_model.encoder.layers.{i}.self_attn.{name}')
    model.requires_grad_(False).eval()
    return merged


def multi_positive_loss(query, reference, query_slugs, reference_slugs,
                        *, temperature=0.07, excluded_pairs=None):
    """Symmetric cross-view supervised contrastive loss.

    Every matching SKU is a positive, including repeated views. Explicit ambiguous
    different-SKU pairs may be excluded from negatives; they never become positives.
    Inputs are normalized here, through the gradient, with no projection head.
    """
    if (query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]
            or len(query_slugs) != len(query) or len(reference_slugs) != len(reference)
            or not len(query) or not len(reference)):
        raise ValueError('Invalid embedding/identity batch shapes')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if any(not isinstance(s, str) or not s for s in [*query_slugs, *reference_slugs]):
        raise ValueError('Exact nonempty identities required')
    if not torch.isfinite(query).all() or not torch.isfinite(reference).all():
        raise ValueError('Non-finite embeddings')
    if (query.norm(dim=1) <= 1e-8).any() or (reference.norm(dim=1) <= 1e-8).any():
        raise ValueError('Zero embeddings')
    positives = torch.tensor([[q == r for r in reference_slugs] for q in query_slugs],
                             dtype=torch.bool, device=query.device)
    allowed = torch.ones_like(positives)
    if excluded_pairs is not None:
        if excluded_pairs.shape != positives.shape or excluded_pairs.dtype != torch.bool:
            raise ValueError('Exclusion mask must be boolean query x reference')
        excluded_pairs = excluded_pairs.to(query.device)
        if (excluded_pairs & positives).any():
            raise ValueError('Cannot exclude a same-SKU positive')
        allowed &= ~excluded_pairs
    negatives = allowed & ~positives
    if any(not mask.any(dim=axis).all() for mask in (positives, negatives) for axis in (0, 1)):
        raise ValueError('Every anchor needs a positive and a permitted negative')
    logits = F.normalize(query.float(), dim=1) @ F.normalize(reference.float(), dim=1).T
    logits = (logits / temperature).masked_fill(~allowed, -torch.inf)

    def directional(scores, positive):
        log_probs = scores - torch.logsumexp(scores, dim=1, keepdim=True)
        return -(log_probs.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1)).mean()

    return (directional(logits, positives) + directional(logits.T, positives.T)) / 2
