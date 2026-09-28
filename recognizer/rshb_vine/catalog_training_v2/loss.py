"""Explicit real and canonical objectives over the stage5 tensor column layout."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _directional(scores, positive):
    if scores.numel() == 0:
        return scores.sum()*0
    if not bool(positive.any(dim=1).all()):
        raise ValueError("Loss anchor has no positive column")
    log_probs = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    return -(log_probs.masked_fill(~positive, 0).sum(dim=1)/positive.sum(dim=1)).mean()


def combined_loss(features, batch, *, temperature, real_global_weight=1.0,
                  real_local_weight=0.25, canonical_weight=0.0625):
    """Only declared negative columns contribute; branches never cross-compare."""
    step = batch["step"]
    n, m, k = batch["real_count"], batch["canonical_count"], batch["mined_count"]
    if features.shape[0] != 2*n+3*m+k or features.ndim != 2:
        raise ValueError("Embedding layout shape differs from image layout")
    q = F.normalize(features[:n].float(), dim=1)
    cq = F.normalize(features[n:n+m].float(), dim=1)
    r = F.normalize(features[n+m:2*n+m].float(), dim=1)
    cp = F.normalize(features[2*n+m:2*n+2*m].float(), dim=1)
    mined = F.normalize(features[2*n+2*m:2*n+2*m+k].float(), dim=1)
    cn = F.normalize(features[2*n+2*m+k:].float(), dim=1)
    if n:
        excluded = torch.tensor(step["real_excluded_base"], dtype=torch.bool, device=features.device)
        if excluded.shape != (n,n) or bool(excluded.diag().any()):
            raise ValueError("Invalid real base exclusion mask")
        # Only the explicitly scheduled query/reference diagonal is a positive.
        # Shared acceptable IDs across other rows are unknown, not extra labels.
        positive = torch.eye(n, dtype=torch.bool, device=features.device)
        if bool((positive & excluded).any()):
            raise ValueError("Positive real pair excluded")
        base = (q @ r.T/temperature).masked_fill(excluded,-torch.inf)
        owners = [i for i,x in enumerate(step["real"]) for _ in x["mined"]]
        if len(owners) != k:
            raise ValueError("Mined ownership mismatch")
        if k:
            own = torch.tensor([[owner==i for owner in owners] for i in range(n)],
                               dtype=torch.bool, device=features.device)
            extra = (q @ mined.T/temperature).masked_fill(~own,-torch.inf)
            forward = _directional(torch.cat((base,extra),dim=1),
                                   torch.cat((positive,torch.zeros_like(own)),dim=1))
        else:
            forward = _directional(base,positive)
        reverse = _directional(base.T,positive.T)
        real_global = (forward+reverse)/2
        local = []
        columns = torch.cat((r,mined),dim=0)
        for i,negative_columns in enumerate(step["real_local_columns"]):
            if not negative_columns:
                continue
            pos = (q[i] @ r[i])/temperature
            neg = (q[i] @ columns[negative_columns].T)/temperature
            local.append(torch.logsumexp(torch.cat((pos[None],neg)),dim=0)-pos)
        real_local = torch.stack(local).mean() if local else features.sum()*0
    else:
        real_global = features.sum()*0
        real_local = features.sum()*0
    if m:
        positive_score = (cq*cp).sum(dim=1)/temperature
        negative_score = (cq*cn).sum(dim=1)/temperature
        weights = features.new_tensor([x["product_weight"] for x in step["canonical"]]).float()
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("Nonpositive canonical product weight")
        # Fixed four-slot denominator; the final one-item batch has three
        # explicit empty slots rather than a fourfold larger gradient.
        canonical = (F.softplus(negative_score-positive_score)*weights).sum()/4
    else:
        canonical = features.sum()*0
    total = real_global_weight*real_global + real_local_weight*real_local + canonical_weight*canonical
    if not bool(torch.isfinite(total)):
        raise ValueError("Combined loss nonfinite")
    return {"total":total,"real_global":real_global,"real_local":real_local,"canonical":canonical}
