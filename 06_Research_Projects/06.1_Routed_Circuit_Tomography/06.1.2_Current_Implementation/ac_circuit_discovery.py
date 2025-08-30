import numpy as np
import torch
import torch.nn.functional as F
import math


def ac_maps(q, k):
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    t, d = q.shape
    scale = 1.0 / math.sqrt(d)
    a_push = F.softmax(q @ k.t() * scale, dim=-1)
    a_pull = F.softmax(k @ q.t() * scale, dim=-1)
    return a_push.detach().cpu().numpy(), (a_push * a_pull.t()).detach().cpu().numpy()


def ac_score(a_push, a_res):
    r = a_res
    s = r.sum() + 1e-8
    return float(r.max() / s)