from functools import singledispatch
import torch
import numpy as np
import mcubes
from libzhifan.geometry import SimpleMesh


@singledispatch
def make_mesh(f: np.ndarray, d=0.0):
    assert f.ndim == 3
    vertices, triangles = mcubes.marching_cubes(f, d)
    return SimpleMesh(verts=vertices, faces=triangles)

@make_mesh.register
def _(f: torch.Tensor, d=0.0):
    assert f.ndim == 3
    return make_mesh(f.detach().cpu().numpy(), d)