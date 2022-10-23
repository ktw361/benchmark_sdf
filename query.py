from collections import namedtuple
import numpy as np
import torch
import trimesh
import torch.nn.functional as F
import einops
from sdf import SDF
from libzhifan.geometry import visualize
from libzhifan.numeric import check_shape, numpize


""" Compute distances """

def sdf_sample_helper(phi, points):
    """
    Args:
        phi: (B, D, D, D) e.g. D=32
        points: (B, N, 3)
    
    Returns:
        samples: (B, N) of distance
    """
    phi = einops.repeat(phi, 'b x y z -> b c x y z', c=1)
    grid = einops.repeat(points, 'b n d3 -> b n d1 d2 d3', d1=1, d2=1)
    vals = F.grid_sample(phi, grid)  # (B, 1, n, 1, 1)
    return vals.view(phi.size(0), points.size(1))


def standardize(verts: torch.Tensor):
    """
    Args:
        verts: (B, V, 3)
    """
    scale_factor = 0.2
    ub = torch.max(verts, dim=1, keepdim=True).values  # (B, 1, 3)
    lb = torch.min(verts, dim=1, keepdim=True).values
    center = (ub + lb) / 2
    scale_xyz = (ub - lb) * (1 + scale_factor) * 0.5
    scale = torch.max(scale_xyz, dim=-1, keepdim=True).values  # (B,)
    v_scaled = (verts - center) / scale.view(verts.size(0), 1, 1)
    return v_scaled, center, scale
    

def calc_scaled_sdf(verts: torch.Tensor, 
                    faces: torch.Tensor,
                    grid_size=32) -> torch.Tensor:
    """
    Args:
        verts: (B, V, 3)
        faces: (F, 3)
    
    Returns:
        field: (B, G, G, G)
        center: (B, 3)
        scale: (B)
    """
    check_shape(verts, (-1, -1, 3))
    check_shape(faces, (-1, 3))
    v_scaled, center, scale = standardize(verts)
    phi = SDF()(faces, v_scaled, grid_size=grid_size)
    return phi, center, scale


def scene_sdf_dist(verts: torch.Tensor, 
                   faces: torch.Tensor, 
                   queries: torch.Tensor,
                   grid_size=32) -> torch.Tensor:
    """
    Args:
        verts: (B, V, 3) cuda tensor
        faces: (F, 3) cuda tensor
        queries: (B, N, 3)
            TODO or list of B x (N_b, 3) queries
    """
    phi, center, scale = calc_scaled_sdf(verts, faces, grid_size)
    q_scaled = (queries - center) / scale
    samples = sdf_sample_helper(phi, q_scaled) * scale
    return samples


def trimesh_sdf(mesh: trimesh.Trimesh, queries: torch.Tensor) -> torch.Tensor:
    """
    Args:
        query: shape (N, 3)
    
    Returns: (N,)
    """
    queries = numpize(queries)
    d = trimesh.proximity.ProximityQuery(mesh).signed_distance(queries)
    return d


""" Sampling """

def sample_surface(mesh, nums=3):
    """
    Returns:
        (nums, 3)
    """
    pts = trimesh.sample.sample_surface(mesh, nums)[0]
    return torch.as_tensor(pts).float()


def sample_bounding_sphere(mesh, nums=3):
    """ => (nums, 3) """
    pts = trimesh.sample.sample_surface(
        mesh.bounding_sphere, nums)[0]
    return torch.as_tensor(pts).float()


""" Evalutating (quantitive & visually) """

def test_surface_distance(mesh, method):
    """

    Args:
        mesh (_type_): _description_
        method (Callable): 
            d = method(mesh, points)
    """
    pts = sample_surface(mesh, 100)
    d = method(mesh, pts)
    max_d = np.max(d)
    edge = np.power(mesh.bounding_sphere.volume, 1/3)
    print(f"max distance: {max_d:.3f} ({max_d / edge:.3f}%)")


Selected = namedtuple("Selected", "points scene")
def select_points(mesh, 
                  method=None, 
                  pts=None, 
                  num_samples=3,
                  as_spheres=True):
    """
    Returns:
        - points
        - Scene
    """
    if pts is None:
        pts = method(mesh, num_samples)
    colors = np.asarray([[255, 0, 0]] * num_samples)
    if as_spheres:
        pcd = visualize.create_spheres(
            pts, colors=colors, radius=0.01, subdivisions=0)
    else:
        pcd = visualize.create_pcd_scene(
            pts, colors=colors/255)
    scene = trimesh.Scene([mesh, pcd])
    return Selected(pts, scene)