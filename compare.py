import numpy as np
import torch
import trimesh
import query as qlib
from libzhifan.geometry import SimpleMesh, visualize
from libzhifan.numeric import check_shape, numpize, NDType

class Comparator:

    Locations = ['v', 'surf', 'bnd_sphere', 'inside']
    
    def __init__(self,
                 verts: NDType, 
                 faces: NDType):
        check_shape(verts, (-1, 3))
        check_shape(faces, (-1, 3))
        verts = numpize(verts)
        faces = numpize(faces)
        self.verts = verts
        self.faces = faces
        self.mesh = SimpleMesh(verts, faces)

    def _sample_inside(self, seed=0):
        spr = self.mesh.bounding_sphere
        c = spr.center
        np.random.seed(seed)
        r = np.power(spr.volume * 3 / 4 / np.pi, 1/3)  # 4/3 * pi * r^3
        for _ in range(100):
            p = c + np.random.randn(3) * 0.2 * r
            d = qlib.trimesh_sdf(self.mesh, torch.as_tensor([p]))[0]
            if d > 0:
                return torch.as_tensor(p)
        raise ValueError("Point inside not found.")
    
    def get_point(self, where, seed=0) -> torch.Tensor:
        """ returns (3,) of (x, y, z) """
        np.random.seed(seed)
        if where == 'v':
            ip = np.random.choice(len(self.verts))
            p = torch.as_tensor(self.verts[ip])
        elif where == 'surf':
            p = qlib.sample_surface(self.mesh, 1).squeeze_()
        elif where == 'bnd_sphere':
            p = qlib.sample_bounding_sphere(self.mesh, 1).squeeze_()
        elif where == 'inside':
            p = self._sample_inside(seed)
            
        return p
    
    def compare_method(self, method_func, seed=0):
        """

        Args:
            method_func (Callable):
                method_func(verts, faces, queries) => dists
                    verts: (V, 3)
                    faces: (F, 3)
                    queries: (1, 3)
                    => d: (1,)
        """
        verts = torch.as_tensor(self.verts)
        faces = torch.as_tensor(self.faces)
        for loc in self.Locations:
            q = self.get_point(loc, seed=seed)
            q = q[None]
            gt = qlib.trimesh_sdf(self.mesh, q)
            pred = method_func(verts, faces, q)
            gt = gt[0]
            pred = float(pred[0])
            print(f"At {loc}, \tTrimesh-SDF = {gt}, \tmethod = {pred}")
    
    def visualize_point(self, p):
        colors = np.asarray([[255, 0, 0]])
        pcd = visualize.create_spheres(
            [p], colors=colors, radius=0.01, subdivisions=0)
        scene = trimesh.Scene([self.mesh, pcd])
        return scene