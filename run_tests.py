import unittest
import torch
import trimesh
import query as qlib
import utils
from libzhifan.geometry import SimpleMesh
from libzhifan.numeric import numpize


def visualize(verts, faces):
    phi, _, _ = qlib.calc_scaled_sdf(verts, faces, 32)
    utils.make_mesh(phi[0], -0.02).show()


class SDFTest(unittest.TestCase):
    
    def test_scenesdf(self):
        # mesh = kal.io.obj.import_mesh('/home/skynet/Zhifan/ihoi/weights/obj_models/cup_simplified.obj')
        mesh = trimesh.load_mesh('/home/skynet/Zhifan/ihoi/weights/obj_models/bottle_simplified.obj')
        # mesh = OBJLoader(obj_models_root='/home/skynet/Zhifan/ihoi/weights/obj_models/'
        #                 ).load_obj_by_name('bottle', return_mesh=False)
        verts = torch.as_tensor(mesh.vertices, device='cuda')[None]
        faces = torch.as_tensor(mesh.faces, dtype=torch.int32, device='cuda')

        # Trimesh sdf
        mesh = SimpleMesh(verts, faces)
        selection = qlib.select_points(
            mesh, 
            # qlib.sample_bounding_sphere,
            qlib.sample_surface,
            num_samples=3, as_spheres=True)
        d_gt = qlib.trimesh_sdf(mesh, selection.points)  # (N,)
    
        # Scene sdf
        qs = selection.points[None].cuda()
        d_pred = qlib.scene_sdf_dist(verts, faces, qs, grid_size=128)  # (1,N)
        d_pred = numpize(d_pred.squeeze_())
        torch.testing.assert_allclose(d_pred, d_gt, rtol=1e-2, atol=1e-2)
        

if __name__ == '__main__':
    unittest.main()