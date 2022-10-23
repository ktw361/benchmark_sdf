import torch
import query as qlib

import trimesh
# import kaolin as kal
# from obj_pose.obj_loader import OBJLoader
from libzhifan.geometry import SimpleMesh
from utils import make_mesh
from compare import Comparator

# mesh = kal.io.obj.import_mesh('/home/skynet/Zhifan/ihoi/weights/obj_models/cup_simplified.obj')
mesh = trimesh.load_mesh('/home/skynet/Zhifan/ihoi/weights/obj_models/bottle_simplified.obj')
verts = torch.as_tensor(mesh.vertices, device='cuda')[None]
faces = torch.as_tensor(mesh.faces, dtype=torch.int32, device='cuda')
print(verts.shape, faces.shape)

comp = Comparator(verts[0], faces)

print("Compare Scene sdf")
def scene_sdf_wrapper(v, f, q):
    return qlib.scene_sdf_dist(v[None].cuda(), f.cuda(), q.cuda(), grid_size=128)[0]
comp.compare_method(scene_sdf_wrapper, seed=0)


print("Compare Knn")
from pytorch3d.ops import knn_points
def knn_wrapper(v, f, q):
    d = knn_points(q[None].float(), v[None].float()).dists.sqrt_()
    return d
comp.compare_method(knn_wrapper, seed=0)


print("Compare o3d")
import open3d as o3d
def o3d_wrapper(v, f, q):
    scene = o3d.t.geometry.RaycastingScene()
    vv = o3d.utility.Vector3dVector(v)
    ff = o3d.utility.Vector3iVector(f)
    mesh = o3d.geometry.TriangleMesh(vv, ff)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh)
    
    x, y, z = q[0]
    query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)
    d = scene.compute_signed_distance(query_point)
    return d.numpy()
comp.compare_method(o3d_wrapper)