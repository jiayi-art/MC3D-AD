import open3d as o3d
import os
from pointnet2_ops import pointnet2_utils
import json
import numpy as np
import torch
import argparse
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    B, N, _ = data.shape
    if number > N:
        raise ValueError("Number of points to sample must not exceed the total number of points")
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx

def tensor_to_o3d_pointcloud(tensor):
    """
    Convert Tensor point cloud data to Open3D point cloud objects

    Args:
        tensor (torch.Tensor):(N, 3)

    Returns:
        o3d.geometry.PointCloud: Open3D pcd object
    """

    if tensor.is_cuda:
        tensor = tensor.cpu()
    points_np = tensor.numpy() 


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    return pcd

parser = argparse.ArgumentParser()
args = parser.parse_args()
parser.add_argument("--radl3d_path", type=str, default="/Your/Path/To/Real3D-AD-PCD")

with open("./data/Real3D/train.json", "r") as f_r:
    metas = []
    for line in f_r:
        meta = json.loads(line)
        metas.append(meta)

for m in metas:
    filename = m["filename"]
    pcd = o3d.io.read_point_cloud(os.path.join(args.radl3d_path,filename))
    points = torch.from_numpy(np.array(pcd.points)).unsqueeze(0).cuda().float()
    down_points,_ = fps(points,150000)
    down_pcd = tensor_to_o3d_pointcloud(down_points.squeeze(0))
    file_path = os.path.join(args.radl3d_path,filename).replace("Real3D-AD-PCD","real3D_down")
    file_name = filename.split("/")[-1]
    file_path = file_path.split(file_name)[0]
    os.makedirs(file_path,exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(file_path,file_name), down_pcd)
    print(f"Save the downsampled point cloud to: {os.path.join(file_path,file_name)}")

