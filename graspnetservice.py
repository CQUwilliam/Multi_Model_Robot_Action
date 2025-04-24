# graspnet_service.py

# 导入所需的库
from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

ROOT_DIR = os.path.dirname('/home/cavalier/lgx/llmrobot/graspnet-baseline/')
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'doc'))

import torch
from graspnetAPI import GraspNet
from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# 创建Flask应用
app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=False,default='/home/cavalier/lgx/llmrobot/graspnet-baseline/logs/log_rs/checkpoint-rs.tar', help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

# 初始化GraspNet模型
net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
net.to(device)
# Load checkpoint
checkpoint = torch.load('/home/cavalier/lgx/llmrobot/graspnet-baseline/logs/log_rs/checkpoint-rs.tar')
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
# set model to eval mode
net.eval()


# 定义抓取姿态检测接口
@app.route('/grasp', methods=['POST'])
def grasp():
    data = request.get_json()
    rgb_image_path = data['color_image_path']
    depth_image_path = data['depth_image_path']
    mask_path = data['mask_path']
    
    print(f'rgb_image_path:{rgb_image_path}')
    
    
    color = np.array(Image.open(rgb_image_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_image_path))
    workspace_mask = np.array(Image.open(mask_path)).astype(bool)
    
    # meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = np.array([[906.6353759765625, 0, 654.9713745117188], [0, 906.5818481445312, 358.79400634765625], [0, 0, 1]])
    factor_depth = 1000.0

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    


    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled
    

    
    
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    
    
    mfcdetector = ModelFreeCollisionDetector(np.array(cloud.points), voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    

    
    gg.nms()
    gg.sort_by_score()
  
    gg = gg[:1]
    
    # 删除后不用渲染
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

    # 进行抓取姿态检测
    # 请根据GraspNet的实际API进行修改，这里仅作示例


    # 序列化抓取姿态（示例，仅供参考）

    # grasp_info = {
    #     'translation':gg.translations.tolist(),
    #     'rotation':gg.rotation_matrices.tolist(),
    #     'width':gg.widths
    # }
    np.save('/home/cavalier/lgx/llmrobot/gripper_info/translation.npy',gg.translations)
    np.save('/home/cavalier/lgx/llmrobot/gripper_info/rotation.npy',gg.rotation_matrices)
    np.save('/home/cavalier/lgx/llmrobot/gripper_info/width',gg.widths)
    

    return "ok"
    # return jsonify({'grasps': grasp_info})

if __name__ == '__main__':
    # 运行服务
    app.run(host='127.0.0.1', port=6001)
