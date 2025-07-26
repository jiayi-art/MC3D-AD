from __future__ import division

import json
import logging
import open3d as o3d
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import os
import torch
from models.backbones.pointmae.pointmae import fps
from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.transforms import RandomColorJitter

logger = logging.getLogger("global_logger")


def build_custom_dataloader(cfg, training, distributed=True,template=None):


    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        cfg["meta_file"],
        cfg["data_dir"],
        training,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    if(training):
        data_loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
            # worker_init_fn=worker_init_fn
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=cfg["workers"],
            pin_memory=True,
            sampler=sampler,
        )

    return data_loader

def get_label_dict(directory):
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    # 创建字典存储子文件夹名称与第一个文件路径的映射
    subfolder_file_dict = {}
    
    for idx,subfolder in enumerate(subfolders):
        subfolder_file_dict[subfolder] = idx
    
    return subfolder_file_dict



class CustomDataset(BaseDataset):
    def __init__(
        self,
        meta_file,
        data_path,
        training,
    ):
        self.meta_file = meta_file
        self.training = training
        self.data_path = data_path
        self.label_dict = get_label_dict(self.data_path)


        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)
                
    def norm_pcd(self, point_cloud):

        center = np.average(point_cloud,axis=0)
        # print(center.shape)
        new_points = point_cloud-np.expand_dims(center,axis=0)
        return new_points
        # centroid = np.mean(point_cloud, axis=0)
        # pc = point_cloud - centroid
        # m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        # pc = pc / m
        # return pc

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]
        test_path = self.data_path
        test_path = test_path.replace("real3D_down","Real3D-AD-PCD")
        filename = meta["filename"]
       

        if(self.training):
        # read image
            pcd = o3d.io.read_point_cloud(os.path.join(self.data_path,filename))
        else:
            pcd = o3d.io.read_point_cloud(os.path.join(test_path,filename))
        pointcloud = np.array(pcd.points)
        label = meta["label"]
        input.update(
            {
                "filename": filename,
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
            input["cls_label"] = self.label_dict[meta["clsname"]]
        else:
            input["clsname"] = filename.split("/")[-4]
            raise ValueError("Error! Dataset don't has clsname")


        # read / generate mask
        if meta.get("maskname", None):
            if('Anomaly_ShapeNet' in self.data_path):
                pcd = np.genfromtxt(os.path.join(test_path,meta["maskname"]), delimiter=",")
            else:
                pcd = np.genfromtxt(os.path.join(test_path,meta["maskname"]), delimiter=" ")
            pointcloud = pcd[:,:3]
            mask = pcd[:,3]
        else:
            if label == 0:  # good
                mask = np.zeros((pointcloud.shape[0]))
            elif label == 1:  # defective
                mask = np.ones((pointcloud.shape[0]))
            else:
                raise ValueError("Labels must be [None, 0, 1]!")
            
        pointcloud = self.norm_pcd(pointcloud)


        pointcloud = transforms.ToTensor()(pointcloud)[0]
        # print(pointcloud.shape)
        point_num = pointcloud.shape[0]
        
        
        input.update({"pointcloud": pointcloud, "mask": mask,"point_num":point_num})
        return input
    
