from .pointmae import Model1

__version__ = "0.7.1"
__all__ = [
    "pointmae",
]

def pointmae(pretrained,outlayers,checkpoint_path,group_size,num_group,data_dir):
    model = Model1(data_dir,device='cuda:0', 
                            xyz_backbone_name='Point_MAE', 
                            checkpoint_path = checkpoint_path,
                            group_size = group_size, 
                            num_group = num_group,
                            )
    return model