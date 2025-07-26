import glob
import logging
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from datasets.metrics import AUPRO
from sklearn import metrics


def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    preds = outputs["pred"].squeeze(1).cpu().numpy()  # B x 1 x H x W
    masks = outputs["mask"].cpu().numpy()  # B x 1 x H x W
    point_cloud = outputs["pointcloud"].cpu().numpy()  # B x 1 x H x W
    center_idx = outputs["center_idx"].cpu().numpy()
    clsnames = outputs["clsname"]
    labels = outputs["label"].cpu().numpy()
    cls_labels = outputs["cls_label"].cpu().numpy()
    cls_preds = torch.argmax(outputs["cls_pred"],dim=1).cpu().numpy()
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        _, subname = os.path.split(file_dir)
        filename = "{}_{}_{}".format(clsnames[i], subname, filename)
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file,
            filename=filenames[i],
            pred=preds[i],
            mask=masks[i],
            point_cloud=point_cloud[i],
            label=labels[i],
            center_idx=center_idx[i],
            clsname=clsnames[i],
            cls_label=cls_labels[i],
            cls_pred=cls_preds[i]
        )


def fill_missing_values(x_data,x_label,y_data, k=1):
    # 创建最近邻居模型
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)

    # 找到每个点的最近邻居
    distances, indices = nn.kneighbors(y_data)
    # print(distances.shape)
    # print(indices.shape)
    avg_values = np.mean(x_label[indices], axis=1)
    # print("avg_values.shape",avg_values.shape)
    return avg_values

def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    fileinfos = []
    preds = []
    masks = []
    labels = []
    image_max_score = []
    cls_label = []
    cls_pred = []
    points = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        fileinfos.append(
            {
                "filename": str(npz["filename"]),
                "clsname": str(npz["clsname"]),
            }
        )
        point_cloud = npz["point_cloud"]
        points.append(point_cloud)
        sample_idx = npz["center_idx"]
        mask_idx = sample_idx.squeeze().astype(np.int64)
        xyz_sampled = point_cloud[mask_idx,:]
        pred = npz["pred"]
        # print(point_cloud.shape,pred.shape,xyz_sampled.shape)
        preds_all = fill_missing_values(xyz_sampled,pred,point_cloud)
        # pixel_auc.append(metrics.roc_auc_score(npz["mask"],preds_all))
        # print("preds_all shape:",preds_all.shape)
        preds_all = F.avg_pool1d(torch.from_numpy(preds_all).unsqueeze(0),kernel_size=511,padding=511//2,stride=1).squeeze(0).numpy()
        preds.append(preds_all)
        masks.append(npz["mask"])
        labels.append(npz["label"])
        image_max_score.append(preds_all.max())
        cls_label.append(npz["cls_label"])
        cls_pred.append(npz["cls_pred"])
    # preds = np.concatenate(np.asarray(preds), axis=0)  # N x H x W
    # masks = np.concatenate(np.asarray(masks), axis=0)  # N x H x W
    return fileinfos, labels,image_max_score,masks,preds,cls_label,cls_pred,points



class Report:
    def __init__(self, heads=None):
        if heads:
            self.heads = list(map(str, heads))
        else:
            self.heads = ()
        self.records = []

    def add_one_record(self, record):
        if self.heads:
            if len(record) != len(self.heads):
                raise ValueError(
                    f"Record's length ({len(record)}) should be equal to head's length ({len(self.heads)})."
                )
        self.records.append(record)

    def __str__(self):
        return tabulate.tabulate(
            self.records,
            self.heads,
            tablefmt="pipe",
            numalign="center",
            stralign="center",
        )


class EvalDataMeta:
    def __init__(self, preds, masks):
        self.preds = preds  # N x H x W
        self.masks = masks  # N x H x W


class EvalImage:
    def __init__(self, data_meta, **kwargs):
        self.preds = self.encode_pred(data_meta.preds, **kwargs)
        self.masks = self.encode_mask(data_meta.masks)
        self.preds_good = sorted(self.preds[self.masks == 0], reverse=True)
        self.preds_defe = sorted(self.preds[self.masks == 1], reverse=True)
        self.num_good = len(self.preds_good)
        self.num_defe = len(self.preds_defe)

    @staticmethod
    def encode_pred(preds):
        raise NotImplementedError

    def encode_mask(self, masks):
        N, _, _ = masks.shape
        masks = (masks.reshape(N, -1).sum(axis=1) != 0).astype(np.int)  # (N, )
        return masks

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc


class EvalImageMean(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).mean(axis=1)  # (N, )


class EvalImageStd(EvalImage):
    @staticmethod
    def encode_pred(preds):
        N, _, _ = preds.shape
        return preds.reshape(N, -1).std(axis=1)  # (N, )


class EvalImageMax(EvalImage):
    @staticmethod
    def encode_pred(preds, avgpool_size):
        N, _, _ = preds.shape
        preds = torch.tensor(preds[:, None, ...]).cuda()  # N x 1 x H x W
        preds = (
            F.avg_pool2d(preds, avgpool_size, stride=1).cpu().numpy()
        )  # N x 1 x H x W
        return preds.reshape(N, -1).max(axis=1)  # (N, )


class EvalPerPixelAUC:
    def __init__(self, data_meta):
        self.preds = np.concatenate(
            [pred.flatten() for pred in data_meta.preds], axis=0
        )
        self.masks = np.concatenate(
            [mask.flatten() for mask in data_meta.masks], axis=0
        )
        self.masks[self.masks > 0] = 1

    def eval_auc(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.masks, self.preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if auc < 0.5:
            auc = 1 - auc
        return auc
def min_max_normalize(data):
    """
    对给定的 NumPy 数组进行 Min-Max 归一化。

    参数:
        data (np.ndarray): 输入的 NumPy 数组。

    返回:
        np.ndarray: 归一化后的数组。
    """
    # 计算最小值和最大值
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 进行归一化
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return normalized_data

eval_lookup_table = {
    "mean": EvalImageMean,
    "std": EvalImageStd,
    "max": EvalImageMax,
    "pixel": EvalPerPixelAUC,
}

def get_auc(label,pred):
    auc = metrics.roc_auc_score(np.asarray(label),min_max_normalize(np.asarray(pred)))
    # if auc < 0.5:
    #     auc = 1-auc
    return auc

def get_pro(label,pred):
    point_AUPRO = AUPRO().cuda()
    point_AUPRO.update(torch.from_numpy(min_max_normalize(np.asarray(pred))).cuda,torch.from_numpy(np.asarray(label)).cuda())

def performances(fileinfos, labels,image_max_score,masks,preds,cls_labels,cls_preds):
    ret_metrics = {}
    clsnames = set([fileinfo["clsname"] for fileinfo in fileinfos])
    for clsname in clsnames:
        labels_l = []
        image_max_cls = []
        mask_l = []
        pred_l = []
        cls_label_l = []
        cls_pred_l = []
        for fileinfo, label,max,mask,pred,cls_label,cls_pred in zip(fileinfos, labels,image_max_score,masks,preds,cls_labels,cls_preds):
            if fileinfo["clsname"] == clsname:
                labels_l.append(float(label))
                image_max_cls.append(max)
                mask_l.append(mask)
                pred_l.append(pred)
                cls_label_l.append(cls_label)
                cls_pred_l.append(cls_pred)

        preds_l = np.concatenate(pred_l)  # B x N
        masks_l = np.concatenate(mask_l)  # B x N
        cls_label_l = np.array(cls_label_l)
        cls_pred_l = np.array(cls_pred_l)
        # data_meta = EvalDataMeta(preds_cls, masks_cls)
        ret_metrics["{}_{}_auc".format(clsname, "pixel-AUROC")] = get_auc(masks_l,preds_l)
        ret_metrics["{}_{}_auc".format(clsname, "obj-AUROC")] = get_auc(labels_l,image_max_cls)
        ret_metrics["{}_{}_auc".format(clsname, "cls-ACC")] = np.mean(cls_label_l==cls_pred_l)
        # auc
        # if config.get("auc", None):
        #     for metric in config.auc:
        #         evalname = metric["name"]
        #         kwargs = metric.get("kwargs", {})
        #         eval_method = eval_lookup_table[evalname](data_meta, **kwargs)
        #         auc = eval_method.eval_auc()
        #         ret_metrics["{}_{}_auc".format(clsname, evalname)] = auc

    for metric in ["obj-AUROC","pixel-AUROC","cls-ACC"]:
        evalname = metric
        evalvalues = [
            ret_metrics["{}_{}_auc".format(clsname, evalname)]
            for clsname in clsnames
        ]
        mean_auc = np.mean(np.array(evalvalues))
        ret_metrics["{}_{}_auc".format("mean", evalname)] = mean_auc

    return ret_metrics


def log_metrics(ret_metrics, config):
    logger = logging.getLogger("global_logger")
    clsnames = set([k.rsplit("_", 2)[0] for k in ret_metrics.keys()])
    clsnames = list(clsnames - set(["mean"])) + ["mean"]

    # auc
    if config.get("auc", None):
        auc_keys = [k for k in ret_metrics.keys() if "auc" in k]
        evalnames = list(set([k.rsplit("_", 2)[1] for k in auc_keys]))
        record = Report(["clsname"] + evalnames)

        for clsname in clsnames:
            clsvalues = [
                ret_metrics["{}_{}_auc".format(clsname, evalname)]
                for evalname in evalnames
            ]
            record.add_one_record([clsname] + clsvalues)

        logger.info(f"\n{record}")
