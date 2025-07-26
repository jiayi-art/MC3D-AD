import copy
import random
from typing import Optional
from sklearn.neighbors import KDTree
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

def flip_normals_to_outward(points, normals):
    """
    Flip the normal direction to ensure it faces outwards

    Args:
        points (np.ndarray): Point cloud coordinates, shape is (N, 3)
        normals (np.ndarray): Point cloud normal, shape (N, 3)

    Returns:
        np.ndarray: Normals of uniform orientation, of shape (N, 3)
    """
    # Calculate the center of gravity of the point cloud
    centroid = np.mean(points, axis=0)

    # Calculate the vector between the normal direction and the center of gravity
    directions = points - centroid
    dot_products = np.sum(normals * directions, axis=1)
    normals[dot_products < 0] = -normals[dot_products < 0]
    return normals

def estimate_parameters_kdtree(points, k=7, sample_size=1000):

    num_points = points.shape[0]
    sample_size = min(sample_size, num_points)
    kdtree = KDTree(points)
    sampled_indices = np.random.choice(num_points, sample_size, replace=False)
    sampled_points = points[sampled_indices]

    distances, _ = kdtree.query(sampled_points, k=2)  
    avg_distance = np.mean(distances[:, 1])  

    # Adaptive calculation of radius and max_nn
    radius = k * avg_distance
    max_nn = min(50, max(20, int(num_points / 1000)))  

    return radius, max_nn

def analyze_normals_curvatures_optimized(point_cloud, k=5,radius=None):
    """
    Optimizing normal and curvature variation analysis of batched point clouds using KDTree.
    """
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()  # 转换为 NumPy 数组

    B, N, _ = point_cloud.shape
    normals = np.zeros((B, N, 3))
    curvatures = np.zeros((B, N))
    normal_variations = np.zeros((B, N))
    curvature_variations = np.zeros((B, N))

    for b in range(B):
        group = point_cloud[b]  # (N, 3)
        kdtree = KDTree(group)
        
        # Adaptive Estimation Radius
        if radius is None:
            radius,_ = estimate_parameters_kdtree(group)
        for i in range(N):
            idx = kdtree.query_radius(group[i].reshape(1, -1), r=radius)[0]
            neighbors = group[idx]

            if len(neighbors) < 3:  
                continue

            centroid = np.mean(neighbors, axis=0)
            cov_matrix = np.cov((neighbors - centroid).T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normals[b, i] = eigenvectors[:, 0]
            curvatures[b, i] = eigenvalues[0] / (np.sum(eigenvalues) + 1e-8)
            normals[b] = flip_normals_to_outward(group,normals[b])

        for i in range(N):
            idx = kdtree.query_radius(group[i].reshape(1, -1), r=radius)[0]
            neighbor_normals = normals[b, idx]

            dot_products = np.dot(neighbor_normals, normals[b, i])
            angles = np.arccos(np.clip(dot_products, -1.0, 1.0)) 
            normal_variations[b, i] = np.mean(angles)

        for i in range(N):
            idx = kdtree.query_radius(group[i].reshape(1, -1), r=radius)[0]
            neighbor_curvatures = curvatures[b, idx]
            curvature_variations[b, i] = np.mean(np.abs(curvatures[b, i] - neighbor_curvatures))

    return normals, curvatures, normal_variations, curvature_variations

class UniAD(nn.Module):
    def __init__(
        self,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        initializer,
        cls_num,
        inplanes=1152,
        k=5,
        mask_ratio=0.4,
        **kwargs,
    ):
        super().__init__()

        self.feature_jitter = feature_jitter
        self.cls_num = cls_num
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim)
        )
        self.distence = torch.nn.MSELoss()

        self.transformer = Transformer(
            hidden_dim, feature_size, neighbor_mask, **kwargs
        )
        self.input_proj = nn.Linear(inplanes, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes)
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(inplanes*2, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_num)
            )
        self.gem_dict = {}
        self.k = k
        self.mask_ratio = mask_ratio
        
        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input):
        feature_align = input["xyz_features"]  # B x C X H x W
        center = input["center"]
        filename = input["filename"]
        filename = filename[0]
        if filename in self.gem_dict:
            geome_vars = self.gem_dict[filename]
        else:
            normals, curvatures, normal_variations, curvature_variations = analyze_normals_curvatures_optimized(
                center,k=self.k
            )
            geome_vars = normal_variations+10*curvature_variations
            self.gem_dict[filename] = geome_vars
        
        feature_tokens = rearrange(
            feature_align, "b n g -> g b n"
        )  # (H x W) x B x C,"b g n -> g b n"
        # feature_tokens_t = feature_tokens
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        pos_embed = self.pos_embed(center).permute(1,0,2)  # (H x W) x C
        output_decoder, _ = self.transformer(
            feature_tokens, pos_embed ,geome_vars,self.mask_ratio
        )  # (H x W) x B x C
        feature_rec_tokens = self.output_proj(output_decoder)  # (H x W) x B x C
        feature_rec = rearrange(
            feature_rec_tokens, "g b n -> b n g"
        )  # B x C X H x W

        feature_cls = feature_rec.detach().clone()
        feature_cls.requires_grad = True
        feature_cls = rearrange(
            feature_cls,"b n g -> b g n"
        )
        concat_f = torch.cat([feature_cls[:, 0], feature_cls[:, 1:].max(1)[0]], dim=-1)
        cls_pred = self.cls_head_finetune(concat_f)

        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x G x 1
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
            "cls_pred": cls_pred,
        }


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        neighbor_mask,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def generate_mask(self, feature_size, geome_vars,mask_ratio=0.4):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        B,_ = geome_vars.shape
        mask = torch.zeros((B,feature_size,), dtype=torch.bool)

        for idx in range(B):
            number = np.random.rand()
            if(number > 0.5):#降序
                sorted_indices_desc = np.argsort(geome_vars[idx])[::-1]
            else:#升序排列
                sorted_indices_desc = np.argsort(geome_vars[idx])
            top_k = int(feature_size * mask_ratio)
            sorted_indices_desc = sorted_indices_desc[:top_k].copy()
            top_40_desc = torch.from_numpy(sorted_indices_desc).cuda()
            mask[idx,top_40_desc] = True
        mask = mask.cuda()
        return mask

    def forward(self, src, pos_embed, geome_vars,mask_ratio):

        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, geome_vars,mask_ratio
            )
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder = self.encoder(
            src, src_key_padding_mask=mask_enc, pos=pos_embed
        )  # (H X W) x B x C
        output_decoder = self.decoder(
            output_encoder,
            tgt_key_padding_mask=mask_dec1,
            pos=pos_embed,
        )  # (H X W) x B x C

        return output_decoder, output_encoder


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = memory

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        num_queries = feature_size
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        tgt = pos
        tgt2 = self.self_attn(
            query=tgt,
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        tgt = pos

        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                out,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            out,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
