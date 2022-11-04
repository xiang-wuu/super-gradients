import dataclasses
from typing import Tuple, Type

import numpy as np
import torch
from torch import nn, Tensor

from super_gradients.modules import ConvBNAct
from super_gradients.training.models.detection_models.pp_yolo_e.assigner import (
    batch_distance2bbox,
)
from super_gradients.training.models.detection_models.pp_yolo_e.nms import MultiClassNMS


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


@torch.no_grad()
def generate_anchors_for_grid_cell(feats: Tuple[Tensor, ...], fpn_strides: Tuple[int, ...], grid_cell_size=5.0, grid_cell_offset=0.5, dtype=torch.float):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    device = feats[0].device
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        anchor = torch.stack(
            [shift_x - cell_half_size, shift_y - cell_half_size, shift_x + cell_half_size, shift_y + cell_half_size],
            dim=-1,
        ).to(dtype=dtype)
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=dtype))

    anchors = torch.concat(anchors).to(device)
    anchor_points = torch.concat(anchor_points).to(device)
    stride_tensor = torch.concat(stride_tensor).to(device)
    return anchors, anchor_points, num_anchors_list, stride_tensor


class ESEAttn(nn.Module):
    def __init__(self, feat_channels: int, activation_type: Type[nn.Module]):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)
        self.conv = ConvBNAct(feat_channels, feat_channels, kernel_size=1, padding=0, stride=1, activation_type=activation_type)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


@dataclasses.dataclass
class PPYOLOEOutput:
    cls_score_list: Tensor
    reg_dist_list: Tensor
    anchor_points: Tensor
    stride_tensor: Tensor


class PPYOLOEHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        nms: MultiClassNMS,
        in_channels: Tuple[int, int, int],
        activation_type: Type[nn.Module] = nn.SiLU,
        fpn_strides: Tuple[int, int, int] = (32, 16, 8),
        grid_cell_scale=5.0,
        grid_cell_offset=0.5,
        reg_max=16,
        eval_size=None,
        exclude_nms=False,
        exclude_post_process=False,
    ):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size

        self.nms = nms

        self.exclude_nms = exclude_nms
        self.exclude_post_process = exclude_post_process
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()

        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, activation_type=activation_type))
            self.stem_reg.append(ESEAttn(in_c, activation_type=activation_type))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(nn.Conv2d(in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(nn.Conv2d(in_c, 4 * (self.reg_max + 1), 3, padding=1))

        # Do not apply quantization to this tensor
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape([1, self.reg_max + 1, 1, 1])
        self.register_buffer("proj_conv", proj)

        self._init_weights()

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            torch.nn.init.constant_(cls_.weight, 0.0)  # TODO: Why zeros here ??
            torch.nn.init.constant_(cls_.bias, bias_cls)
            torch.nn.init.constant_(reg_.weight, 0.0)  # TODO: Why zeros here ??
            torch.nn.init.constant_(reg_.bias, 1.0)

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward_train(self, feats: Tuple[Tensor, ...]):
        anchors, anchor_points, num_anchors_list, stride_tensor = generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset
        )

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            # TODO: Can improve numberical stability here by not applying sigmoid
            # TODO: and calling log_sigmoid in loss computation (This may give some free boost to mAP score)
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(torch.permute(cls_score.flatten(2), [0, 2, 1]))
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))
        cls_score_list = torch.concat(cls_score_list, dim=1)
        reg_distri_list = torch.concat(reg_distri_list, dim=1)

        return cls_score_list, reg_distri_list, anchors, anchor_points, num_anchors_list, stride_tensor

    def forward_eval(self, feats: Tuple[Tensor, ...]):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            avg_feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = torch.permute(reg_dist.reshape([-1, 4, self.reg_max + 1, height_mul_width]), [0, 2, 3, 1])

            # reg_dist = self.proj_conv(torch.nn.functional.softmax(reg_dist, dim=1)).squeeze(1)
            reg_dist = torch.nn.functional.conv2d(torch.nn.functional.softmax(reg_dist, dim=1), weight=self.proj_conv).squeeze(1)

            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, height_mul_width]))
            reg_dist_list.append(reg_dist)

        cls_score_list = torch.concat(cls_score_list, dim=-1)  # [B, C, Anchors]
        reg_dist_list = torch.concat(reg_dist_list, dim=1)  # [B, Anchors, 4]

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def _generate_anchors(self, feats=None, dtype=torch.float):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype))
        anchor_points = torch.concat(anchor_points)
        stride_tensor = torch.concat(stride_tensor)
        return anchor_points, stride_tensor

    def forward(self, feats):
        assert len(feats) == len(self.fpn_strides), "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats)
        else:
            return self.forward_eval(feats)

    def post_process(self, head_outs: Tuple[Tensor, Tensor, Tensor, Tensor], scale_factor: Tensor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        if self.exclude_post_process:
            return torch.concat([pred_bboxes, torch.permute(pred_scores, [0, 2, 1])], dim=-1), None
        else:
            # scale bbox to origin
            scale_y, scale_x = torch.split(scale_factor, 2, dim=-1)
            scale_factor = torch.concat([scale_x, scale_y, scale_x, scale_y], dim=-1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor
            if self.exclude_nms:
                # `exclude_nms=True` just use in benchmark
                return pred_bboxes, pred_scores
            else:
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                return bbox_pred, bbox_num