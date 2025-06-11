import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller, max_logit_based_temperature
from ._common import *

import math
import numpy as np

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

# 计算教师学生的logit软概率之间的Wasserstein流量
def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs, dim = w1.shape
    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)

    u = 1 / dim * torch.ones_like(w1, device=w1.device, dtype=w1.dtype)  # [batch,N,1]
    K = torch.exp(-cost / reg)
    Kt = K.transpose(2, 1)
    for i in range(max_iter):
        v = w2 / (torch.bmm(Kt, u) + 1e-8)  # [batch,N,1]
        u = w1 / (torch.bmm(K, v) + 1e-8)  # [batch,N,1]

    flow = u.reshape(bs, -1, 1) * K * v.reshape(bs, 1, -1)
    return flow

# 最终损失是流量 × cost matrix 的加权和
def wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix=None, sinkhorn_lambda=25, sinkhorn_iter=30):
    logit_stand = True
    logits_student = normalize(logits_student) if logit_stand else logits_student
    logits_teacher = normalize(logits_teacher) if logit_stand else logits_teacher

    # CDK, 修改温度对样本适配
    temperature = max_logit_based_temperature(logits_student, logits_teacher)

    pred_student = F.softmax(logits_student / temperature, dim=-1).to(torch.float32)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1).to(torch.float32)

    cost_matrix = F.relu(cost_matrix) + 1e-8
    cost_matrix = cost_matrix.to(pred_student.device)

    # flow shape [bxnxn]
    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)

    ws_distance = (flow * cost_matrix).sum(-1).sum(-1)
    ws_distance = ws_distance.mean()
    return ws_distance

# 带分离的Wasserstein知识蒸馏损失计算，包括正确类别和错误类别的分别处理
def wkd_logit_loss_with_speration(logits_student, logits_teacher, gt_label, temperature, gamma, cost_matrix=None,
                                  sinkhorn_lambda=0.05, sinkhorn_iter=10):
    if len(gt_label.size()) > 1:
        label = torch.max(gt_label, dim=1, keepdim=True)[1]
    else:
        label = gt_label.view(len(gt_label), 1)

    # N*class
    N, c = logits_student.shape
    s_i = F.log_softmax(logits_student, dim=1)
    t_i = F.softmax(logits_teacher, dim=1)
    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()
    loss_t = - (t_t * s_t).mean()

    mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()
    logits_student = logits_student[mask].reshape(N, -1)
    logits_teacher = logits_teacher[mask].reshape(N, -1)

    cost_matrix = cost_matrix.repeat(N, 1, 1)
    gd_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    cost_matrix = cost_matrix[gd_mask].reshape(N, c - 1, c - 1)

    # N*class
    loss_wkd = wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix, sinkhorn_lambda, sinkhorn_iter)

    return loss_t + gamma * loss_wkd


# 提取每个局部区域的均值和标准差（mean + std）
def adaptive_avg_std_pool2d(input_tensor, out_size=(1, 1), eps=1e-5):
    def start_index(a, b, c):
        return int(np.floor(a * c / b))

    def end_index(a, b, c):
        return int(np.ceil((a + 1) * c / b))

    b, c, isizeH, isizeW = input_tensor.shape
    if len(out_size) == 2:
        osizeH, osizeW = out_size
    else:
        osizeH = osizeW = out_size

    avg_pooled_tensor = torch.zeros((b, c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    # cov_pooled_tensor = torch.zeros((b, c*c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    cov_pooled_tensor = torch.zeros((b, c, osizeH, osizeW), dtype=input_tensor.dtype, device=input_tensor.device)
    # block_list = []
    for oh in range(osizeH):
        istartH = start_index(oh, osizeH, isizeH)
        iendH = end_index(oh, osizeH, isizeH)
        kH = iendH - istartH
        for ow in range(osizeW):
            istartW = start_index(ow, osizeW, isizeW)
            iendW = end_index(ow, osizeW, isizeW)
            kW = iendW - istartW

            # avg pool2d
            input_block = input_tensor[:, :, istartH:iendH, istartW:iendW]
            avg_pooled_tensor[:, :, oh, ow] = input_block.mean(dim=(-1, -2))
            # diagonal cov pool2d
            cov_pooled_tensor[:, :, oh, ow] = torch.sqrt(input_block.var(dim=(-1, -2)) + eps)

    return avg_pooled_tensor, cov_pooled_tensor

class WKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(WKD, self).__init__(student, teacher)
        self.cfg = cfg

        # 交叉熵损失（Cross-Entropy Loss）的权重
        self.ce_loss_weight = cfg.WKD.LOSS.CE_WEIGHT
        # Logit 层蒸馏损失（WKD Logit Loss）的权重
        self.wkd_logit_loss_weight = cfg.WKD.LOSS.KD_WEIGHT

        # 启用 WKD Logit Loss
        self.enable_wkdl = True
        # 不启用 WKD Feature Loss
        self.enable_wkdf = False

        # WKD-L: WD for logits distillation
        if self.enable_wkdl:
            # 设置Logit蒸馏相关参数
            self.temperature = cfg.WKD.TEMPERATURE
            self.sinkhorn_lambda = 0.05
            self.sinkhorn_iter = 10

            # 构建类别间距离矩阵
            # if cfg.WKD.COST_MATRIX == "fc":
            print("Using fc weight of teacher model as category prototype")
            self.prototype = self.teacher.fc.weight
            # caluate cosine similarity
            proto_normed = F.normalize(self.prototype, p=2, dim=-1)
            cosine_sim = proto_normed.matmul(proto_normed.transpose(-1, -2))
            self.dist = 1 - cosine_sim
            # else:
            #     print("Using " + cfg.WKD.COST_MATRIX + " as cost matrix")
            #     path_gd = cfg.WKD.COST_MATRIX_PATH
            #     self.dist = torch.load(path_gd).cuda().detach()

            # # 对原始距离矩阵进行指数变换，增强类别间的区分度
            # if cfg.WKD.COST_MATRIX_SHARPEN != 0:
            #     print("Sharpen ", cfg.WKD.COST_MATRIX_SHARPEN)
            #     sim = torch.exp(-cfg.WKD.COST_MATRIX_SHARPEN * self.dist)
            #     self.dist = 1 - sim

        self.teacher = self.teacher.eval()
        self.student = self.student.eval()

    def get_learnable_parameters(self):
        student_params = [v for k, v in self.student.named_parameters()]
        if self.enable_wkdf:
            return student_params + list(self.conv_reg.parameters())
        else:
            return student_params

    def get_extra_parameters(self):
        return 0

    def forward_train(self, image, target, **kwargs):
        with torch.cuda.amp.autocast():
            # Student outputs
            logits_student_base, feats_student = self.student(image)
            logits_student_base = logits_student_base.to(torch.float32)

        # Teacher outputs (no grad)
        with torch.no_grad():
            logits_teacher_base, feats_teacher = self.teacher(image)
            logits_teacher_base = logits_teacher_base.to(torch.float32)

        # CE Loss Branch
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student_base, target)

        # Logit Distillation Branch
        loss_wkd = 0
        if self.enable_wkdl:
            logits_student_kd = logits_student_base.detach().clone().requires_grad_(True)
            logits_teacher_kd = logits_teacher_base.detach().clone()
            loss_wkd += wkd_logit_loss_with_speration(
                logits_student_kd, logits_teacher_kd,
                target, self.temperature, self.wkd_logit_loss_weight, self.dist, self.sinkhorn_lambda, self.sinkhorn_iter
                # target, self.temperature, self.wkd_logit_loss_weight, self.dist, self.sinkhorn_lambda, self.sinkhorn_iter
            )

        losses_dict = {"loss_ce": loss_ce, "loss_kd": loss_wkd}
        return logits_student_base, losses_dict