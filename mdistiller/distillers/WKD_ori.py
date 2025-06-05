# this file is written by CDK
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller, max_logit_based_temperature
# from ._common import *

import math
import numpy as np


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    # CDK, 修改温度对样本适配
    temperature = max_logit_based_temperature(logits_student, logits_teacher)

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()

    # CDK
    # loss_kd *= temperature**2

    # 每个样本的 KL 散度，不立即平均
    loss_kd_per_sample = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)  # shape: [64]

    # 对每个样本应用温度平方
    loss_kd = (loss_kd_per_sample * (temperature ** 2)).mean()

    return loss_kd

# 计算教师学生的logit软概率之间的Wasserstein流量
def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs, dim = w1.shape
    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)

    u = 1/dim*torch.ones_like(w1, device=w1.device, dtype=w1.dtype) # [batch,N,1]
    K = torch.exp(-cost / reg)
    Kt= K.transpose(2, 1)
    for i in range(max_iter):
        v=w2/(torch.bmm(Kt,u)+1e-8) #[batch,N,1]
        u=w1/(torch.bmm(K,v)+1e-8)  #[batch,N,1]

    flow = u.reshape(bs, -1, 1) * K * v.reshape(bs, 1, -1)
    return flow


# 最终损失是流量 × cost matrix 的加权和
def wkd_logit_loss(logits_student, logits_teacher, temperature, cost_matrix=None, sinkhorn_lambda=25, sinkhorn_iter=30):
    pred_student = F.softmax(logits_student / temperature, dim=-1).to(torch.float32)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1).to(torch.float32)

    cost_matrix = F.relu(cost_matrix) + 1e-8
    cost_matrix = cost_matrix.to(pred_student.device)

    # flow shape [bxnxn]
    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)

    ws_distance = (flow * cost_matrix).sum(-1).sum(-1)
    ws_distance = ws_distance.mean()
    return ws_distance


# 支持对真实标签和非标签部分分别建模
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


def wkd_feature_loss(f_s, f_t, eps=1e-5, grid=1):
    if grid == 1:
        f_s_avg, f_t_avg = f_s.mean(dim=(-1, -2)), f_t.mean(dim=(-1, -2))
        f_s_std, f_t_std = torch.sqrt(f_s.var(dim=(-1, -2)) + eps), torch.sqrt(f_t.var(dim=(-1, -2)) + eps)
        mean_loss = F.mse_loss(f_s_avg, f_t_avg, reduction='sum') / f_s.size(0)
        cov_loss = F.mse_loss(f_s_std, f_t_std, reduction='sum') / f_s.size(0)
    elif grid > 1:
        f_s_avg, f_s_std = adaptive_avg_std_pool2d(f_s, out_size=(grid, grid), eps=eps)
        f_t_avg, f_t_std = adaptive_avg_std_pool2d(f_t, out_size=(grid, grid), eps=eps)
        mean_loss = F.mse_loss(f_s_avg, f_t_avg, reduction='sum') / (grid ** 2 * f_s.size(0))
        cov_loss = F.mse_loss(f_s_std, f_t_std, reduction='sum') / (grid ** 2 * f_s.size(0))

    return mean_loss, cov_loss



class WKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(WKD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.WKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.WKD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND

        self.loss_cosine_decay_epoch = cfg.WKD.LOSS.COSINE_DECAY_EPOCH

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        decay_start_epoch = self.loss_cosine_decay_epoch
        if kwargs['epoch'] > decay_start_epoch:
            # cosine decay
            self.wkd_logit_loss_weight_1 = 0.5 * self.wkd_logit_loss_weight * (1 + math.cos(
                (kwargs['epoch'] - decay_start_epoch) / (self.cfg.SOLVER.EPOCHS - decay_start_epoch) * math.pi))
            self.wkd_feature_loss_weight_1 = 0.5 * self.wkd_feature_loss_weight * (1 + math.cos(
                (kwargs['epoch'] - decay_start_epoch) / (self.cfg.SOLVER.EPOCHS - decay_start_epoch) * math.pi))

        else:
            self.wkd_logit_loss_weight_1 = self.wkd_logit_loss_weight
            self.wkd_feature_loss_weight_1 = self.wkd_feature_loss_weight

        loss_wkd = 0
        # WD loss
        logits_teacher = logits_teacher.to(torch.float32)
        loss_wkd_logit = wkd_logit_loss_with_speration(logits_student, logits_teacher, target, self.temperature,
                                                       self.wkd_logit_loss_weight_1, self.dist, self.sinkhorn_lambda,
                                                       self.sinkhorn_iter)
        loss_wkd += loss_wkd_logit

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # loss_kd = self.kd_loss_weight * kd_loss(
        #     logits_student, logits_teacher, self.temperature, self.logit_stand
        # )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_wkd,
        }
        return logits_student, losses_dict
