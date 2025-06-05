import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

# CDK, change temperature
def max_logit_based_temperature(logit_s, logit_t):
    with torch.no_grad():
        temperature, _ = torch.max(torch.maximum(torch.abs(logit_s), torch.abs(logit_t)), dim=1, keepdim=True)
        temperature = (temperature * (1 + torch.sqrt(torch.tensor(3.0)))) / 2
    return temperature

class DKDloss(nn.Module):
    def __init__(self):
        super(DKDloss, self).__init__()

    def forward(self, logits_student, logits_teacher, target, alpha, beta, temperature):
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)

        # CDK, 修改温度对样本适配
        temperature = max_logit_based_temperature(logits_student, logits_teacher)

        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (temperature**2) / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
            * (temperature**2) / target.shape[0]
        )
        
        loss = alpha * tckd_loss + beta * nckd_loss

        return loss

