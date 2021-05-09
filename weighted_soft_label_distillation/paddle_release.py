import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class WSLDistillerLoss(nn.Layer):
    '''
    reference: https://github.com/bellymonster/Weighted-Soft-Label-Distillation/blob/master/knowledge_distiller.py
    '''

    def __init__(self, T=2.0):
        self.T = T

    def __call__(self, logits_s, logits_t, label, mode="train"):
        s_input_for_softmax = logits_s / self.T
        t_input_for_softmax = logits_t / self.T
        t_soft_label = F.softmax(t_input_for_softmax)
        st_ce_loss = -paddle.sum(t_soft_label *
                                 F.log_softmax(s_input_for_softmax),
                                 1,
                                 keepdim=True)
        fc_s_auto = logits_s.detach()
        fc_t_auto = logits_t.detach()
        log_softmax_s = F.log_softmax(fc_s_auto)
        log_softmax_t = F.log_softmax(fc_t_auto)
        one_hot_label = F.one_hot(label, num_classes=logits_s.shape[-1])
        softmax_loss_s = -paddle.sum(one_hot_label * log_softmax_s,
                                     1,
                                     keepdim=True)
        softmax_loss_t = -paddle.sum(one_hot_label * log_softmax_t,
                                     1,
                                     keepdim=True)
        # prevent it from nan
        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-20)
        focal_weight = paddle.clip(focal_weight, min=0.0)
        focal_weight = 1 - paddle.exp(-focal_weight)
        softmax_loss = focal_weight * st_ce_loss
        loss = (self.T**2) * paddle.mean(softmax_loss)
        return loss
