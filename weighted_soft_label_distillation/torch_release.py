import torch
import torch.nn as nn
import torch.nn.functional as F


class WSLDistillerLoss(nn.Module):
    def __init__(self):
        '''
        reference: https://github.com/bellymonster/Weighted-Soft-Label-Distillation/blob/master/knowledge_distiller.py
        '''
        super().__init__()

        self.T = 2

        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax().cuda()

    def forward(self, fc_s, fc_t, label):
        s_input_for_softmax = fc_s / self.T
        t_input_for_softmax = fc_t / self.T

        t_soft_label = self.softmax(t_input_for_softmax)

        softmax_loss = -torch.sum(t_soft_label *
                                  self.logsoftmax(s_input_for_softmax),
                                  1,
                                  keepdim=True)

        fc_s_auto = fc_s.detach()
        fc_t_auto = fc_t.detach()
        log_softmax_s = self.logsoftmax(fc_s_auto)
        log_softmax_t = self.logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(label, num_classes=fc_s.shape[-1]).float()
        softmax_loss_s = -torch.sum(one_hot_label * log_softmax_s,
                                    1,
                                    keepdim=True)
        softmax_loss_t = -torch.sum(one_hot_label * log_softmax_t,
                                    1,
                                    keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-20)
        ratio_lower = torch.zeros(1)
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(-focal_weight)
        softmax_loss = focal_weight * softmax_loss

        soft_loss = (self.T**2) * torch.mean(softmax_loss)

        return soft_loss
