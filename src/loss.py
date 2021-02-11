import torch
import torch.nn.functional as F
from util import sim_matrix
import numpy as np
def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp
def mixup( tar_inputs,tar_outputs,model,beta):
    model.to(tar_inputs.device)
    bs = tar_inputs.shape[0]
    idx = torch.randperm(bs)
    tar_inputs1,tar_inputs2 = tar_inputs,tar_inputs[idx]
    tar_outputs1,tar_outputs2 = tar_outputs,tar_outputs[idx]
    
    d = np.random.beta(beta, beta,bs)
    d = torch.FloatTensor(d).to(tar_inputs.device)
    x_d = d.view(bs, 1, 1, 1)
    y_d = d.view(bs, 1)
    logit_p = tar_outputs1*y_d+tar_outputs2*(1-y_d)
    
    logit_m = model(tar_inputs1*x_d.detach()+tar_inputs2*(1-x_d.detach()))
    delta_kl = kl_div_with_logit(logit_p.detach(), logit_m)
    return delta_kl

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = -1.0 * torch.mean(F.softmax(x, dim=1) * F.log_softmax(x, dim=1),axis=1)
        return b
    