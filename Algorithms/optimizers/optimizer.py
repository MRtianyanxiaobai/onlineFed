from torch.optim import Optimizer
import torch
import copy

class ASOOptimizer(Optimizer):
    def __init__(self, params, lr, lamda, beta):
        defaults = dict(lr=lr, lamda=lamda, beta=beta)
        super(pFedMeOptimizer, self).__init__(params, defaults)

        self.sk_grad = copy.deepcopy(list(self.model.parameters()))
        self.hk = copy.deepcopy(list(self.model.parameters()))
        for param, sk, hk in zip(params, self.sk_grad, self.hk):
            sk.data = torch.zeros_like(param.data)
            hk.data = torch.zeros_like(param.data)

    # no dynamic_step_size and feature learning    
    def step(self, central_weight, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = central_weight.copy()
        for group in self.param_groups:
            for p, localweight, pre_sk_grad, pre_hk in zip( group['params'], central_weight, self.sk_grad, self.hk):
                current_sk_grad = p.grad.data + group['lamda'] * (p.data - localweight.data)
                p.data = p.data - group['lr'] * (current_sk_grad - pre_sk_grad.data + pre_hk.data)
                pre_hk.data = group['beta']*pre_hk.data + (1 - group['beta'])*current_sk_grad
                pre_sk_grad.data = current_sk_grad
        return  group['params'], loss