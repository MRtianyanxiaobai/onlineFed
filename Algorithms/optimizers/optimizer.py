from torch.optim import Optimizer
import torch
import copy

class ASOOptimizer(Optimizer):
    def __init__(self, params, lr, lamda, beta):
        defaults = dict(lr=lr, lamda=lamda, beta=beta)
        super(ASOOptimizer, self).__init__(params.parameters(), defaults)

        self.sk_grad = copy.deepcopy(list(params.parameters()))
        self.hk = copy.deepcopy(list(params.parameters()))
        for param, sk, hk in zip(params.parameters(), self.sk_grad, self.hk):
            sk.data = torch.zeros_like(param.data)
            hk.data = torch.zeros_like(param.data)
    # no dynamic_step_size  
    def step(self, central_weight, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p, server_weight, pre_sk_grad, pre_hk in zip( group['params'], central_weight, self.sk_grad, self.hk):
                current_sk_grad = p.grad.data + group['lamda'] * (p.data - server_weight.data)
                p.data = p.data - group['lr'] * (current_sk_grad - pre_sk_grad.data + pre_hk.data)
                pre_hk.data = group['beta']*pre_hk.data + (1 - group['beta'])*pre_sk_grad.data
                pre_sk_grad.data = current_sk_grad
        return loss

class lamdaSGDOptimizer(Optimizer):
    def __init__(self, params, lr, lamda):
        print(params, lr, lamda)
        defaults = dict(lr=lr, lamda=lamda)
        super(lamdaSGDOptimizer, self).__init__(params.parameters(), defaults)
    
    def step(self, central_weight, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p in group['params']:
                # p.data = p.data - group['lr']*p.grad.data
                d_p = p.grad.data
                p.data.add_(d_p, alpha=-group['lr'])
        return loss