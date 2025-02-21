# optimizer.py
from typing import Iterator, Dict, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer

class LaProp(Optimizer):
    """
    LaProp optimizer implementation.
    Paper: https://arxiv.org/abs/2002.04839
    """
    def __init__(
        self, 
        params: Iterator[nn.Parameter], 
        lr: float = 1e-3, 
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-20,  # Paper recommends 1e-20 for better stability
        weight_decay: float = 0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LaProp does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Use the moving averages to update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction = 1 - beta1 ** state['step']
                
                step_size = group['lr'] / bias_correction
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def adaptive_gradient_clipping(parameters, gradients, clip_factor=0.3, eps=1e-3):
    """Match JAX AGC parameters exactly.
    
    Args:
        parameters: Model parameters
        gradients: Dictionary of parameter gradients
        clip_factor: AGC clip factor (default: 0.3)
        eps: Small constant for numerical stability (default: 1e-3)
    
    Returns:
        Dictionary of clipped gradients
    """
    clipped_gradients = {}
    for param in parameters:
        if param.grad is None:
            continue
            
        # Get parameter and gradient norms
        param_norm = torch.norm(param.detach(), p=2)
        grad_norm = torch.norm(param.grad.detach(), p=2)
        
        if param_norm < eps:  # Skip small params
            clipped_gradients[param] = param.grad
            continue
            
        # Compute clip threshold and apply clipping
        clip_threshold = clip_factor * param_norm
        if grad_norm > clip_threshold:
            scale = clip_threshold / (grad_norm + eps)
            clipped_gradients[param] = param.grad * scale
        else:
            clipped_gradients[param] = param.grad
            
    return clipped_gradients