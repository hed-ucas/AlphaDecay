
class AdaptiveWeightDecay:
    def __init__(self, lambda_awd=0.1):
        self.lambda_awd = lambda_awd
        self.lambda_bar = 0

    def step(self, optimizer, update=True):
        total_grad_norm = 0.0
        total_weight_norm = 0.0
        
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    total_grad_norm += param.grad.norm(2).pow(2).item()  # ||∇w||^2
                    total_weight_norm += param.data.norm(2).pow(2).item()  # ||w||^2
        
        total_grad_norm = total_grad_norm ** 0.5
        total_weight_norm = total_weight_norm ** 0.5
        
        lambda_t = (total_grad_norm * self.lambda_awd) / (total_weight_norm + 1e-8)
        
        self.lambda_bar = 0.1 * self.lambda_bar + 0.9 * lambda_t
        
        if update:
            for group in optimizer.param_groups:
                group['weight_decay'] = self.lambda_bar
                
        return self.lambda_bar