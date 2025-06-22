import torch
import sys
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from operator import itemgetter
import os
import matplotlib.pyplot as plt

def safe_log10(x, epsilon=1e-10):
    if x <= 0:
        return math.log10(epsilon)
    return math.log10(x)

class layerTempbalance(object):
    def __init__(self, 
                    net, 
                    use_modulewise_wd,
                    alpha_positively_with_WD=True, 
                    EVALS_THRESH=0.00001,
                    bins=100, 
                    conv_norm=0.5,
                    pl_fitting='median',
                    xmin_pos=2,
                    filter_zeros=False,
                    remove_first_layer=True,
                    remove_last_layer=True,
                    eigs_thresh=50,
                    esd_metric_for_tb='alpha',
                    assign_func='tb_linear_map',
                    wd_min_ratio=0.5,
                    wd_max_ratio=1.5,
                    batchnorm=True,
                    batchnorm_type='name',
                    layernorm=False,
                    sigmoid_alpha = 4, 
                    wandb_name=None
                    ):
        """init function
        Args:
            net (nn.module):             net to train
            EVALS_THRESH (float, ):      threshold to filter small eigenvalue. Defaults to 0.00001.
            bins (int, int):             ESD bins. Defaults to 100.
            conv_norm (float, ):         conv norm. Defaults to 0.5.
            pl_fitting (str, ):          powerlaw fitting method. Defaults to median, ['median', 'goodness-of-fit', 'fix-finger']
            xmin_pos (int, ):            set the position of minimum eigenvalue in the tail. Defaults to 2.
            filter_zeros (bool, ):       filter small eigenvalues or not. Defaults to False.
            remove_first_layer (bool, ): whether exclude first layer in TB. Defaults to True.
            remove_last_layer (bool, ): whether exclude last layer in TB. Defaults to True.
            esd_metric_for_tb (str, ): metric for TB scheduling. Defaults to 'alpha'.
            assign_func (str, ):         learning rate assignment function. Defaults to 'tb_linear_map'.
            wd_min_ratio (float, ):      learning rate lower bound. Defaults to 0.5.
            wd_max_ratio (float, ):       learning rate upper bound. Defaults to 1.5.
            batchnorm (bool, ):          whether adjust batch norm learning rate using TB. Defaults to True.
            batchnorm_type (str, ):      how to set learning rate for batchnorm layers
            layernorm (bool, ):          whether adjust layer norm learning rate using TB. Defaults to True.
        """
        self.net = net
        self.use_modulewise_wd = use_modulewise_wd
        self.alpha_positively_with_WD = alpha_positively_with_WD 
        self.EVALS_THRESH = EVALS_THRESH
        self.bins = bins
        self.conv_norm = conv_norm
        self.pl_fitting = pl_fitting
        self.xmin_pos = xmin_pos
        self.filter_zeros = filter_zeros
        self.remove_first_layer = remove_first_layer
        self.remove_last_layer = remove_last_layer
        self.eigs_thresh = eigs_thresh
        self.esd_metric_for_tb = esd_metric_for_tb
        self.assign_func = assign_func
        self.wd_min_ratio = wd_min_ratio
        self.wd_max_ratio = wd_max_ratio
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.bn_to_conv = {}
        self.ln_to_linear = {}
        self.sigmoid_alpha = sigmoid_alpha
        self.wandb_name = wandb_name
        
        if batchnorm and batchnorm_type == 'name':
            # let the batch norm layer change wd corresponding to the layer
            # with the same layer name 
            longname_lst = []
            for name, m in self.net.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    longname_lst.append(name) 
            for name, module in self.net.named_modules():
                if isinstance(module, nn.BatchNorm2d) \
                        and name.replace('bn', 'conv') in longname_lst: 
                    self.bn_to_conv[name] = name.replace('bn', 'conv') 
        
        if self.layernorm:
            longname_lst = []
            type_lst = []
            for name, module in self.net.named_modules():
                if isinstance(module, nn.Linear):
                    longname_lst.append(name)
                    type_lst.append('nn.Linear')
                if isinstance(module, nn.LayerNorm):
                    if type_lst[-1] == 'nn.Linear':
                        self.ln_to_linear[name] = longname_lst[-1]
                    longname_lst.append(name)
                    type_lst.append('nn.LayerNorm')
            
        
    def build_optimizer_param_group(self, untuned_wd=0.1, initialize=True):
        """build the parameter group for optimizer

        Args:
            untuned_wd (float, ): global learning rate that is not tuned. Defaults to 0.1.
            initialize (bool, ): if True, build a list of dictionary, if False, build a list of learning rate . Defaults to True.

        Returns:
            _type_: _description_
        """
        metrics = self.net_esd_estimator() 
        layer_stats = pd.DataFrame({key:metrics[key] for key in metrics if key!='eigs'})
        if self.remove_first_layer:
            layer_stats = layer_stats.drop(labels=0, axis=0) 
            # index must be reset otherwise may delete the wrong row 
            layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb]))) 
        if self.remove_last_layer:
            layer_stats = layer_stats.drop(labels=len(layer_stats) - 1, axis=0) 
            # index must be reset otherwise may delete the wrong row 
            layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb])))
        
        # remove layers with number of eigs less than a threshold
        layer_stats = layer_stats[layer_stats['eigs_num'] >= self.eigs_thresh] 
        layer_stats.index = list(range(len(layer_stats[self.esd_metric_for_tb]))) 

        metric_scores = np.array(layer_stats[self.esd_metric_for_tb])
        scheduled_wd = self.get_layer_temps(assign_func=self.assign_func, 
                                            metric_scores=metric_scores, 
                                            untuned_wd=untuned_wd,
                                            layer_stats=layer_stats)
        layer_stats['scheduled_wd'] = scheduled_wd 
        
        layer_name_to_tune = list(layer_stats['longname'])
        opt_params_groups = []
        params_to_tune_ids = []
        layer_count = 0 
        # these params should be tuned
        for name, module in self.net.named_modules(): 
            if name in layer_name_to_tune: 
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_wd = layer_stats[layer_stats['longname'] == name]['scheduled_wd'].item()

                if initialize: 
                    # append a dictionary for initialize optimizer
                    opt_params_groups.append({'params': module.parameters(), 'weight_decay': scheduled_wd})
                else:
                    # append tuned learning rate 
                    opt_params_groups.append(scheduled_wd)
                layer_count += 1
            # decide should we tune the batch norm accordingly
            elif self.batchnorm \
                and isinstance(module, nn.BatchNorm2d) \
                    and name in self.bn_to_conv \
                        and self.bn_to_conv[name] in layer_name_to_tune:
                
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_wd = layer_stats[layer_stats['longname'] == self.bn_to_conv[name]]['scheduled_wd'].item()

                if initialize: 
                    # append a dictionary for initialize optimizer
                    opt_params_groups.append({'params': module.parameters(), 'weight_decay': scheduled_wd})
                else:
                    # append tuned learning rate 
                    opt_params_groups.append(scheduled_wd)
                layer_count += 1
            
            elif self.layernorm \
                and isinstance(module, nn.LayerNorm) \
                    and name in self.ln_to_linear \
                        and self.ln_to_linear[name] in layer_name_to_tune:
                
                params_to_tune_ids += list(map(id, module.parameters()))
                scheduled_wd = layer_stats[layer_stats['longname'] == self.ln_to_linear[name]]['scheduled_wd'].item()

                if initialize: 
                    opt_params_groups.append({'params': module.parameters(), 'weight_decay': scheduled_wd})
                else:
                    opt_params_groups.append(scheduled_wd)
                layer_count += 1
        
        if initialize:
            # those params are untuned
            untuned_params = \
                filter(lambda p: id(p) not in params_to_tune_ids, self.net.parameters()) 
            opt_params_groups.append({'params': untuned_params, 'weight_decay': untuned_wd}) 
            return opt_params_groups, layer_count, layer_stats
        else:
            return opt_params_groups, layer_count, layer_stats


    def step(self, optimizer, untuned_wd, step_count=None, rank0=True, gradnorm=None):
        opt_params_groups, layer_count, layer_stats = \
            self.build_optimizer_param_group(untuned_wd=untuned_wd, initialize=False)
        for index, param_group in enumerate(optimizer.param_groups):
            if index <= layer_count - 1:
                param_group['weight_decay'] = opt_params_groups[index]
            else:
                param_group['weight_decay'] = untuned_wd

        target_types = ['attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        layermean_alpha = {'Step': step_count}
        for t in target_types:
            mask = layer_stats['longname'].str.contains(t)
            alphas = layer_stats.loc[mask, 'alpha']
            if len(alphas) > 0:
                layermean_alpha[t] = alphas.mean()
            else:
                layermean_alpha[t] = None 

        return layermean_alpha
                
    def net_esd_estimator(
            self,
            verbose=False):
        """evaluate the ESD of the conv nets
        Args:
            verbose: 
        Returns:
            _type_: _description_
        """
        results = {
            'gradnorm': [],
            'gradnorm_d_weightnorm': [],
            'fnorm':[],
            'spectral_norm': [],
            'entropy': [],
            'stable_rank': [],
            'alphahat':[],
            'alpha':[],
            'longname':[],
            'eigs':[],
            'eigs_num':[]
            }
        if verbose:
            print("=================================")
            print(f"pl_fitting: {self.pl_fitting}, xmin_pos: {self.xmin_pos}, conv_norm: {self.conv_norm}, filter_zeros: {self.filter_zeros}")
            print("=================================")
        # iterate through layers
        for name, m in self.net.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                matrix = m.weight.data.clone().to(torch.float) 

                if isinstance(m, nn.Conv2d): 
                    matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(self.conv_norm) 
                    matrix = matrix.transpose(1, 2).transpose(0, 1) 
                eigs = torch.square(torch.linalg.svdvals(matrix).flatten()) 
                # ascending order 
                eigs, _ = torch.sort(eigs, descending=False) 
                spectral_norm = eigs[-1].item()
                fnorm = torch.sum(eigs).item() 
                stable_rank = fnorm / spectral_norm
                entropy = self.matrix_entropy(torch.sqrt(eigs)) 

                if m.weight.grad is not None:
                    grad_norm = m.weight.grad.data.norm(2).item()
                    gradnorm_div_weightnorm = grad_norm / (spectral_norm + 1e-8)
                else:
                    grad_norm = 10 
                    gradnorm_div_weightnorm = 10

                if self.filter_zeros:
                    nz_eigs = eigs[eigs > self.EVALS_THRESH] 
                    N = len(nz_eigs)
                    # somethines N may equal 0, if that happens, we don't filter eigs
                    if N == 0:
                        nz_eigs = eigs
                        N = len(nz_eigs)
                else:
                    nz_eigs = eigs
                    N = len(nz_eigs)

                log_nz_eigs  = torch.log(nz_eigs) 

                if self.pl_fitting == 'median': #
                    i = int(len(nz_eigs) / self.xmin_pos)
                    xmin = nz_eigs[i] 
                    n = float(N - i) 
                    seq = torch.arange(n).cuda() 
                    final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i]) 
                    final_D = torch.max(torch.abs(
                                1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                            )) 
                else:
                    alphas = torch.zeros(N-1)
                    Ds     = torch.ones(N-1)
                    if self.pl_fitting == 'fix-finger':
                        hist_nz_eigs = torch.log10(nz_eigs) 
                        min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max() 
                        counts = torch.histc(hist_nz_eigs, self.bins, min=min_e, max=max_e) 
                        boundaries = torch.linspace(min_e, max_e, self.bins + 1) 
                        h = counts, boundaries
                        ih = torch.argmax(h[0]) 
                        xmin2 = 10 ** h[1][ih] 
                        xmin_min = torch.log10(0.95 * xmin2) 
                        xmin_max = 1.5 * xmin2 
                    
                    for i, xmin in enumerate(nz_eigs[:-1]):
                        if self.pl_fitting == 'fix-finger':
                            if xmin < xmin_min:
                                continue
                            if xmin > xmin_max:
                                break

                        n = float(N - i) 
                        seq = torch.arange(n).cuda() 
                        alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i]) 
                        alphas[i] = alpha 
                        if alpha > 1:
                            Ds[i] = torch.max(torch.abs(
                                1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                            ))

                    min_D_index = torch.argmin(Ds) 
                    final_alpha = alphas[min_D_index]
                    final_D = Ds[min_D_index]
                
                final_alpha = final_alpha.item()
                final_D = final_D.item()
                final_alphahat=final_alpha*safe_log10(spectral_norm)
                final_alphahat=math.log(1.0 + math.exp(final_alphahat)) 

                results['gradnorm'].append(grad_norm)
                results['gradnorm_d_weightnorm'].append(gradnorm_div_weightnorm)
                results['fnorm'].append(fnorm)
                results['spectral_norm'].append(spectral_norm)
                results['entropy'].append(entropy.detach().cpu().item())
                results['stable_rank'].append(stable_rank)
                results['alphahat'].append(final_alphahat)
                results['alpha'].append(final_alpha)
                results['longname'].append(name) 
                results['eigs'].append(eigs.detach().cpu().numpy()) 
                results['eigs_num'].append(len(eigs)) 
        return results
    
    def matrix_entropy(self, svals):
        EPSILON = 6e-05

        rank = torch.count_nonzero(svals > EPSILON) 
        evals = svals*svals 
        p = evals / torch.sum(evals) + EPSILON
        entropy = -torch.sum(p * torch.log(p)) / torch.log(torch.tensor(rank.detach().cpu().numpy() + EPSILON, dtype=torch.float))
        return entropy

    def get_layer_temps(self, assign_func, metric_scores, untuned_wd, layer_stats):
        n = len(metric_scores)
        idx = [i for i in range(n)]
        temps = np.array([untuned_wd] * n) 

        # Get the layer names from layer_stats
        layer_metrics = {}
        for idx, name in enumerate(layer_stats['longname']):
            layer_name = name.split('.')[2]  # Gets 'layer1' from 'module.layer1.0.conv1'
            if layer_name not in layer_metrics:
                layer_metrics[layer_name] = []
            layer_metrics[layer_name].append(metric_scores[idx])  # Use index to get corresponding score

        if self.alpha_positively_with_WD: 
            if assign_func == 'tb_linear_map':
                wd_range = [self.wd_min_ratio * untuned_wd,  self.wd_max_ratio * untuned_wd]
                score_range = [min(metric_scores),  max(metric_scores)]
                temps = np.interp(metric_scores, score_range, wd_range) 
            elif assign_func == 'tb_sqrt':
                temps = np.sqrt(metric_scores)/np.sum(np.sqrt(metric_scores)+1e-8) * n * untuned_wd
            elif assign_func == 'tb_log2':
                temps = np.log2(metric_scores)/np.sum(np.log2(metric_scores)+1e-8) * n * untuned_wd
            elif assign_func == 'layerwise_sigmoid':
                # Process each layer separately
                temps = np.zeros_like(metric_scores)
                for layer_name, layer_scores in layer_metrics.items():
                    layer_scores = np.array(layer_scores)
                    # Compute layer-specific mean and std
                    layer_mean = np.mean(layer_scores)
                    layer_std = np.std(layer_scores) + 1e-8
                    # Normalize within layer
                    layer_norm = (layer_scores - layer_mean) / layer_std
                    # Apply sigmoid
                    layer_theta = 2 / (1 + np.exp(-self.sigmoid_alpha * layer_norm))
                    # Assign back to original positions
                    layer_indices = [i for i, name in enumerate(layer_stats['longname']) 
                                    if name.split('.')[2] == layer_name]
                    for idx, theta in zip(layer_indices, layer_theta):
                        temps[idx] = theta * untuned_wd
            else:
                raise NotImplementedError
        return temps
