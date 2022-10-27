import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import torch
import torch.nn as nn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from nncf.torch.layers import NNCFConv2d


# define layer parameter accumulations which we care about
def layer_accumulator(model, layer_filter=None):
    def layer_filt(nm):
        if layer_filter is not None:
            return layer_filter not in name
        else:
            return True

    layers = []
    names = []
    param_nums = []
    params = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module,NNCFConv2d):
            if (layer_filt(name)):
                for n, p in list(module.named_parameters()):
                    if n.endswith('weight'):
                        names.append(name)
                        p.collect = True
                        layers.append(module)
                        param_nums.append(p.numel())
                        params.append(p)
                    else:
                        p.collect = False
                continue
        for p in list(module.parameters()):
            if p.requires_grad:
                p.collect = False
    #print(f'num layers: {len(layers)} sum params:{np.sum(param_nums)}')
    for i, (n, p) in enumerate(zip(names, param_nums)):
        print(i, n, p)

    return names, np.array(param_nums), params


## define fisher computation:
def Fish(model, device, data_loader, params, criterion, max_iterations=100, min_iterations=20,
         max_number_of_datapoints=64, tol=1e-5, save=True):
    ''' Computes an approximate EF for the model - can definitely be optimised!
    Args:
        model - model with which to compute the Tr(EF)
        device
        data_loader
        params - The parameters for which we compute the EF
        criterion - Cross entropy
        max_iterations - maximum number of allowed iteration blocks (i.e. total = max_number_of_datapoints*max_iterations)
        min_iterations - minimum number of allower iteration blocks
        max_number_of_datapoints - size of each iteration block
        tol - tolerance with which to end the iterations (smaller takes longer!)
    Returns:
        vFv_c - EF trace for each parameter group defined in params
        vFc_acc - accumulated convergence plot for the EF trace

    Notes:
        1. max_iterations and max_number_of_datapoints is useful for normalising iterations relative to a Hessian
        computation,mas this requires averaging over each block also. It has no effect on the overall convergence of
        the Fisher, whilst for the Hessian it does. In a standalone case it acts more like a printout call!
        2. Don't make the batch size too large! Accurate EF relies on the sum of many low rank matricies.
        3. Smaller blocks slows things down because we save the parameters at each iteration!

    '''

    model.eval()
    vFv_acc = []
    Ftrace_last = 0.
    Ftrace_average_param = None
    F_flag = False

    iteration = 0
    batches = 0
    total_batches = 0.

    TFv = [torch.zeros(p.size()).to(device) for p in params]  # accumulate result

    while (iteration < max_iterations and not F_flag):
        model.zero_grad()
        #print(f'Iteration: {iteration}')

        for i, data in enumerate(data_loader, 1):

            model.zero_grad()

            inputs, labels = data[0].to(device), data[1].to(device)
            batch_size = inputs.size(0)
            # print('model ',next(model.parameters()).device)
            # for pname, p in model.named_parameters():
            #     print('model ',pname,' device ',p.device)
            # print('input ',inputs.device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            paramsH = []
            gradsH = []
            for paramH in model.parameters():
                if hasattr(paramH,'collect'):
                    if not paramH.collect:
                        continue
                paramsH.append(paramH)
                gradsH.append(0. if paramH.grad is None else paramH.grad + 0.)

            # Fisher Accumulation
            G2 = []
            for g in gradsH:
                G2.append(batch_size * g * g)
            TFv = [TFv_ + G2_ + 0. for TFv_, G2_ in zip(TFv, G2)]

            batches += 1
            total_batches += 1

            TFv_normed = [TFv_ / float(total_batches) for TFv_ in TFv]

            vFv = [torch.sum(x) for x in TFv_normed]

            vFv_c = np.array([i.detach().cpu().numpy() for i in vFv])

            Fdiff = np.abs((np.sum(vFv_c) - Ftrace_last) / (Ftrace_last + 1e-6))

            Ftrace_last = np.sum(vFv_c)

            if Fdiff < tol and iteration > min_iterations:
                F_flag = True

            if F_flag or iteration >= max_iterations:
                break

            if batches * batch_size >= max_number_of_datapoints:

                if save:
                    vFv_acc.append(vFv_c)

                batches = 0
                iteration += 1
                #print(f'Iteration: {iteration}')
                #print(Fdiff)

    #print('\nFdiff: ', Fdiff, ' F Iterations: ',
    #      (iteration) + batches * batch_size / np.minimum(max_number_of_datapoints, len(data_loader.dataset)), '\n')

    return vFv_c, vFv_acc


def Fisher_Full(model, device, data_loader, params, param_nums, criterion, max_iterations=100):
    model.eval()
    # loop over full dataset for each iteration
    g2_cum = np.zeros([np.sum(param_nums), np.sum(param_nums)])  # accumulate result

    datapoints = 0
    for n, data in enumerate(data_loader, 0):
        if n % 10 == 0:
            print('Batch: ', n)

        if n >= max_iterations:
            break
        model.zero_grad()

        inputs, labels = data[0].to(device), data[1].to(device)
        batch_size = inputs.size(0)

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        params = []
        grads = []
        for param in model.parameters():
            if not param.collect:
                continue
            params.append(param)
            grads.append(param.grad)

        G = torch.autograd.grad(loss, params)
        G = torch.cat([g.view(-1) for g in G])
        P = torch.outer(G, G)

        g2_cum = P.detach().cpu().numpy() * batch_size + g2_cum

        datapoints += batch_size

    P_ = g2_cum / datapoints

    return P_



# Plot Trace per Layer
def plot_average_hessian(y1,arch,show=False,name=''):

    x = np.arange(1, len(y1)+1)
    fig = plt.figure(tight_layout=True)
    plt.yscale('log')
    plt.xlabel('Blocks')
    plt.ylabel('Average Fisher Trace')
    plt.plot(x, y1, 'o-', label='Fisher')
    plt.legend()
    plt.grid(True, which='both')
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    save_name= 'Empirical Fisher Hessian '+arch + ' '+name
    plt.savefig('imgs/' + save_name + '.png')
    if show:
        plt.show()


def plot_full_hessian(hessian,arch,show=False,name=''):

    plt.clf()
    ax = sns.heatmap(hessian, linewidths=.5)
    #ax.matshow(self.score, aspect='auto', cmap=plt.get_cmap('Blues'))

    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    save_name = 'Full Hessian Empirical Fisher ' + arch + ' ' + name
    plt.savefig('imgs/' + save_name + '.png')
    if show:
        plt.show()


def EF_Hessian(model,arch, data_loader,name='',criterion=None,device=None,plot=False,Full=False):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        torch.cuda.set_device(device)
    if criterion:
        criterion.to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    names, param_nums, params = layer_accumulator(model)
    if Full:
        EF_full_hessian = Fisher_Full(model, device, data_loader, params, param_nums, criterion, max_iterations=200)
        layer_fisher = np.zeros([param_nums.size, param_nums.size])
        for i, ni in enumerate(param_nums):
            for j, nj in enumerate(param_nums):
                layer_fisher[i, j] = np.mean(EF_full_hessian[np.sum(param_nums[:i]):np.sum(param_nums[:i + 1]),
                                             np.sum(param_nums[:j]):np.sum(param_nums[:j + 1])])
        plot_full_hessian(hessian=layer_fisher,arch=arch,show=False,name=name)
        return layer_fisher
    else:
        Fisher, ffa = Fish(model, device, data_loader, params, criterion=criterion, tol=1e-3, min_iterations=0, max_iterations=200, max_number_of_datapoints=512, save=True)
        print(Fisher/param_nums)
        if plot:
            plot_average_hessian(Fisher/param_nums,arch,name=name)
        #print(f' EF of hessian trace: {Fisher/param_nums}')
        return (Fisher/param_nums)[1:]