import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os


class Quantize_scores(nn.Module):
    def __init__(self,n_layers,bits_rep=[2,3,4,8],sampling_method="MH",T=1.,alpha=0.1,beta=100,ref_lat=1.,ref_ce=0.07,EMA=True):#,Hessian=None): #max: 1.84372
        """
        Implemented as Random walk -MH
        n_layers: num of layers to quantize
        bits_rep: bit representation for each layer
        T: temperature for the score softmax
        alpha: CE_loss coefficient
        beta: Latency_loss coefficient
        max_lat: float32 network latency
        """
        #todo: rw-mh
        super(Quantize_scores, self).__init__()
        self.n_layers = n_layers
        self.bits_rep = bits_rep
        self.sampling_method = sampling_method
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.ref_lat = ref_lat
        self.ref_ce = ref_ce
        self.ref_zeta = self.alpha * self.ref_ce + self.beta * self.ref_lat
        _shape = (n_layers, len(bits_rep))
        self.score = torch.rand(_shape)
        self.EMA= EMA #update score using EMA
        #self.curr_choosen_quant=[]
        #for _ in range(n_layers):
        #    self.curr_choosen_quant.append(np.random.choice([0,1,2,3]))
        # the current quantization
        self.curr_choosen_quant = [2]*n_layers



    def select_quantization(self,stochastic=False):
        prob_score = F.softmax(self.score / self.T, dim=1)
        quant_selection=[]
        #self.curr_choosen_quant=[]
        if self.sampling_method == "MH":
            for i,layer_prob in enumerate(prob_score):
                theta_star = np.random.choice(self.bits_rep, p=layer_prob.numpy())
                curr_theta = self.bits_rep[self.curr_choosen_quant[i]]
                g_theta_star = layer_prob[self.bits_rep.index(theta_star)]
                g_curr_theta = layer_prob[self.bits_rep.index(curr_theta)]
                a = g_theta_star / g_curr_theta
                if a >= 1 or a > np.random.rand():
                    quant_selection.append(theta_star)
                    #print('update MH')
                else:
                    quant_selection.append(curr_theta)
                self.curr_choosen_quant[i] = self.bits_rep.index(quant_selection[-1])
                #print(self.curr_choosen_quant[i])
        else:
            for p_layer in prob_score:
                if stochastic:
                    quant_selection.append(np.random.choice(self.bits_rep, p=p_layer.numpy()))
                else:
                    quant_selection.append(self.bits_rep[torch.argmax(p_layer)])
                #self.curr_choosen_quant.append(torch.argmax(p_layer))
            self.curr_choosen_quant = quant_selection
        return quant_selection

    def update_quant_scores(self,CE_loss,Lat,inter_rep_update=False): #hessian_aware=None
        """
        CE_loss: CE_loss
        Lat: Latency of the quant network, normelized with 8-bit as max
        epoch: epoch
        inter_rep_update: update probs for abobe and below representations
        """
        Lat_loss = Lat #/ self.max_lat
        #print(self.curr_choosen_quant)
        mask = F.one_hot(torch.tensor(self.curr_choosen_quant), num_classes=len(self.bits_rep))

        if inter_rep_update:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            idxes = torch.where(mask == 1)[1]
            if Lat > self.ref_lat * .9:
                for line, i in enumerate(idxes):
                    mask[line, :i] = 1
            elif CE_loss > 0.07: #think about it...
                for line, i in enumerate(idxes):
                    mask[line, i:] = 1
        zeta = self.alpha * CE_loss + self.beta * Lat_loss

        #option 1 - linear scale
        #d_scores = mask * ((1/zeta) - (1/ self.ref_zeta))
        # option 2 - log/ln scale
        d_scores = mask * torch.log(torch.tensor(self.ref_zeta/zeta))
        d_scores /= torch.log(torch.tensor(10)) #delete for ln
        #option 4 - smoother log
        #d_scores = mask * torch.log(1 / (zeta+self.ref_zeta))
        if self.EMA==-1:
            self.score += d_scores
        else:
            self.score = (1-self.EMA) * self.score + self.EMA * d_scores

    def plot_scores(self,save_name=None):
        #torch.save(self.score,'imgs/'+save_name+'.pt')
        #return
        # plot score matrix
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.matshow(self.score, aspect='auto', cmap=plt.get_cmap('Blues'))
        plt.ylabel('Layer')
        #plt.yticks(range(n_layers), classes)
        plt.xlabel('Quant level score')
        plt.xticks(range(len(self.bits_rep)), self.bits_rep)
        for (i, j), z in np.ndenumerate(self.score):
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
        if save_name:
            if not os.path.exists('imgs'):
                os.makedirs('imgs')
            plt.savefig('imgs/'+save_name+'.png')