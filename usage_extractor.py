import torch
from contextlib import contextmanager
import time
from collections import defaultdict
import copy
import pandas as pd
from time import sleep
import os
#from nncf.torch.layers import NNCFConv2d
import sys
sys.path.append('./scale-sim-v2')
from scalesim.scale_sim import scalesim1 as scalesim
import numpy as np

#context manager is the most acurate way to work on CPU, GPU need to be non syncronized
# Edge devices has not been tested yet

models_name = {'resnet50': 'resnet','resnet18': 'resnet','resnet20': 'resnet', 'resnet101': 'resnet','inceptionv3': 'inception',
               'mobilenetv2_100': 'mobilenet','mobilenetv1': 'mobilenet','resnet20_cifar10':'resnet'}


def eval_inference(model_name,bit_allocation=None):
    """ return the latency for model with given bitmap, as exist in the model latency table"""
    model_performance = pd.read_csv('Latency_table_'+model_name+'.csv').to_dict()
    latency = 0.0
    for layer_num in model_performance['Unnamed: 0'].keys():
        lat_bits = 'latency_' + str(bit_allocation[layer_num])
        if lat_bits in model_performance:
            latency+=model_performance[lat_bits][layer_num]
        else:
            raise TypeError("Didn't find the bit allocation in latenct table!")
    return latency

def eval_inference1(model_name,bit_allocation=None):
    """ return the latency for model with given bitmap, as exist in the model latency table"""
    model_performance = pd.read_csv('Latency_table_'+model_name+'.csv').to_dict()
    #print(model_performance)
    latency = 0.0
    for layer_num in model_performance['Unnamed: 0'].keys():

        if bit_allocation[layer_num] in [5,6,7]:
            lat_bits = 'latency_8'
            l1= model_performance[lat_bits][layer_num]
            lat_bits = 'latency_4'
            l2 = model_performance[lat_bits][layer_num]
            if bit_allocation[layer_num] == 5:
                latency +=(0.75*l2 + 0.25*l1)
            elif bit_allocation[layer_num] == 6:
                latency += (0.5 * l2 + 0.5 * l1)
            else:
                latency += (0.25 * l2 + 0.75 * l1)
        else:
            lat_bits = 'latency_' + str(bit_allocation[layer_num])
            latency+=model_performance[lat_bits][layer_num]
    return latency




class hw_descriptor():
    def __init__(self,name='scale',ArrayHeight=32,ArrayWidth=32,IfmapSramSzkB=64,FilterSramSzkB=64,
                 OfmapSramSzkB=64,Dataflow='os',Bandwidth=10,MemoryBanks=1,speed=200000000,new_hw=False):
        self.name=name #the name of the hardware, also the name of the config file
        self.run_name = '_'.join([name,str(ArrayHeight),str(ArrayWidth),Dataflow]) #eyeriss_12_14_ws
        self.ArrayHeight=ArrayHeight
        self.ArrayWidth=ArrayWidth
        self.IfmapSramSzkB=IfmapSramSzkB
        self.FilterSramSzkB=FilterSramSzkB
        self.OfmapSramSzkB=OfmapSramSzkB
        self.Dataflow=Dataflow
        self.Bandwidth=Bandwidth
        self.MemoryBanks=MemoryBanks
        self.speed=speed
        if new_hw:
            self.create_config()
    def create_config(self):
        path = os.path.join('scale-sim-v2','configs',run_name + '.cfg')
        #pass #TODO: create costum config.


class HW_simulator():
    def __init__(self,hw_descriptor,arch='resnet18',logpath='simulator_analitics',bits_rep=[2,3,4,8]):
        """
        """
        super(HW_simulator, self).__init__()
        self.hw_descriptor = hw_descriptor
        self.arch=arch
        self.logpath=logpath+'_'+ arch
        self.bits_rep=bits_rep
        self.config = os.path.join('scale-sim-v2', 'configs', self.hw_descriptor.name + '.cfg')
        self.topology = os.path.join('scale-sim-v2', 'topologies', 'conv_nets', self.arch + '.csv')

    def create_scalesim_reports(self):
        """
        first create 1 byte (8 bit) report and then factor for all presisions in self.bits_rep
        """
        print(f'create 8 bit report for {self.hw_descriptor.run_name} at {self.logpath}:')
        self._create_scalesim_latency_scheme(self.config,self.topology,self.logpath)

        print('create reports for quant nets')
        self._scale_scalesim_latency_scheme()
        print('=====================================================')
        print(f'HW reports for {self.hw_descriptor.run_name} complete')

    def _create_scalesim_latency_scheme(self,config,topology,logpath):
        s = scalesim(save_disk_space=True, verbose=True, config=config, topology=topology)
        s.run_scale(top_path=logpath)
    def _scale_scalesim_latency_scheme(self):
        ratios = self.bits_rep[2::-1] #4,3,2
        for ratio in ratios:
            quant_topo = pd.read_csv(self.topology)
            #quant_topo[' Strides'] *= ratio
            quant_topo[' Num Filter'] = (quant_topo[' Num Filter'] // ratio)+1 #**2
            df = pd.DataFrame(quant_topo)
            quant_topo_name = self.topology[:-4]+'_'+str(int(np.ceil(8/ratio)))+'.csv'
            df.to_csv(quant_topo_name,index=False)  # ,index=False)
            self._create_scalesim_latency_scheme(self.config,quant_topo_name,self.logpath+'_'+str(int(np.ceil(8/ratio))))
            #topology[' Channels'] = int(topology[' Channels']*ratio)
            #topology[' Num Filter'] = int(topology[' Num Filter'] * ratio)
            #quant activations by size, quant filters by channels?
    def create_latency_table(self,model):
        """
        create a latency table for all precisions by checking the maximum cycles between bw and compute for each layer
        """
        model_performance = defaultdict(dict)
        #model_performance = pd.read_csv('Latency_table_' + self.arch + '.csv') #if exist?
        for bits in self.bits_rep:
            # paths
            logpath = self.logpath if bits==8 else self.logpath+'_'+str(bits)
            report_path = os.path.join(logpath,self.hw_descriptor.run_name)
            bw_report_path = os.path.join(report_path,'BANDWIDTH_REPORT.csv')
            compute_reprt_path = os.path.join(report_path,'COMPUTE_REPORT.csv')

            #load bw and compute the df
            bw_report = pd.read_csv(bw_report_path)
            compute_reprt = pd.read_csv(compute_reprt_path)
            net_topology = pd.read_csv(self.topology)

            #for i in range(len(compute_reprt['LayerID'])):
            i=0
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Conv2d):
                    compute_cycles = compute_reprt[' Total Cycles'][i]
                    compute_latency = 1000 * compute_cycles / self.hw_descriptor.speed #msec
                    #print(f'compute: {compute_latency}')
                    if i<len(compute_reprt['LayerID'])-1:
                        out_bits = bits * net_topology[' Num Filter'][i] * net_topology[' IFMAP Height'][i+1] * \
                                   net_topology[' IFMAP Width'][i+1]
                    else:
                        out_bits = bits * net_topology[' Num Filter'][i] * net_topology[' IFMAP Height'][i] * \
                                   net_topology[' IFMAP Width'][i]

                    output_dram_lat = 1000 * (1/ bw_report[' Avg OFMAP DRAM BW'][i] * 8) * out_bits / self.hw_descriptor.speed
                    #if bits in [4,8] and i<5:
                        #print(f'mem: {output_dram_lat} for {out_bits} bits')
                        #print(bw_report[' Avg OFMAP DRAM BW'][i])
                    model_performance['latency_'+str(bits)][name] = max(compute_latency,output_dram_lat) #compute_reprt['LayerID'][i]i
                    i+=1
            #print(sum(list(model_performance['latency_'+str(bits)].values())))
        df = pd.DataFrame(model_performance)
        df.to_csv('Latency_table_' + self.arch + '.csv')  # ,index=False)
        #print(model_performance)


def create_model_latency_scheme(model,arch,ds,device, reps=10):
    """
    create model latency table
    model: torch model
    arch: str of the model arch
    ds: dataset used (to create dummy input for inference)
    device: as we measure latency over a given device
    reps: num of reps for warmup the hardware before timing

    return: the dict saved in Latency table
    """
    model_performance = defaultdict(dict)
    ##model
    model = model.to(device)
    if models_name[arch]=='resnet':
        resnet_convs_names = ['conv1','shortcut.0'] #TODO: same for all models supported
        if ds=='imagenet':
            resnet_convs_names = ['conv1', 'downsample.0']
    else: #mobilenet_v2
        resnet_convs_names = ['conv1', 'downsample.0']
    # create input
    img_shape = (1, 3, 32 , 32)
    if ds=='imagenet':
        img_shape = (1, 3, 256, 256)
    x = torch.rand(img_shape, dtype=torch.float).to(device)
    dummy = torch.clone(x)

    #cuda events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # warmup:
        for _ in range(max(reps, 100)):
            model(x)
        initial_conv=True
        for name, m in model.named_modules():
            if isinstance(m,torch.nn.Conv2d) or isinstance(m,NNCFConv2d): #NNCFConv2d
                # save features for the shortcut
                if resnet_convs_names[0] in name:
                    dummy = torch.clone(x)

                start.record()
                if resnet_convs_names[1] in name:
                    for _ in range(reps):
                        m(dummy)
                else:
                    for _ in range(reps):
                        m(x)
                end.record()
                torch.cuda.synchronize()
                time = start.elapsed_time(end)
                if resnet_convs_names[1] not in name:
                    x = m(x)
                if initial_conv:
                    initial_conv = False
                    continue
                model_performance['latency_32'][name] = time / reps


    conv_time=0
    df = pd.DataFrame(model_performance)
    df.to_csv('Latency_table_'+arch+'.csv')#,index=False)
    for layer in model_performance['latency_32'].keys():
        print(f"L: {layer} | TIME: {model_performance['latency_32'][layer]}")
        conv_time+=model_performance['latency_32'][layer]
    print(f'==== inference estimated time:: {conv_time} ====')
    _scale(arch)
    return model_performance

def _scale(arch):
    """
    scaling the model latency scheme by model and by bit-width
    """
    model_name = models_name[arch]
    factor_dict = defaultdict(dict)
    #from TVM: https://tvm.apache.org/2019/04/29/opt-cuda-quantized
    #mobilenet: https://pocketflow.github.io/performance/
    factor_dict['8'] = {'resnet': 0.37836,'mobilenet': 0.400,'vgg': 0.31103,'inception': 0.19923,'resnext': 0.12355}
    #from TensortRT: https://developer.nvidia.com/blog/int4-for-ai-inference/
    # and from "Quantization and Deployment of Deep Neural Networks on Microcontrollers" Novac et al

    for k in factor_dict['8']:
        factor_dict['16'][k] = factor_dict['8'][k] * 1.39177
        factor_dict['7'][k] = factor_dict['8'][k] * 0.9725
        factor_dict['6'][k] = factor_dict['8'][k] * 0.9289
        factor_dict['5'][k] = factor_dict['8'][k] * 0.8937
        factor_dict['4'][k]  = factor_dict['8'][k] * 0.85783
        factor_dict['3'][k]  = factor_dict['8'][k] * 0.6338
        factor_dict['2'][k]  = factor_dict['8'][k] * 0.4545
        #factor_dict['1'][k]  = factor_dict['8'][k] * 0.3165
    model_performance = pd.read_csv('Latency_table_'+arch+'.csv')
    model_performance['latency_16'] = model_performance['latency_32']*factor_dict['16'][model_name]
    model_performance['latency_8']  = model_performance['latency_32']*factor_dict['8'][model_name]
    model_performance['latency_7']  = model_performance['latency_32']*factor_dict['7'][model_name]
    model_performance['latency_6']  = model_performance['latency_32']*factor_dict['6'][model_name]
    model_performance['latency_5']  = model_performance['latency_32']*factor_dict['5'][model_name]
    model_performance['latency_4']  = model_performance['latency_32']*factor_dict['4'][model_name]
    model_performance['latency_3']  = model_performance['latency_32'] * factor_dict['3'][model_name]
    model_performance['latency_2']  = model_performance['latency_32']*factor_dict['2'][model_name]
    #model_performance['latency_1']  = model_performance['latency_32']*factor_dict['1'][model_name]
    print(model_performance)
    df = pd.DataFrame(model_performance)
    df.to_csv('Latency_table_'+arch+'.csv',index=False)



def layers_for_quant_list(model_name):
    "return a list of all quantize conv layers"
    model_performance = pd.read_csv('Latency_table_' + model_name + '.csv').to_dict()
    return list(model_performance['Unnamed: 0'].values())

def create_bit_allocation(method,bit_allocation,q_layers_list,quant_selection={},init=None):
    """
    assign the quant_selection for the layers in q_layers_list in bit_config
    init- in not None, use the init value TODO: should implement smart initilization
    """
    if init:
        quant_selection = [init] * len(q_layers_list)
    for i,layer_name in enumerate(q_layers_list):
        bit_allocation[layer_name] =quant_selection[i]
    return bit_allocation

def assign_bit_allocation2(model, bit_allocation):
    act_bits = []
    for name, m in model.named_modules():
        name= name[7:]
        # remove 'pre_ops.0.op' from name
        if name[:-13] in bit_allocation:
            #print(f'name: {name[:-13]}')
            if name.split('.')[-1] == 'op':
                layer_bits = bit_allocation[name[:-13]]
                m.num_bits = layer_bits
                if 'shortcut' not in name and 'downsample' not in name:
                    act_bits.append(layer_bits)
        elif 'external_quantizers.' in name and 'layer' in name and 'relu' in name:  # and name is not 'external_quantizers':
            bits = act_bits.pop(0)
            m.num_bits = bits

def check_model_size(model,bitwidths):
    total_bits=0
    bitwidths = [32]+bitwidths+[32]
    i=0
    for name, m in model.named_modules():
        if 'conv' not in name and 'Linear' not in name:
            continue
        bits=sum(param.numel() for param in m.parameters())*bitwidths[i]
        total_bits+=bits
        if bits>250 and 'bn' not in name and 'BatchNorm' not in name:
            i+=1
    print(f'Model size: {total_bits*1.25e-7} MB')
    return total_bits*1.25e-7



##### for CPU only #####
@contextmanager
def timer(name=None):
    start = time.time()
    yield # context breakdown
    end = time.time()
    print(f"{name} executed in {round(end - start, 5)} seconds.")

#contextmanager class
class CM_tinme:
    def __init__(self):
        self.time = 0
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self,exc_type, exc_obj, exc_tb):
        self.end = time.time()
        self.time = self.end - self.start
        return time
