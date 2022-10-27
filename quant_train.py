import argparse
import os
import random
import shutil
import time
import logging
import warnings
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict,defaultdict
import datetime



import torch
#import nncf  # Important - should be imported directly after torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.data_utils import dataloader
from utils.amp import AMP

from tensorboardX import SummaryWriter
import torchvision.models as tvmodels
import utils.models.resnet_cifar as resnet_cifar
from utils.models.mobilenetv1 import *
import timm


#from nncf import NNCFConfig
#from nncf.torch import create_compressed_model, register_default_init_args


from bit_config import *
from utils import *
from usage_extractor import *
from Latency_loss import Quantize_scores
import quan

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--ds', type=str,default='imagenet',
                    help='imagenet / cifar10 / cifar100')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('--teacher-arch',
                    type=str,
                    default='resnet101',
                    help='teacher network used to do distillation')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-qf', '--quant-freq', default=5, type=int,
                    metavar='N', help='quant print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save-path',
                    type=str,
                    default='checkpoints/imagenet/test/',
                    help='path to save the quantized model')
parser.add_argument('--data-percentage',
                    type=float,
                    default=1,
                    help='data percentage of training data')
parser.add_argument('--checkpoint-iter',
                    type=int,
                    default=-1,
                    help='the iteration that we save all the featuremap for analysis')
parser.add_argument('--resume-quantize',
                    action='store_true',
                    help='if True map the checkpoint to a quantized model,'
                         'otherwise map the checkpoint to an ordinary model and then quantize')
parser.add_argument('--resume-metadata',
                    action='store_true',
                    help='if True resume metadata')
parser.add_argument('-q-method',
                    type=str,
                    default='lsq',
                    help='lsq or nncf')


##==================KD===========
parser.add_argument('--distill-method',
                    type=str,
                    default='None',
                    help='you can choose None or KD_naive')
parser.add_argument('--distill-alpha',
                    type=float,
                    default=0.95,
                    help='how large is the ratio of normal loss and teacher loss')
parser.add_argument('--temperature',
                    type=float,
                    default=6,
                    help='how large is the temperature factor for distillation')
##==================end of KD===========

parser.add_argument('--create_compressed_model',
                    action='store_true',
                    help='flag to create latency statistics file')
parser.add_argument('--create_sim',
                    action='store_true',
                    help='flag to create latency statistics file')
parser.add_argument('--create_table',
                    action='store_true',
                    help='flag to create latency statistics file')
parser.add_argument('--alpha',
                    type=float,
                    default=1.,
                    help='CE coefficient')
parser.add_argument('--beta',
                    type=float,
                    default=1.,
                    help='Latency coefficient')
parser.add_argument('-T', default=1., type=float,
                    help='temperture for the latency loss')


parser.add_argument('--mem_size', default=64, type=float,
                    help='memory size for the HW simulator')
parser.add_argument('--array_size', default=32, type=float,
                    help='memory size for the HW simulator')


parser.add_argument('--inter_rep_update', action='store_true',
                    help='enable inter_rep_update')
parser.add_argument('-EMA', default=1., type=float,
                    help='exponential moving avarage for the quant score. -1 means deterministic update')
parser.add_argument('--retrain-quant-model',
                    action='store_true',
                    help='if True take a quantized model and continue training it')
args = parser.parse_args()


## =============== save path and dirs for experiments ===============
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
datatime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
args.save_path = os.path.join(args.save_path, datatime_str)
args.save_path = args.save_path + '_al_' + str(args.alpha) + '_be_' + str(args.beta) + '_ema_' + str(args.EMA)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
filename_log = os.path.join(args.save_path, 'log.log')
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filename=filename_log)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())

logging.info(args)
## =============== ================================== ===============


## =============== global macros ===============
best_net_lat,best_net_score,best_net_acc = 1e5,0,0
bits_rep = [2,3,4,8]
_n_layers= {'resnet18':20,'resnet20':23 ,'resnet50':55,'mobilenetv2_100':53,'mobilenetv1':27,'mobilenetv3_large_100':61}
uniform_bits = lambda arch,init_bits:   [init_bits]*_n_layers[arch]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
calc_preformence = lambda ACC,LAT:   ACC /  LAT #to define best model so far...
## =============== ============= ===============


## =============== Hardware macros ==============
#for existing configs (scale/scale_mem/google/eyeriss)
#hw_args = ('scale',args.array_size,args.array_size,args.mem_size,args.mem_size,args.mem_size,'os',10,1,200000000,False)
#hw_args = ('scale_mem',args.array_size,args.array_size,args.mem_size,args.mem_size,args.mem_size,'os',10,1,200000000,False)
hw_args = ('eyeriss',12,14,108,108,108,'ws',10,1,200000000,False)
## =============== ============= ===============


def main():
    mckp_18 = [8, 7, 6, 7, 5, 7, 6, 5, 6, 4, 4, 5, 4, 4, 3, 5, 3, 2, 2, 3]
    ddq_18 = [8, 7, 6, 6, 6, 6, 6, 7, 6, 6, 6, 5, 3, 6, 3, 3, 4, 4, 5, 3]
    hawq_18 = [8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 8]
    mckp_50 = [8, 7, 8, 5, 5, 4, 5, 4, 4, 4, 4, 5, 5, 3, 4, 3, 4, 3, 4, 4, 4, 4, 3, 4, 3, 5, 3, 4, 3, 3, 3, 3, 2, 3, 3,
               2, 3, 3, 2, 3, 4, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 2]
    hawq_50 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 4, 8, 8, 8, 8, 8, 8, 4, 4,
               8, 4, 8, 8, 4, 8, 8, 4, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8]
    our_50_1=   [8, 4, 8, 8, 4, 4, 4, 3, 2, 4, 4, 4, 4, 3, 8, 2, 3, 4, 2, 2, 4, 3, 3, 4, 8, 2, 2, 2, 8, 4, 3, 4, 8, 8, 3,
               4, 8, 3, 2, 4, 2, 4, 8, 8, 4, 8, 4, 2, 2, 2, 3, 4, 4]

    mckp_mobile = [6, 8, 8, 5, 7, 6, 4, 7, 6, 5, 5, 6, 4, 6, 5, 3, 5, 5, 4, 4, 5, 3, 5, 3, 2, 5, 3, 3, 5, 3, 3, 5, 3, 2,
                   4, 3, 2, 4, 3, 3, 3, 4, 2, 4, 3, 2, 4, 2, 2, 2, 2, 2]
    ddq_mobile = [8, 8, 8, 8, 7, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7, 6, 6, 7, 6, 7, 6, 5, 5, 5, 5, 5, 6, 5, 5, 6, 6, 4,
                  6, 4, 4, 6, 4, 3, 4, 4, 4, 6, 4, 3, 6, 4, 3, 8, 4, 3]
    haq_mobile = [6, 4, 4, 6, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 6, 3, 3, 6,
                  3, 3, 6, 3, 3, 6, 4, 4, 6, 4, 4, 5, 4, 4, 5, 5, 5, 6]

    """model, teacher = create_model(args)
    check_model_size(model, our_50_1)
    exit(0)"""
    global writer
    writer = SummaryWriter()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen a seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def create_model(args):
    teacher=None
    logging.info(f"=> pytorch model {args.arch} for {args.ds} pretrained: {args.pretrained}")
    if "cifar" in args.ds:
        num_classes = 10 if args.ds =="cifar10" else 100
        if args.arch == 'resnet20':
            model = resnet_cifar.ResNet20(num_classes)
        elif args.arch== 'resnet18':
            model = resnet_cifar.ResNet18(num_classes)
    else:
        #model = tvmodels.__dict__[args.arch](pretrained=pretrained)
        if args.arch=='mobilenetv2_100':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        elif args.arch=="mobilenetv1":
            model = mobilenetv1()
        else:
            model = timm.create_model(args.arch, pretrained=args.pretrained)

        """for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                logging.info(name)"""
        #save a pretrained model
        if args.distill_method != 'None':
            if args.teacher_arch:
                teacher = timm.create_model(args.teacher_arch, pretrained=True)
            else:
                logging.info("Couldn't find teacher arch, preforming self- distillation")
                teacher = timm.create_model(args.arch, pretrained=True)


    if args.q_method == 'lsq':
        list_bit = uniform_bits(args.arch, 8)
        quan_scheduler = quan.create_quan_scheduler()
        modules_to_replace = quan.find_modules_to_quantize(model, quan_scheduler,list_bit)
        model = quan.replace_module_by_names(model, modules_to_replace)
        if args.pretrained:
            model = load_pretrained_quant_model(model,args.ds,args.arch)
    elif args.q_method=='nncf': #does not officially support
        conf_name = f'{args.arch}_{args.ds}_manual_config.json'
        nncf_config = NNCFConfig.from_json(conf_name)
        cudnn.benchmark = True
        nncf_config = register_default_init_args(nncf_config, train_loader, val_loader=val_loader)
        compression_ctrl, model = create_compressed_model(model, nncf_config, dump_graphs=False)
    return model,teacher


def load_pretrained_quant_model(model,ds,arch,pretrained_dir="pretrained_quant"):
    name = arch+'_'+ds+'.pth.tar'
    file = os.path.join(".",pretrained_dir,arch,name)
    if os.path.isfile(file):
        logging.info("=> loading pretrained model '{}'".format(file))
        checkpoint = torch.load(file)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_state_dict[k[7:]] = v  # remove `module.`
        model.load_state_dict(new_state_dict)
    else:
        logging.info("=> no checkpoint found in '{}', train from scratch".format(file))
    return model


def update_model(model,quan_scheduler,bit_allocation):
    state_dict = model.state_dict()
    modules_to_replace = quan.find_modules_to_quantize(model, quan_scheduler, bit_allocation)
    model = quan.replace_module_by_names(model, modules_to_replace)
    model.load_state_dict(state_dict, strict=False)
    return model

def main_worker(gpu, ngpus_per_node, args):
    global best_net_acc
    global best_net_lat
    global best_net_score
    args.gpu = gpu

    if args.gpu is not None:
        logging.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # load DS
    train_loader, val_loader, train_sampler, ds_length = dataloader(args, download=True)

    # create model
    model, teacher = create_model(args)
    quan_scheduler = quan.create_quan_scheduler()

    #HW simulator from scale-sim:
    hw_des = hw_descriptor(*hw_args)
    simulator = HW_simulator(hw_des,arch=args.arch,logpath='simulator_analitics',bits_rep=bits_rep)

    #check_model_size(model)
    show_model_presision(model)


    if args.create_table:
        # create the reports for all bits- (this step take some time)
        if args.create_sim:
            simulator.create_scalesim_reports()
        simulator.create_latency_table(model)

    bit_allocation = uniform_bits(args.arch, 8)#[8]*100 #just for refrence for all models up to 100 layers..
    ref_lat = eval_inference(args.arch,bit_allocation)
    latencies = []

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            if args.distill_method != 'None':
                teacher.cuda(args.gpu)
                teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            if args.distill_method != 'None':
                teacher.cuda(args.gpu)
                teacher = torch.nn.parallel.DistributedDataParallel(teacher)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        #model = model.cuda(args.gpu)
        model = torch.nn.DataParallel(model).cuda(args.gpu)
        if args.distill_method != 'None':
            teacher = teacher.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda(args.gpu)
        if args.distill_method != 'None':
            teacher = torch.nn.DataParallel(teacher).cuda(args.gpu)

    # Resume training from checkpoint
    if args.resume and not args.resume_quantize:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)['state_dict']
            model.load_state_dict(checkpoint)
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    if args.resume and args.resume_quantize: #todo: load correct bit allocation?
        args.retrain_quant_model = True
        if os.path.isfile(args.resume):
            logging.info("=> loading quantized checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            bit_allocation = checkpoint['bit_allocation']
            best_net_acc = checkpoint['best_net_acc']
            best_net_lat = checkpoint['best_net_lat']
            if args.gpu is not None:
                best_net_acc = best_net_acc.to(args.gpu)
                # best_net_lat = best_net_lat.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            assign_bit_allocation2(model,bit_allocation)
            latencies.append(eval_inference(args.arch, bit_allocation)/ref_lat)
            logging.info("=> accuracy '{}', latency '{}', epoch '{}'".format(checkpoint['best_net_acc'],
                                                                             checkpoint['best_net_lat'],
                                                                             checkpoint['epoch']))
        else:
            logging.info("=> no quantized checkpoint found at '{}'".format(args.resume))


    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    #if "mobilenet" in args.arch:
        #optimizer = torch.optim.AdamW(model.parameters(), args.lr,weight_decay=args.weight_decay)
    #optimizer = AMP(model.parameters(), lr=args.lr,weight_decay=args.weight_decay, momentum=args.momentum,epsilon=0.5)

    ##we currently use custome scheduler for quantization acceleration via the optimizer
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)

    # optionally resume optimizer and meta information from a checkpoint
    if args.resume_metadata:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_net_acc = checkpoint['best_net_acc']
            best_net_lat = checkpoint['best_net_lat']
            if args.gpu is not None:
                best_net_acc = best_net_acc.to(args.gpu)
            #     best_net_lat = best_net_lat.to(args.gpu)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=> loaded optimizer and meta information from checkpoint '{}' (epoch {})".
                         format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    latencies.append(eval_inference(args.arch, bit_allocation)/ref_lat)

    logging.info("=> Initial Cross entropy loss for calibration MP algorithm")
    CE_loss = 2#train(train_loader, model, criterion, optimizer, 0, args)


    ##collect statistics for mobilenet
    """for layer in model.modules():
        if hasattr(layer,'collect_stats'):
            layer.collect_stats=True
    validate(val_loader, model, criterion, args)
    for layer in model.modules():
        if hasattr(layer,'collect_stats'):
            layer.collect_stats=False"""

    Q_scores = Quantize_scores(n_layers=len(bit_allocation), bits_rep=bits_rep, T=args.T, alpha=args.alpha,
                               beta=args.beta,ref_lat=ref_lat,ref_ce =CE_loss,EMA =args.EMA)
    logging.info(f'Latencty: {latencies[-1]}')
    logging.info(f'bit_allocation = {bit_allocation}')
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        CE_loss = train(train_loader, model, criterion, optimizer, epoch, args,teacher)
        #scheduler.step()
        acc1 = validate(val_loader, model, criterion, args)
        curr_bit_allocation = bit_allocation
        if epoch % args.quant_freq == 0 and epoch != 0:# and not args.retrain_quant_model:

            logging.info(f'last bit_allocation = {bit_allocation}')
            latencies.append(eval_inference(args.arch, bit_allocation)/ref_lat)


            Q_scores.update_quant_scores(CE_loss, latencies[-1])
            #bit_allocation = assign_bit_allocation(bit_allocation, quant_layers_list, Q_scores.select_quantization(stochastic=True))



            bit_allocation= Q_scores.select_quantization(stochastic=False)
            logging.info('-----')
            logging.info(bit_allocation)
            logging.info('-----')
            logging.info(Q_scores.score)
            logging.info('-----')
            #assign_bit_allocation2(model,bit_allocation)
            update_model(model, quan_scheduler, bit_allocation)

            #show_model_presision(model)
            #print('our quant model:')
            #print(model)
            #Q_scores.plot_scores(save_name=args.arch+'_'+str(epoch))

        net_score = calc_preformence(acc1, latencies[-1])
        #todo: delete
        """if args.retrain_quant_model:
            net_score = acc1"""

        logging.info(f' * Score [{net_score}] Acc@1 [{acc1}] Latency [{latencies[-1]}]')
        logging.info(f'                       scaled CE: {args.alpha * CE_loss} |scaled lat: {args.beta * latencies[-1]} ')


        # remember best acc@1 and save checkpoint
        is_best = net_score >= best_net_score
        if acc1 > best_net_acc+.5 and best_net_lat + 0.05 > latencies[-1]:
            logging.info('** replaced best by small margin **')
            is_best = True
        best_net_score = max(net_score, best_net_score)
        writer.add_scalar("Best_net_score", best_net_score, epoch)

        if is_best:
            best_epoch = epoch
            best_net_acc = acc1 # the network that achive the best score
            best_net_lat = latencies[-1]
        logging.info(f'==Best net==: epoch: {best_epoch}: [{best_net_score}] accuracy [{best_net_acc}] latency [{best_net_lat}]')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            """if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)"""

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_net_acc': best_net_acc,
                'best_net_lat': best_net_lat,
                'best_score': best_net_score,
                'optimizer': optimizer.state_dict(),
                "bit_allocation": curr_bit_allocation,
            }, is_best, args.save_path)
    #plot_latency(latencies)
    writer.export_scalars_to_json("./tensorboard.json")
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args,teacher=None):
    batch_time = AverageMeter('Time', ':6.3f')
    #data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(len(train_loader),[batch_time, losses, top1, top5],prefix="Epoch: [{}]".format(epoch))

    model.train()
    if teacher:
        teacher.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        if args.distill_method != 'None':
            with torch.no_grad():
                teacher_output = teacher(images)
            loss = loss_kd(output, target, teacher_output, args)
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        writer.add_scalar("Train_Loss", loss.item(), epoch)
        writer.add_scalar("Train_Acc1", top1.avg, epoch)
        writer.add_scalar("Train_Acc5", top5.avg, epoch)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #adjust_learning_rate_step(optimizer, epoch, args, i)
        adjust_learning_rate(optimizer, epoch, args, i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    return top1.avg


def save_checkpoint(state, is_best, path=None):
    filename = os.path.join(path ,'checkpoint.pth.tar')
    torch.save(state, filename )
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def loss_kd(output, target, teacher_output, args):
    """
    Compute the knowledge-distillation (KD) loss given outputs and labels.
    "Hyperparameters": temperature and alpha
    The KL Divergence for PyTorch comparing the softmaxs of teacher and student.
    The KL Divergence expects the input tensor to be log probabilities.
    """
    alpha = args.distill_alpha
    T = args.temperature
    KD_loss = F.kl_div(F.log_softmax(output / T, dim=1), F.softmax(teacher_output / T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(output, target) * (1. - alpha)

    return KD_loss

def adjust_learning_rate(optimizer, epoch, args, step_num):
    step = 30 if args.ds == 'imagenet' else 60
    decay = 0.1 if args.ds == 'imagenet' else 0.5
    if "mobile" in args.arch:
        step=20
    lr = args.lr * (decay ** (epoch // step))
    """if "mobile" in args.arch:
        lr = args.lr * (decay ** epoch)"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_step(optimizer, epoch, args, step_num):
    step = 30 if args.ds == 'imagenet' else 60
    decay = 0.1 if args.ds == 'imagenet' else 0.5
    lr = args.lr * (decay ** (epoch // step))
    if step_num < 250:
        lr *= step_num/250
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate1(optimizer, epoch, bit_allocation, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    step = 30 if args.ds == 'imagenet' else 60
    lr = args.lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    """for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        if len(param_group['name']) > 10:
            name = param_group['name'][7:]
        if name in bit_allocation:
            if bit_allocation.get(name) == 2:
                param_group['lr'] = lr *2"""

def plot_latency(latencies,save_name='A graph of latency'):
    plt.clf()
    plt.plot(range(len(latencies)),latencies,'g^')
    plt.ylabel('latency')
    plt.xlabel('epoch')
    if save_name:
        plt.savefig('imgs/' + save_name + '.png')


def show_model_presision(model):
    for name, m in model.named_modules():
        if hasattr(m,'bits'):
            logging.info(f'{name},{m.bits}')


if __name__ == '__main__':
    main()
