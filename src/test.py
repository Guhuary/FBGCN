#from __future__ import division
#from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from sample import Sampler
from metric import accuracy, roc_auc_compute_fn
# from deepgcn.utils import load_data, accuracy
# from deepgcn.models import GCN

from utils import load_citation, load_reddit_data
from network import *
from network_fr import *

# Training settings
parser = argparse.ArgumentParser()
# Training parameter 
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Disable validation during training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.02,
                    help='Initial learning rate.')
parser.add_argument('--lradjust', action='store_true',
                    default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument("--mixmode", action="store_true",
                    default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--dataset', default="citeseer", help="The data set")
parser.add_argument('--datapath', default="../FBGCN/data/", help="The data path.")
parser.add_argument('--save_path', default="../FBGCN/src/Models/", help="The path that save the models.")

# Model parameter
parser.add_argument('--type',default='resgcn',
                    help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
parser.add_argument('--inputlayer', default='gcn',
                    help="The input layer of the model.")
parser.add_argument('--outputlayer', default='gcn',
                    help="The output layer of the model.")
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--withbn', action='store_true', default=True,
                    help='Enable Bath Norm GCN')
parser.add_argument('--withloop', action="store_true", default=False,
                    help="Enable loop layer GCN")
parser.add_argument('--nhiddenlayer', type=int, default=2,
                    help='The number of hidden layers.')
parser.add_argument('--use_known_lable', action="store_true", default=True,
                    help="Use train & valid lable or not in test")
parser.add_argument("--normalization", default="AugNormAdj",
                    help="The normalization on the adj matrix.")
# parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
parser.add_argument("--nbaseblocklayer", type=int, default=1,
                    help="The number of layers in each baseblock")
parser.add_argument("--aggrmethod", default="default",
                    help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

args = parser.parse_args()
# pre setting
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()
if args.aggrmethod == "default":
    if args.type == "resgcn":
        args.aggrmethod = "add"
    else:
        args.aggrmethod = "concat"

if args.type == "mutigcn":
    #print("For the multi-layer gcn model, the aggrmethod is fixed to nores and nhiddenlayers = 1.")
    #args.nhiddenlayer = 1
    args.aggrmethod = "nores"

# random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda or args.mixmode:
    torch.cuda.manual_seed(args.seed)

# should we need fix random seed here?
sampler = Sampler(args.dataset, args.datapath, args.task_type)

# get labels and indexes
labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

# The model
model = GCNModel(nfeat=nfeat,
                 nhid=args.hidden,            
                 nclass=nclass,
                 nhidlayer=args.nhiddenlayer,  
                 dropout=args.dropout,         
                 baseblock=args.type,
                 inputlayer=args.inputlayer,      
                 outputlayer=args.outputlayer,  
                 nbaselayer=args.nbaseblocklayer,
                 activation=F.relu,
                 withbn=args.withbn,           
                 withloop=args.withloop,       
                 aggrmethod=args.aggrmethod,   
                 mixmode=args.mixmode)

PATH = os.path.join(args.save_path, '{}_{}_l_{}.ckpt'.format(args.type, args.dataset, args.nhiddenlayer+2))
model.load_state_dict(torch.load(PATH))

# convert to cuda
if args.cuda:
    model.cuda()

# For the mix mode, lables and indexes are in cuda. 
if args.cuda or args.mixmode:
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def test(test_adj, test_fea):
    model.eval()
    output,layer_sizes = model(test_fea, test_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])
    return (loss_test.item(), acc_test.item(), layer_sizes)

# Testing
(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)

if args.mixmode:
    test_adj = test_adj.cuda()
    test_fea = test_fea.cuda()

(loss_test, acc_test, layer_sizes) = test(test_adj, test_fea)
print("%.6f\t%.6f\t" % (loss_test,  acc_test))

model2 = GCNModel_fsp(nfeat=nfeat,
                 nhid=args.hidden,              #128
                 nclass=nclass,
                 nhidlayer=args.nhiddenlayer,   #1
                 dropout=args.dropout,          #0.5
                 layer_sizes=layer_sizes,
                 baseblock=args.type,
                 inputlayer=args.inputlayer,    #gcn   
                 outputlayer=args.outputlayer,  #gcn
                 nbaselayer=args.nbaseblocklayer,#1
                 activation=F.relu,
                 withbn=args.withbn,            #False
                 withloop=args.withloop,        #FalseF
                 aggrmethod=args.aggrmethod,    #concat
                 mixmode=args.mixmode)

model2.load_state_dict(torch.load(PATH))
model2.to('cuda')

model2.reset()
for iteration in range(10):
    model2.eval()
    output = model2(test_fea, test_adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('acc in iteration %d = %f'%(iteration+1,acc_test))
    _, predicted = torch.max(output.data, 1)  #2708
    gradient = torch.zeros(*output.size()).to('cuda')
    if args.use_known_lable:
        for i in idx_train:
            gradient[i,labels[i]] = 1
        for i in idx_val:
            gradient[i,labels[i]] = 1
        for i in idx_test:
            gradient[i,predicted[i]] = 1
    else:
        for i,v in enumerate(predicted):
            gradient[i,v]=1
    model2.backward(gradient)
