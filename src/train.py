# train the models and save one that performs best in val
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
parser.add_argument('--dataset', default="cora", help="The data set")
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
parser.add_argument("--normalization", default="AugNormAdj",
                    help="The normalization on the adj matrix.")
# parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
parser.add_argument("--nbaseblocklayer", type=int, default=1,
                    help="The number of layers in each baseblock, please set it to 1")
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

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
# convert to cuda
if args.cuda:
    model.cuda()

# For the mix mode, lables and indexes are in cuda. 
if args.cuda or args.mixmode:
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# define the training function.
def train(epoch, train_adj, train_fea, idx_train, val_adj=None, val_fea=None):
    if val_adj is None:
        val_adj = train_adj
        val_fea = train_fea

    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,_ = model(train_fea, train_adj)
    # special for reddit
    if sampler.learning_type == "inductive":
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    train_t = time.time() - t
    val_t = time.time()
    # We can not apply the fastmode for the reddit dataset.
    # if sampler.learning_type == "inductive" or not args.fastmode:

    if not args.fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        model.eval()
        output,_ = model(val_fea, val_adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
    else:
        loss_val = 0
        acc_val = 0

    if args.lradjust:
        scheduler.step()

    val_t = time.time() - val_t
    # test_t and val_t are not used
    return (loss_train.item(), acc_train.item(), loss_val, acc_val, train_t, val_t)


def test(test_adj, test_fea):
    model.eval()
    output,_ = model(test_fea, test_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])
    return (loss_test.item(), acc_test.item())


# Train model
t_total = time.time()
loss_train = np.zeros((args.epochs,))
acc_train = np.zeros((args.epochs,))
loss_val = np.zeros((args.epochs,))
acc_val = np.zeros((args.epochs,))

sampling_t = 0
#(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
(train_adj, train_fea) = sampler.stub_sampler(normalization=args.normalization,cuda=args.cuda)                                          
if args.mixmode:
    train_adj = train_adj.cuda()
    train_fea = train_fea.cuda()

for epoch in range(args.epochs):
    input_idx_train = idx_train
    sampling_t = time.time()
    # no sampling
    sampling_t = time.time() - sampling_t
    
    # The validation set is controlled by idx_val
    # if sampler.learning_type == "transductive":
    if False:
        outputs = train(epoch, train_adj, train_fea, input_idx_train)
    else:
        (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
        if args.mixmode:
            val_adj = val_adj.cuda()
        outputs = train(epoch, train_adj, train_fea, input_idx_train, val_adj, val_fea)


    loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch] = outputs[0], outputs[1], outputs[2], outputs[
        3]
    if epoch==0:
        acc_max = acc_val[0]
    if acc_val[epoch] > acc_max and epoch >=args.epochs - 100:
        torch.save(model.state_dict(), os.path.join(args.save_path, '{}_{}_l_{}.ckpt'.format(args.type, args.dataset, args.nhiddenlayer+2)))
        acc_max = acc_val[epoch]

    acc_current = acc_val[epoch]

# Testing
(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
if args.mixmode:
    test_adj = test_adj.cuda()
    test_fea = test_fea.cuda()
(loss_test, acc_test) = test(test_adj, test_fea)
print("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f" % (
loss_train[-1], loss_val[-1], loss_test, acc_train[-1], acc_val[-1], acc_test))

