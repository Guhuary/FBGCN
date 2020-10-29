import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.autograd import Variable
from torch.nn.parameter import Parameter

device = torch.device("cuda:0")


class GCNModel_fsp(nn.Module):
    def __init__(self,
                 nfeat,
                 nhid,                  #128
                 nclass,
                 nhidlayer,             #1
                 dropout,               #0.5
                 layer_sizes,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,          #1
                 activation=lambda x: x,
                 withbn=True,           #F
                 withloop=True,         #F
                 aggrmethod="add",      #cat
                 mixmode=False):        #T
        super(GCNModel_fsp, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout

        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        elif baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x 
        else:
            self.outgc = Dense(nhid, nclass, activation)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod
                                 )
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)
        #****************************************
        self.layers = self.make_layers()
        self.layer_sizes = layer_sizes
        self.z = {}
        for i, layer in list(enumerate(self.layers)):
            self.z[i] = torch.ones(self.layer_sizes[i]).to(device)

    def make_layers(self):
        layers = []
        layers += [self.ingc]
        layers += self.midlayer
        layers += [self.outgc]
        return layers

    def reset_parameters(self):
        pass

    def reset(self):
        for i in self.z.keys():
            self.z[i] = torch.ones(self.layer_sizes[i]).to(device)
    
    def forward(self, x,adj):
        self.input = []
        self.output = []
        for i, layer in list(enumerate(self.layers)):
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            if i in self.z:
                x = x * self.z[i]
            x = layer(x, adj)
            self.output.append(x)

        x = F.log_softmax(x, dim=1)
        return x
    
    def backward(self,g):
        for i, output in reversed(list(enumerate(self.output))):
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
            else:
                output.backward(self.input[i+1].grad.data)
            if i in self.z:
                alpha = self.input[i].grad
                self.z[i] = (alpha > 0).float()
                self.input[i].grad = self.z[i] * alpha
    