import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter

device = torch.device("cuda:0")

class GCNModel(nn.Module):
    def __init__(self,
                 nfeat,
                 nhid,                  
                 nclass,
                 nhidlayer,          
                 dropout,               
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,         
                 activation=lambda x: x,
                 withbn=True,         
                 withloop=True,        
                 aggrmethod="add",    
                 mixmode=False):      
        super(GCNModel, self).__init__()
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
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()
        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

        self.layers = self.make_layers()

    def make_layers(self):
        layers = []
        layers += [self.ingc]
        #layers += [nn.Dropout()]
        layers += self.midlayer
        layers += [self.outgc]
        print('The number of layers is', len(layers))
        return layers

    def reset_parameters(self):
        pass

    def forward(self, x, adj):
        layer_sizes = []
        for i, layer in list(enumerate(self.layers)):
            layer_sizes.append(x.size())
            x = layer(x,adj)
            if i == 0:
                x = F.dropout(x, self.dropout, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x, layer_sizes