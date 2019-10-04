import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F



class GraphConvolutionBS(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=True, withloop=True, bias=True,
                 res=False):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # Is this the best practice or not?
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphBaseBlock(Module):
    """
    The base block for Multi-layer GCN / ResGCN / Dense GCN 
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=False):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(GraphBaseBlock, self).__init__()
        self.in_features = in_features
        self.hiddendim = out_features
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.hiddenlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat" and dense == False:
            self.out_features = in_features + out_features
        elif self.aggrmethod == "concat" and dense == True:
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in add model.")
            self.out_features = out_features
        elif self.aggrmethod == "nores":
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat','add' and 'nores'.")

    def __makehidden(self):
        # for i in xrange(self.nhiddenlayer):
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                           self.withloop)
            else:
                layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, adj):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for gc in self.hiddenlayers:
            denseout = self._doconcat(denseout, x)
            x = gc(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)
        return self._doconcat(x, denseout)

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nhiddenlayer,
                                              self.out_features)


class MultiLayerGCNBlock(Module):
    """
    Muti-Layer GCN with same hidden dimension.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod=None, dense=None):
        """
        The multiple layer GCN block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: not applied.
        :param dense: not applied.
        """
        super(MultiLayerGCNBlock, self).__init__()
        self.model = GraphBaseBlock(in_features=in_features,
                                    out_features=out_features,
                                    nbaselayer=nbaselayer,
                                    withbn=withbn,
                                    withloop=withloop,
                                    activation=activation,
                                    dropout=dropout,
                                    dense=False,
                                    aggrmethod="nores")

    def forward(self, input, adj):
        return self.model.forward(input, adj)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.model.in_features,
                                              self.model.hiddendim,
                                              self.model.nhiddenlayer,
                                              self.model.out_features)


class ResGCNBlock(Module):
    """
    The multiple layer GCN with residual connection block.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod=None, dense=None):
        """
        The multiple layer GCN with residual connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: not applied.
        :param dense: not applied.
        """
        super(ResGCNBlock, self).__init__()
        self.model = GraphBaseBlock(in_features=in_features,
                                    out_features=out_features,
                                    nbaselayer=nbaselayer,
                                    withbn=withbn,
                                    withloop=withloop,
                                    activation=activation,
                                    dropout=dropout,
                                    dense=False,
                                    aggrmethod="add")

    def forward(self, input, adj):
        return self.model.forward(input, adj)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.model.in_features,
                                              self.model.hiddendim,
                                              self.model.nhiddenlayer,
                                              self.model.out_features)


class DenseGCNBlock(Module):
    """
    The multiple layer GCN with dense connection block.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=True):
        """
        The multiple layer GCN with dense connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for the output. For denseblock, default is "concat".
        :param dense: default is True, cannot be changed.
        """
        super(DenseGCNBlock, self).__init__()
        self.model = GraphBaseBlock(in_features=in_features,
                                    out_features=out_features,
                                    nbaselayer=nbaselayer,
                                    withbn=withbn,
                                    withloop=withloop,
                                    activation=activation,
                                    dropout=dropout,
                                    dense=True,
                                    aggrmethod=aggrmethod)

    def forward(self, input, adj):
        return self.model.forward(input, adj)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.model.in_features,
                                              self.model.hiddendim,
                                              self.model.nhiddenlayer,
                                              self.model.out_features)


class InecptionGCNBlock(Module):
    """
    The multiple layer GCN with inception connection block.
    """

    def __init__(self, in_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=False):
        """
        The multiple layer GCN with inception connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: not applied. The default is False, cannot be changed.
        """
        super(InecptionGCNBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hiddendim = out_features
        self.nbaselayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.midlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat":
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in 'add' model.")
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat', 'add'.")

    def __makehidden(self):
        # for j in xrange(self.nhiddenlayer):
        for j in range(self.nbaselayer):
            reslayer = nn.ModuleList()
            # for i in xrange(j + 1):
            for i in range(j + 1):
                if i == 0:
                    layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                               self.withloop)
                else:
                    layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn,
                                               self.withloop)
                reslayer.append(layer)
            self.midlayers.append(reslayer)

    def forward(self, input, adj):
        x = input
        for reslayer in self.midlayers:
            subx = input
            for gc in reslayer:
                subx = gc(subx, adj)
                subx = F.dropout(subx, self.dropout, training=self.training)
            x = self._doconcat(x, subx)
        return x

    def get_outdim(self):
        return self.out_features

    def _doconcat(self, x, subx):
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nbaselayer,
                                              self.out_features)


class Dense(Module):
    """
    Simple Dense layer, Do not consider adj.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, bias=True, res=False):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.res = res
        self.bn = nn.BatchNorm1d(out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
