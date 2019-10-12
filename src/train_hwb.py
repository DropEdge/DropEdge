from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from earlystopping import EarlyStopping
from models import GCN
from sample import Sampler
from metric import accuracy

#from pygcn.utils import load_data, accuracy
#from pygcn.models import GCN

from metric import accuracy
from utils import load_citation, load_reddit_data
from models import *
from earlystopping import EarlyStopping

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--lradjust',action='store_true', default=False, help = 'Enable leraning rate adjust.(ReduceLROnPlateau)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--withbn', action='store_true', default=False, help='Enable Bath Norm GCN')
parser.add_argument('--nhiddenlayer', type=int, default=0, help='The number of hidden layers.(may outdated)')
parser.add_argument('--debug', action='store_true', default=False, help="Enable the detialed training output.")
parser.add_argument('--dataset', default="cora", help="The data set")
parser.add_argument("--earlystopping", type=int, default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument("--normalization", default="AugNormAdj", help="The normalization on the adj matrix.")
parser.add_argument("--debug_samplingpercent", type=float, default=1.0,help="The percent of the preserve edges (debug only)")
parser.add_argument("--gpu", type=int, default=0,help="The gpu to be applied")
parser.add_argument("--mixmode", action="store_true", default=False, help="Enable CPU GPU mixing mode.")

args = parser.parse_args()

if args.debug:
    print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()

#no fix seed here
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#if args.dataset=='cora':
#    adj, features, labels, idx_train, idx_val, idx_test = load_data()
#else:

# train_adj = torch.Tensor([1]) #only for reddit
# train_features = torch.Tensor([1]) #only for reddit
# if args.dataset == 'reddit':
#    adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test = load_reddit_data()
# elif args.dataset == 'pubmed':
#    adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, "BingGeNormAdj")
# else:
#    adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, "BingGeNormAdj")

sampler = Sampler(args.dataset)

# get labels and indexes
labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes()
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

# Model and optimizer
model = GCNBS(nfeat=nfeat,
            nhid=args.hidden,
            nclass=nclass,
            withbn=args.withbn,
            nhiddenlayer=args.nhiddenlayer,
            dropout=args.dropout,
            mixmode=args.mixmode)

# model = GCNFlatRes(nfeat=nfeat,
#            nhid=args.hidden,
#            nclass=nclass,
#            withbn=args.withbn,
#            nreslayer=args.nhiddenlayer,
#            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,  weight_decay=args.weight_decay)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.5)

# convert to cuda
if args.cuda:
    model.cuda()
    # features = features.cuda()
    # adj = adj.cuda()
    # train_adj = train_adj.cuda() #only for reddit
    # train_features = train_features.cuda() #only for reddit
    #Maybe not the best practice here.
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


if args.cuda or args.mixmode:
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# set early_stopping
if args.earlystopping > 0:
    early_stopping = EarlyStopping(patience=args.earlystopping, verbose=False)


# define the training function.

def train(epoch, train_adj, train_fea, val_adj=None, val_fea=None):
    if val_adj is None:
        val_adj = train_adj
        val_fea = train_fea
    t = time.time()

    #adjust lr
    if args.lradjust:
        #scheduler.step(loss_val)
        scheduler.step()

    model.train()
    optimizer.zero_grad()
    output = model(train_fea, train_adj)
    #special for reddit
    if sampler.learning_type == "inductive":
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
    
    loss_train.backward()
    optimizer.step()

    #We can not apply the fastmode for the reddit dataset.
    if sampler.learning_type == "inductive" or not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(val_fea, val_adj)
     
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    if args.earlystopping > 0:
        early_stopping(loss_val, model)
    
    if args.debug and epoch % 1 == 0:
        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()),
            'time: {:.4f}s'.format(time.time() - t))
    return (loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item())


def test(test_adj,test_fea):
    model.eval()
    output = model(test_fea, test_adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    if args.debug:
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
    return (loss_test.item(), acc_test.item())

# Visualize 
#params = list(model.named_parameters())
#print(params[0])
#exit()

# Train model
t_total = time.time()
loss_train = np.zeros((args.epochs, ))
acc_train = np.zeros((args.epochs, ))
loss_val = np.zeros((args.epochs, ))
acc_val = np.zeros((args.epochs, ))

for epoch in range(args.epochs):
    #(train_adj, train_fea) = sampler.stub_sampler(normalization=args.normalization, cuda=args.cuda)
    (train_adj, train_fea) = sampler.randomedge_sampler(percent=args.debug_samplingpercent, normalization=args.normalization, cuda=args.cuda)
    if sampler.learning_type == "transductive":
        outputs = train(epoch, train_adj, train_fea)
    else:
        (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
        outputs = train(epoch, train_adj, train_fea, val_adj, val_fea)
    loss_train[epoch], acc_train[epoch], loss_val[epoch], acc_val[epoch] = outputs[0], outputs[1], outputs[2], outputs[3]

    if args.earlystopping > 0 and early_stopping.early_stope:
        print("Early stopping.")
        model.load_state_dict(early_stopping.load_checkpoint())
        break

np.savetxt('./results_'+args.dataset+'_'+str(args.nhiddenlayer), np.vstack((loss_train, loss_val, acc_train, acc_val)), delimiter='\t')

if args.debug:
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
(train_adj, train_fea) = sampler.stub_sampler(normalization=args.normalization, cuda=args.cuda)
(loss_test, acc_test) = test(test_adj, test_fea)
print("%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f"%(loss_train[-1], loss_val[-1], loss_test, acc_train[-1],acc_val[-1],acc_test))


