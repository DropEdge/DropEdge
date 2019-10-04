#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import random,string
import datetime
import os

# TODO: hard coding the model path here.
folder = "tmpmodel"
if not os.path.exists(folder):
    os.mkdir(folder)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, datasets="tmp", patience=7, fname=None, clean=False, verbose=False):
        """ 
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        timstr = datetime.datetime.now().strftime("%m%d-%H%M%S")
        if fname is None:
            fname = datasets + "-" + timstr + "-" + self._random_str() + ".pt"
        self.fname = os.path.join(folder, fname)
        self.clean = clean
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: %d out of %d"%(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def _random_str(self, randomlength=3):
        a = list(string.ascii_letters)
        random.shuffle(a)
        return ''.join(a[:randomlength])

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased (%.6f --> %.6f).  Saving model ...'%(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.fname)
        self.val_loss_min = val_loss

    def load_checkpoint(self):
        return  torch.load(self.fname)
