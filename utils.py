import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DataWrapper(DataLoader):
    def __init__(self, plib, extend=False):
        self.plib = plib
        self.coords = plib.LoadCoord(extend=extend)
        self.gt = plib.LoadData(extend=extend)
        
    def __len__(self):
        return self.gt.shape[0]
    
    def __getitem__(self, idx):
        weights = self.plib.WeightFromIdx(idx)
        in_dict = {'idx': idx, 'coords': self.coords[idx], 'weights': weights}
        gt_dict = {'idx': idx, 'coords': self.gt[idx]}

        return in_dict, gt_dict

def make_weights(in_tensor, num_bins):
    """
    A function that returns the appropriate weight vector to statistically
    weight imbalanced data.
    """

    weight = torch.histc(in_tensor, bins=num_bins)
    weight = weight.max()/weight
    
    return torch.nan_to_num(weight, 1.0, 1.0, 1.0)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """
    This function will load the most current training checkpoint.
    """
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        total_steps = checkpoint['step']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losslogger = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {}, steps {})"
                  .format(filename, checkpoint['epoch'], checkpoint['step']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, total_steps, start_epoch, losslogger

def find_latest_checkpoint(model_dir):
    """
    This helper function finds the checkpoint with the largest
    epoch value given a model directory.
    """
    tmp_dir = os.path.join(model_dir, 'checkpoints')
    list_of_files = glob.glob(tmp_dir+'/model_epoch_*.pth') 
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def plot_losses(total_steps, train_losses, filename):
    x_steps = np.linspace(0, total_steps, num=len(train_losses))
    plt.figure(tight_layout=True)
    plt.plot(x_steps, train_losses)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()

def plot_pred_vs_gt(pred, gt, filename):
    plt.figure(tight_layout=True)
    plt.scatter(gt*16, pred*16)
    plt.grid()        
    plt.xlabel('Truth')
    plt.ylabel('Pred')

    min_scale = min(plt.gca().get_xlim()[0], plt.gca().get_ylim()[0])
    max_scale = max(plt.gca().get_xlim()[1], plt.gca().get_ylim()[1])
    plt.xlim(min_scale, max_scale)
    plt.ylim(min_scale, max_scale)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()

def plot_hist_overlap(pred, gt, filename):
    calc = (gt - pred)/(2*(gt + pred))
    plt.hist(calc.flatten(), alpha=0.5)
    plt.xlabel('asymmetry')
    plt.ylabel('samples')
    plt.yscale('log')
    plt.savefig(filename)
    plt.clf()