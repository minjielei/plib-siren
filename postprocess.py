import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import Siren
from photon_library import PhotonLibrary

device = list(range(torch.cuda.device_count()))

class PostProcess():
    def __init__(self, cfg_file):
        cfg = yaml.load(open(cfg_file), Loader=yaml.Loader)
        self.out_dir = os.path.join(cfg['FilePath']['model_dir'], cfg['FilePath']['experiment_name'])
        self.siren_file = os.path.join(self.out_dir, cfg['FilePath']['siren_file'])
        self.plib_file = cfg['FilePath']['plib_file']
        self.pmt_file = cfg['FilePath']['pmt_file']

        self.in_features = cfg['Model']['in_features']
        self.out_features = cfg['Model']['out_features']
        self.hidden_features = cfg['Model']['hidden_features']
        self.hidden_layers = cfg['Model']['hidden_layers']
        self.omega = cfg['Model']['omega']

        self.pred_file = os.path.join(self.out_dir, cfg['Process']['pred_file'])
        self.grad_file = os.path.join(self.out_dir, cfg['Process']['grad_file'])
        self.plt_file = os.path.join(self.out_dir, cfg['Process']['plt_file'])
        self.plt_data_file = os.path.join(self.out_dir, cfg['Process']['plt_data_file'])
        self.vis_cutoff = cfg['Process']['vis_cutoff']

        self.load_plib()
        self.load_model()

    def load_plib(self):
        self.plib = PhotonLibrary(self.plib_file, self.pmt_file)

    def load_model(self):
        self.model = Siren(self.in_features, self.hidden_features, self.hidden_layers, self.out_features, True, self.omega)
        self.model = self.model.float()
        self.model = torch.nn.DataParallel(self.model, device_ids=device)
        self.model.cuda()
        self.model.load_state_dict(torch.load(self.siren_file)['model_state_dict'])
        self.model.eval()

    def make_siren_pred(self, n = 22):
        coord = self.plib.LoadCoord()
        coord = torch.from_numpy(coord).cuda()

        pred_arr = []
        grad_arr = []
        start = 0
        end = coord.shape[0]//n
        for i in range(n):
            input = coord[start:end, :]
            model_output = self.model(input)
            pred = self.plib.DataTransformInv(model_output['model_out'])
            grad_outputs = torch.ones_like(pred)
            grad = torch.autograd.grad(pred, model_output['model_in'], grad_outputs=grad_outputs)[0]
            pred_arr.append(np.clip(pred.cpu().detach().numpy(), 0.0, 1.0))
            grad_arr.append(grad.cpu().detach().numpy())
            start += coord.shape[0]//n
            end += coord.shape[0]//n
        pred_clipped = np.vstack(pred_arr)
        grad_clipped = np.vstack(grad_arr)
        np.save(self.pred_file, pred_clipped)
        np.save(self.grad_file, pred_clipped)
        self.make_rel_bias_plt(pred_clipped)

    def make_rel_bias_plt(self, pred, eps=1e-10):
        gt = self.plib.LoadData(transform=False)
        re = 2 * abs((pred - gt) / (pred + gt + eps))
        mre = np.nanmean(np.where(gt > self.vis_cutoff, re, np.nan), axis = 0) * 100
        std = np.nanstd(np.where(gt > self.vis_cutoff, re, np.nan), axis = 0) * 100
        np.save(self.plt_data_file, mre)
        
        npmt=180
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor='w')
        ax[0].plot(np.arange(npmt), mre)
        ax[0].set_xlim(0, npmt)
        # ax[0].set_ylim(0, 10)
        ax[0].set_xlabel("PMT ID", fontsize=14)
        ax[0].set_ylabel("Mean Bias (%)", fontsize=14)
        ax[1].plot(np.arange(npmt), std)
        ax[1].set_xlabel("PMT ID", fontsize=14)
        ax[1].set_ylabel("STD (%)", fontsize=14)
        ax[1].set_xlim(0, npmt)
        plt.savefig(self.plt_file, dpi=300, bbox_inches='tight')
        plt.clf()
        
if __name__=="__main__":
    cfg_file = 'process_cfg.yml'
    process = PostProcess(cfg_file)

    process.make_siren_pred()