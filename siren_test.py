import shutil
import os
import loss_functions, modules, training, utils
from torch.utils.data import DataLoader

from functools import partial
import yaml
import torch
import torch.nn as nn
import numpy as np

from photon_library import PhotonLibrary

device = list(range(torch.cuda.device_count()))

class PlibSiren():
    """
    Train Siren on photon library
    """
    def __init__(self, cfg_file):
        cfg = yaml.load(open(cfg_file), Loader=yaml.Loader)
        self.output_dir = os.path.join(cfg['FilePath']['output_dir'], cfg['FilePath']['experiment_name'])
        self.plib_file = cfg['FilePath']['plib_file']
        self.pmt_file = cfg['FilePath']['pmt_file']
        self.lut_file = cfg['FilePath']['lut_file']

        self.in_features = cfg['Model']['in_features']
        self.out_features = cfg['Model']['out_features']
        self.hidden_features = cfg['Model']['hidden_features']
        self.hidden_layers = cfg['Model']['hidden_layers']
        self.omega = cfg['Model']['omega']

        self.batch_size = cfg['Training']['batch_size']
        self.num_epochs = cfg['Training']['num_epochs']
        self.lr = cfg['Training']['learning_rate']
        self.steps_til_summary = cfg['Training']['steps_til_summary']
        self.epochs_til_ckpt = cfg['Training']['epochs_til_checkpoint']

    def train(self):
        # Load plib dataset
        print('Load data ...')
        plib = PhotonLibrary(self.plib_file, self.pmt_file, self.lut_file)
        data = plib.LoadData()
        
        print('Assigning Model...')
        model = modules.Siren(in_features=self.in_features, out_features=self.out_features, \
            hidden_features=self.hidden_features, hidden_layers=self.hidden_layers, outermost_linear=True, omega=self.omega)
        model = model.float()
        model = nn.DataParallel(model, device_ids=device)
        model.cuda()

        print('Prepare dataloader')
        train_data = utils.DataWrapper(plib, data)
        dataloader = DataLoader(train_data, shuffle=True, batch_size=self.batch_size, pin_memory=False, num_workers=4)
        
        loss = partial(loss_functions.image_mse)

        print('Training...')
        training.train(model=model, train_dataloader=dataloader, epochs=self.num_epochs, lr=self.lr,
                steps_til_summary=self.steps_til_summary, epochs_til_checkpoint=self.epochs_til_ckpt,
                model_dir=self.output_dir, loss_fn=loss)

if __name__=="__main__":
    cfg_file = 'siren_cfg.yml'
    plib_siren = PlibSiren(cfg_file)

    if not os.path.exists(plib_siren.output_dir):
        os.makedirs(plib_siren.output_dir)
    shutil.copy('siren_cfg.yml', plib_siren.output_dir)

    plib_siren.train()