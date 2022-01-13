# Noise reduction main
import torch
import logging
import pdb
import os
import numpy as np
import hydra
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.io import wavfile
import argparse
import scipy.signal as sps
from demucs import Demucs
from utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from dataset import NR_Dataset
from train import Trainer

logger = logging.getLogger(__name__)

@hydra.main(config_path="./config.yaml")
def main(args):

    ### zero padding
    def collate_fn_pad(batch):
        noisy_list = []
        clean_list = []
        names = []

        for noisy, clean, name in batch:
            noisy_list.append(torch.tensor(noisy))
            clean_list.append(torch.tensor(clean))
            names.append(name)
            # print(f"noisy = {noisy}, len_noisy = {len(noisy)}, clean = {clean}, len_clean = {len(clean)}, name = {name}")

        noisy_list = pad_sequence(noisy_list)            
        clean_list = pad_sequence(clean_list)            
        
        return noisy_list, clean_list, names

    #### Dataset def
    SNR_list = [0, 5, 10, 15]
    train_dataset = NR_Dataset(args, mode='train', SNR_list = SNR_list)
    valid_dataset = NR_Dataset(args, mode='valid', SNR_list = SNR_list)
    # test_dataset = NR_Dataset(mode='test', SNR_list = SNR_list)

    #### Data loading
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn_pad, num_workers = args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers = args.num_workers)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    #### Model & optimizer
    model = Demucs(**args.demucs).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=1)

    if args.model_resume:
        model_path = '../' + args.model_resume_name 
        model_dict = torch.load(model_path, 'cpu')
        model.load_state_dict(model_dict["model"])
        optimizer.load_state_dict(model_dict['optimizer'])
        scheduler.load_state_dict(model_dict['scheduler'])
        best_val_loss = model_dict['scheduler']['best']
        current_epoch = model_dict['epoch']
        logger.info(f'Pretrained model... {model_path}')
        logger.info(f'last_epoch = {current_epoch} / best_val_loss = {best_val_loss} / last_lr = {scheduler._last_lr}')
    else:
        best_val_loss = np.inf
        current_epoch = 0

    ### Training
    logger.info(f"Train_dataset_length = {len(train_dataset)}")
    logger.info(f"Valid_dataset_length = {len(valid_dataset)}")
    trainer = Trainer(args, model, optimizer, scheduler, train_dataloader, valid_dataloader, len(train_dataset), len(valid_dataset), best_val_loss, current_epoch, args.model_resume)
    trainer.training()


if __name__ == '__main__':
    main()


