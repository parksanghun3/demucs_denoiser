#### Training ####

import torch
import logging
import pdb
import os
import numpy as np
import hydra
import time
import sys
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.io import wavfile
import scipy.signal as sps
from demucs import Demucs
from utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from stft_loss import MultiResolutionSTFTLoss
from dataset import NR_Dataset 
from augment import Remix, BandMask, Shift, RevEcho
import matplotlib.pyplot as plt
import csv 
from pesq import pesq
from pystoi import stoi

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args, model, optimizer, scheduler, train_dataloader, valid_dataloader, len_train_dataset, len_valid_dataset, best_val_loss, current_epoch, model_resume):
        self.args = args
        self.model = model
        self.current_epoch = current_epoch
        self.model_resume = model_resume
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.best_val_loss = best_val_loss
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=self.args.stft_sc_factor,factor_mag=self.args.stft_mag_factor).to(self.args.device)
        self.train_num_print = self.args.train_num_print
        self.valid_num_print = self.args.valid_num_print

        # data augments
        augments = []
        if args.remix:
            augments.append(Remix())
        if args.bandmask:
            augments.append(BandMask(args.bandmask, sample_rate=args.sample_rate))
        if args.shift:
            augments.append(Shift(args.shift, args.shift_same))
        if args.revecho:
            augments.append(RevEcho(args.revecho))
        
        self.augment = torch.nn.Sequential(*augments)
    
    def training(self):
        if self.model_resume:
            data = open('listfile.csv')
            load_data = csv.reader(data)
            loss_data = list(load_data)
            train_loss_list = list(map(float, loss_data[0]))
            valid_loss_list = list(map(float, loss_data[1]))
            self.start_epoch = self.current_epoch
            epochs = list(range(1, self.start_epoch+1))
            update_valid_loss = self.best_val_loss
        else:
            train_loss_list = []
            valid_loss_list = []
            epochs = []
            self.start_epoch = 0
            update_valid_loss = np.inf

        if self.start_epoch >= self.args.epoch:
            print("Please increase the args.epoch")
            sys.exit()

        for i in range(self.start_epoch, self.args.epoch):
            start = time.time()
            loss_total = 0.0
            logger.info('-'*70)
            logger.info("Training...")
            self.model.train()

            # Train
            name = "Train" + f" | Epoch {i+1}"
            train_logprog = LogProgress(logger,self.train_dataloader, updates=self.train_num_print, name=name)
            self.data_limit = self.args.sample_rate * self.args.data_len_limit
            for batch_idx, samples in enumerate(train_logprog):
                clean = samples[0].to(self.args.device, dtype=torch.float32).unsqueeze(1).permute(2,1,0)
                noisy = samples[1].to(self.args.device, dtype=torch.float32).unsqueeze(1).permute(2,1,0)
                clean = clean[:,:,:self.data_limit] # batch_size x 1 x ~self.data_limit
                noisy = noisy[:,:,:self.data_limit]

                # Augmentation
                sources = torch.stack([noisy - clean, clean]) # 2 x batch_size x 1 x time_length
                sources = self.augment(sources)
                noise, clean = sources
                noisy = noise + clean

                enhanced = self.model(noisy)
                if self.args.loss == "l1":
                    loss = F.l1_loss(clean,enhanced)

                # MultiResolution STFT loss
                if self.args.stft_loss:
                    sc_loss, mag_loss = self.mrstftloss(enhanced.squeeze(1), clean.squeeze(1))
                    loss += sc_loss + mag_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_total += loss.item()
                train_logprog.update(loss=format(loss.item(), ".5f"))
                del loss, enhanced

            train_loss = loss_total / (batch_idx + 1)
            train_loss_list.append(train_loss)
            
            logger.info(
                bold(f'Train Summary | End of Epoch {i + 1} | '
                    f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))
            state_dict = {
                "train_loss" : train_loss,
                "epoch" : i+1,
                "optimizer" : self.optimizer.state_dict(),
                "scheduler" : self.scheduler.state_dict(),
                "model" : self.model.state_dict()
            }

            # Validation
            name = "Valid" + f" | Epoch {i+1}"
            valid_logprog = LogProgress(logger,self.valid_dataloader, updates=self.valid_num_print, name=name)
            with torch.no_grad():
                loss_total = 0.0
                for batch_idx, samples in enumerate(valid_logprog):
                    clean = samples[0].to(self.args.device, dtype=torch.float32).unsqueeze(0)
                    noisy = samples[1].to(self.args.device, dtype=torch.float32).unsqueeze(0)
                    clean = clean[:,:,:self.data_limit] 
                    noisy = noisy[:,:,:self.data_limit]

                    enhanced = self.model(noisy)
                    if self.args.loss == "l1":
                        loss = F.l1_loss(clean,enhanced)
                            
                    # MultiResolution STFT loss
                    if self.args.stft_loss:
                        sc_loss, mag_loss = self.mrstftloss(enhanced.squeeze(1), clean.squeeze(1))
                        loss += sc_loss + mag_loss
                        
                    loss_total += loss.item()
                    valid_logprog.update(loss=format(loss.item(), ".5f"))
                    del loss, enhanced

                valid_loss = loss_total / (batch_idx + 1)
                valid_loss_list.append(valid_loss)

                self.scheduler.step(valid_loss)    
                logger.info(
                    bold(f'Valid Summary | End of Epoch {i + 1} | '
                        f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
            
            epochs.append(i+1)
            torch.save(state_dict, "./latest_model.tar")
            if valid_loss < update_valid_loss: 
                update_valid_loss = valid_loss
                logger.info(f"Found best loss in {i+1} epoch, saving...")
                torch.save(state_dict, "./best_model.tar")
            
            # visualize
            with open('listfile.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(train_loss_list)
                writer.writerow(valid_loss_list)
            plt.plot(epochs, train_loss_list, 'b')
            plt.plot(epochs, valid_loss_list, 'r')
            plt.legend(['train_loss','valid_loss'])
            plt.xlabel('epochs')
            plt.ylabel('Loss')
            plt.ylim([0,1])
            plt.grid(True)
            plt.show()
            plt.savefig('./result_loss.png')
