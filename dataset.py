### Dataset def ###
import torch
import logging
import pdb
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.io import wavfile
import pandas as pd
import argparse
import random
import scipy.signal as sps
from sklearn.model_selection import train_test_split
from demucs import Demucs
from utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
import warnings
warnings.filterwarnings("error")

logger = logging.getLogger(__name__)
random.seed(1)
torch.manual_seed(1)

class NR_Dataset(Dataset):

    def __init__(self, args, mode='train', SNR_list=[0,5,10,15]): # main에서 dataset 정의할때 들어옴.
        f = pd.read_csv('/home/work/data/DB/data.tmp/K_tvcomm/tvcomm_0609_sanghun.list', encoding='cp949',sep='\t')
        self.args = args

        ## tvcomm df 
        f['text_len'] = f['text'].str.len()
        f1 = f.loc[f.index[:-1885]] # remove files in /201605 folder
        f2 = f1.loc[~f1['wav_name'].str.contains('2ch_01_2m')].reset_index(drop=True) # remove "2ch_01_2m"
        limit = f2.text_len >= 4
        df_tmp = f2.loc[limit,:].reset_index(drop=True) # remove files for utter_length <= 3
        train_tvcomm_df = df_tmp.loc[~df_tmp['wav_name'].str.contains('2ch_01_0m')].reset_index(drop=True)
        test_tvcomm_df = df_tmp.loc[df_tmp['wav_name'].str.contains('2ch_01_0m')].reset_index(drop=True)

        ## AIhub df
        train_base_path = '/home/work/data/DB/ai_hub/Training/'
        test_base_path = '/home/work/data/DB/ai_hub/Validation/'
        train_file_list = make_path_list(train_base_path, 'clean')
        train_aihub_df = pd.DataFrame(train_file_list)
        train_aihub_df.columns=['wav_name']

        test_file_list = make_path_list(test_base_path, 'clean')
        test_aihub_df = pd.DataFrame(test_file_list)
        test_aihub_df.columns = ['wav_name']
        test_aihub_df = test_aihub_df.sample(n=6000, random_state=1).reset_index(drop=True)

        #final df
        # self.train_df_all = pd.concat([train_tvcomm_df, train_aihub_df]).reset_index(drop=True)
        self.train_df_all = train_tvcomm_df ## train data tvcomm only
        self.train_df, self.valid_df = train_test_split(self.train_df_all, test_size=0.01, random_state=1) ## 전체 420,685개 중 train(378,616) / valid(42,069) 로 분리
        self.train_df= self.train_df.reset_index(drop=True)
        self.valid_df= self.valid_df.reset_index(drop=True)
        self.test_df = pd.concat([test_tvcomm_df, test_aihub_df]).reset_index(drop=True)
        test_num_data = len(self.test_df)

        ## Noise
        dns_path = '/home/work/data/DB/dns/DNS-Challenge/datasets/noise/'
        dns_file_list = make_path_list(dns_path, 'dns')
        self.train_dns_df = pd.DataFrame(dns_file_list)
        
        demand_path = '/home/work/data/DB/noise/DEMAND/demand_16k/'
        demand_file_list = make_path_list(demand_path, 'demand')
        self.demand_df = pd.DataFrame(demand_file_list)

        qut_demand_path = '/home/work/sanghun/NR/noise_data/'
        qut_demand_file_list = make_path_list(qut_demand_path, 'qut_demand')
        self.qut_demand_df = pd.DataFrame(qut_demand_file_list)
        self.SNR_list = SNR_list
        self.mode = mode

    def __len__(self): # for문으로 dataloader 부를 때 들어옴.
        if self.mode == 'train':
            if self.args.train_dataset_len == "full" :
                dataset_len = len(self.train_df)
            else:
                dataset_len = self.args.train_dataset_len
        else:
            if self.args.valid_dataset_len == "full" :
                dataset_len = len(self.valid_df)
            else:
                dataset_len = self.args.valid_dataset_len
        return dataset_len


    def __getitem__(self, index):

        if self.mode == 'train':
            fs, clean, clean_file = clean_choice(self.train_df)
            # noise_df = self.train_dns_df
            # noise_df = self.demand_df
            noise_df = self.qut_demand_df
        elif self.mode == 'valid':
            fs, clean, clean_file = clean_choice(self.valid_df)
            # noise_df = self.train_dns_df
            # noise_df = self.demand_df
            noise_df = self.qut_demand_df
        elif self.mode == 'test':
            fs, clean, clean_file = clean_choice(self.test_df)
            noise_df = self.demand_df

        # Down sampling to 16k
        if fs == 48000:
            number_of_samples = round(len(clean) * float(16000) / fs)
            clean = sps.resample(clean, number_of_samples)

        clean_rms = cal_rms(clean)
        noise, divided_noise = noise_choice(noise_df, clean)

        while sum(divided_noise) == 0: ## noise data가 전부 0이 나오지 않을 때까지 새로 선택함.
            noise, divided_noise = noise_choice(noise_df, clean)

        noise_rms = cal_rms(divided_noise)

        ### Noisy mixing with SNR
        SNR = random.choice(self.SNR_list)
        adj_noise_rms = cal_adjusted_noise_rms(clean_rms, SNR)
        # print(noise_rms)
        adj_noise = divided_noise * (adj_noise_rms / noise_rms)
        
        # adj_noise2_rms = cal_adjusted_noise_rms(clean_rms, SNR+20)
        # adj_noise2 = divided_noise * (adj_noise2_rms / noise_rms)

        noisy = clean + adj_noise
        # clean = clean + adj_noise2

        # Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if noisy.max(axis=0) > max_int16 or noisy.min(axis=0) < min_int16:
            if noisy.max(axis=0) >= abs(noisy.min(axis=0)):
                    reduction_rate = max_int16 / noisy.max(axis=0)
            else :
                    reduction_rate = min_int16 / noisy.min(axis=0)
            noisy = noisy * (reduction_rate)
            clean = clean * (reduction_rate)

        # ### Normalize
        try:
            clean_norm = clean / np.max(np.abs(clean), axis=0)
            noisy_norm = noisy / np.max(np.abs(noisy), axis=0)
        except RuntimeWarning:
            print(clean)
            print(np.max(np.abs(clean), axis=0))
            import ipdb; ipdb.set_trace()

        ### file checking
        # file_check_path = '/home/work/sanghun/NR/data_checking/'
        # noisy_file_save = file_check_path + str(index) + '_noisy_' + clean_path.replace('/','_')[:-4]+'_'+os.path.basename(noise_file)[:-4] + '_' + str(SNR) + 'dB.wav'
        # clean_file_save = file_check_path + str(index) + '_clean_' + clean_path.replace('/','_')[:-4] + '.wav'
        # print(noisy_file_save, clean_file_save)
        # wavfile.write(noisy_file_save, 16000, noisy.astype(np.int16))
        # wavfile.write(clean_file_save, 16000, clean.astype(np.int16))
        # print(index, len(clean))

        return clean_norm, noisy_norm, clean_file


def make_path_list(base_path, d_type):

    if d_type == 'clean':
        data_list = []
        dir_list = os.listdir(base_path)
        for dir_name in dir_list:
            dir_path = base_path + dir_name
            file_list = os.listdir(dir_path)
            file_path_list = []
            for file_name in file_list:
                file_path_list.append(dir_path + '/' + file_name)
            data_list += file_path_list
    else:
        file_list = os.listdir(base_path)
        data_list = []
        for file_name in file_list:
            if d_type == 'dns' or d_type == 'qut_demand':
                file_path = base_path + file_name
            elif d_type == 'demand':
                file_path = base_path + file_name +'/ch01.wav'
            data_list.append(file_path)

    return data_list


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp.astype(np.float64)), axis=-1))


def cal_adjusted_noise_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms


def clean_choice(clean_df):
    clean_file = clean_df.loc[random.randrange(0,len(clean_df))][0] # train 경우, 378,616 중에 랜덤 한개씩 뽑아서 총 dataset_length 개 만큼 학습데이터에 사용
    if clean_file[0] == 'K': # K_tvcomm dataset 
        fs, clean = wavfile.read('/home/work/data/DB/data.tmp/' + clean_file)
    else: # ai_hub dataset
        fs, clean = wavfile.read(clean_file)
    # print("처음 " , len(clean))
    
    while len(clean) > 320000:# 20초 미만의 clean 선택
        clean_file = clean_df.loc[random.randrange(0,len(clean_df))][0] # train 경우, 378,616 중에 랜덤 한개씩 뽑아서 총 dataset_length 개 만큼 학습데이터에 사용
        if clean_file[0] == 'K': # K_tvcomm dataset 
            fs, clean = wavfile.read('/home/work/data/DB/data.tmp/' + clean_file)
        else: # ai_hub dataset
            fs, clean = wavfile.read(clean_file)
        # print("루프 " , len(clean))
    return fs, clean, clean_file


def noise_choice(noise_df, clean):
    noise_file = noise_df.loc[random.randrange(0,len(noise_df))][0]
    fs_, noise = wavfile.read(noise_file)
    # print("noise clean 비교 : ", len(noise), len(clean))
    
    while (len(noise) - len(clean)) <= 0: ## noise의 길이가 clean보다 클 때까지 새로 선택함.
        noise_file = noise_df.loc[random.randrange(0,len(noise_df))][0]
        fs_, noise = wavfile.read(noise_file)
        # print("clean이 커서 새로 선택중 : ", len(noise), len(clean))
 
    offset = random.randint(0, len(noise)-len(clean))
    divided_noise = noise[offset: offset + len(clean)]
    return noise, divided_noise