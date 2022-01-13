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
import random
from demucs import Demucs
from utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
import torchaudio

logger = logging.getLogger(__name__)

def write(wav, filename, sr=16_000, encoding="PCM_S", bits_per_sample=16):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr, encoding="PCM_S", bits_per_sample=16)

@hydra.main(config_path="./config.yaml")
def enhance(args):
    args.device ='cuda:0'
    with torch.no_grad():
        base_path = "/home/work/sanghun"
        exp_name = base_path + "/NR/outputs/" + "exp_220110_tvcomm_qut_demand"
        model_path = exp_name + "/best_model.tar"
        # noisy_path = base_path + "/denoiser/dataset/test_dataset/KT_GG2_test_dataset/GG2_MR12차_GG3대비_성능검증_ASR_Starbucks_50dB_PC5_Speech_50dB_SAID_TT210331251_20210518_SVC35_68.6%/noisy/오디오 트랙_norm.wav"
        # noisy_path = base_path + "/denoiser/dataset/test_dataset/KT_GG2_test_dataset/GG2_MR12차_GG3대비_성능검증_ASR_Home_60dB_PC22_Speech_60dB_SAID_TT210331251_20210518_SVC35_65.2%/noisy/오디오 트랙_norm.wav"
        noisy_path = base_path + "/NR/test/GG2_MR12차_GG3대비_성능검증_ASR_Starbucks_50dB_PC5_Speech_60dB_SAID_TT210331251_20210512_SVC35_91.7%/오디오 트랙_norm.wav"
        save_path = base_path + "/NR/test/GG2_MR12차_GG3대비_성능검증_ASR_Starbucks_50dB_PC5_Speech_60dB_SAID_TT210331251_20210512_SVC35_91.7%/"
        noisy, sr = torchaudio.load(noisy_path)
        # write(noisy, exp_name + "/noisy_65.2%.wav", sr=sr, encoding="PCM_S", bits_per_sample=16)
        noisy = noisy.to(args.device).unsqueeze(0)
        model = Demucs(**args.demucs).to(args.device)
        model_dict = torch.load(model_path, 'cpu')
        # epoch = model_dict["epoch"]
        # best_loss = model_dict["best_loss"]
        # optimizer = model_dict["optimizer"]
        model.load_state_dict(model_dict["model"])
        model.eval()
        pdb.set_trace()
        enhance = model(noisy).squeeze(0)

        #Save as 16-bit signed integer Linear PCM
        # write(enhance, exp_name + "/" + os.path.basename(exp_name) + "_enhance_65.2%.wav" , sr=sr, encoding="PCM_S", bits_per_sample=16)
        write(enhance, save_path + "enhance_" + os.path.basename(exp_name) + ".wav" , sr=sr, encoding="PCM_S", bits_per_sample=16)

# def enhance(args):
#     args.device ='cuda:2'
#     with torch.no_grad():
#         base_path = "/home/work/sanghun"
#         exp_name = base_path + "/NR/outputs/" + "exp_220110_tvcomm_qut_demand"
#         model_path = exp_name + "/best_model.tar"
#         model = Demucs(**args.demucs).to(args.device)
#         model_dict = torch.load(model_path, 'cpu')
#         model.load_state_dict(model_dict["model"])
#         model.eval()
#         file_list = os.listdir(base_path + '/NR/test/noisy/')
#         for file_name in file_list:
#             noisy_path = base_path + '/NR/test/noisy/' 
#             noisy, sr = torchaudio.load(noisy_path + file_name)
#             noisy = noisy.to(args.device).unsqueeze(0)
#             enhance = model(noisy).squeeze(0)
#             print(file_name)
#             save_path = base_path + '/NR/test/enhance_220110_tvcomm_qut_demand/enhance_' + file_name
#             #Save as 16-bit signed integer Linear PCM
#             write(enhance, save_path , sr=sr, encoding="PCM_S", bits_per_sample=16)



if __name__ == '__main__':
    enhance()


