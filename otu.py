import librosa
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from utils import play_audio,maintain_size
import os
import sys

def prepare_data():
    data_pth='./data/finetune_data/validate/on/'

    if len(sys.argv) > 1:
        data_pth = sys.argv[1]

    wavs = librosa.util.find_files(data_pth)
    for i,wav in enumerate(wavs):
        print('Processing Record: ',str((i+1)/len(wavs)),wav,' ....')
        y,sr=librosa.load(wav,sr=None,)
        play_audio(y,sr)
        continue
        if sr != 16000:
            y = librosa.resample(y,sr,16000)
        if len(y) != 16000:
            y,ind = librosa.effects.trim(y,top_db=32)
            #print(y.shape)  
            y = maintain_size(y,16000,center_crop=True)      
        #print(y.shape)
        if wav[-4:] != '.wav':
            os.remove(wav)
            wav = wav[:-4] + '.wav'
        librosa.output.write_wav(wav, y, 16000)
def move_files():
    subDataFile = './data/speech_commands_v0.02/validation_list.txt'
    data = './data/benchmark/'
    dest = './data/benchmark/validation/'
                
    sublist = [x.strip() for x in open(subDataFile)]

    for wav in sublist:         
        outDir = dest+ (wav.split('/')[0])
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        os.rename(data+wav, dest+wav)            
def split_wavs():
    data_pth='./data/benchmark/_background_noise_/'

    wavs = librosa.util.find_files(data_pth)
    for i,wav in enumerate(wavs):
        print('Processing Record: ',str(round((i+1)/len(wavs),2)),wav,' ....')
        y,sr=librosa.load(wav,sr=16000,)
        splits = []
        filename = wav.replace('\\','/').split('/')[-1][:-4]
        clip_size = 16000
        count = 0
        while count < len(y)//clip_size:
            librosa.output.write_wav(filename+str(count+1)+'.wav', y[count*clip_size:(count+1)*clip_size], 16000)
            count+=1

        librosa.output.write_wav(filename+str(count+1)+'.wav', maintain_size(y[count*clip_size:],16000), 16000)
prepare_data()        