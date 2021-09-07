# import librosa
# from librosa.display import waveplot,cmap
# import numpy as np
# from scipy.io import wavfile
# import sounddevice as sd
# import matplotlib.pyplot as plt
# from utils import plot_spectrum,calculate_mel_spect,plot_wave,play_audio,maintain_size
# from random import shuffle
# from classic_approach import train,test,test_zcr,preprocess_wav
# import os
# from augmentation import padd_zeros,play_speed,Augmentor
# from data import generateAnnotationList,load_wav


# # data_pth='./data/finetune_data/'
# # wavs = librosa.util.find_files(data_pth)
# # for i,wav in enumerate(wavs):
# #     #print('Processing Record: ',str((i+1)/len(wavs)),wav,' ....')
# #     y,sr=librosa.load(wav,sr=None,)
# #     #play_audio(y,sr)
# #     if len(y) != 16000:
# #         print('Processing Record: ',str((i+1)/len(wavs)),wav,' ....')
# #     continue
# #     if sr != 16000:
# #         y = librosa.resample(y,sr,16000)
# #     print(y.shape)
# #     if len(y) != 16000:
# #         y,ind = librosa.effects.trim(y,top_db=60)
# #         y = maintain_size(y,16000,center_crop=True)
# #     print(y.shape)
# #     librosa.output.write_wav(wav, y, 16000)

# wav = 'C:/Users/Void/Desktop/adk/p1.mp3'
# es = 16000
# wanted_sec = 17
# y,sr=librosa.load(wav,sr=16000,)
# print(y.shape,sr)
# ind =es*wanted_sec
# crop =  y[ind:ind+es]
# librosa.output.write_wav('dk.wav', crop, 16000)
    # y2,ind = librosa.effects.trim(y,top_db=30)
    # y2 = maintain_size(y2,16000)
    # librosa.output.write_wav(wav, y2, 16000)

# labels = [x.strip() for x in open('class_list.txt')]
# x = generateAnnotationList(dataPath,labels)
# print(x[0])
# print(len(x))
# F = open('class_list.txt', 'w')
# L = os.listdir('./data/benchmark/validation/') 
# for i in L:
#     F.write(str(i) + "\n")

# F.close()



# dataPath = './data/test/'
# wavs = librosa.util.find_files(dataPath)
# es = 0

# augmentor = Augmentor(noise_chance=0.6,shift_chance=0.5,playspeed_chance=.6,amplify_chance=.5,sr=16000)
# for i,wav in enumerate(wavs):
#     print('Processing Record: ',str((i+1)/len(wavs)),wav,' ....')
#     mc = load_wav(wav,augmentor)
#     print(mc.shape)

# print(es)
# print('removed: ',len(wavs)-len(es))        
# colors = (0,0,0)
# area = np.pi*3
# indexes = np.arange(start=0,stop=len(es),step=1)
# # Plot
# plt.scatter(es,indexes, s=area, c=colors, alpha=0.5)
# plt.title('E')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


# dataPath = 'C:/Users/Void/Desktop/SOUND/data/small_data/on/'
# offpath = 'C:/Users/Void/Desktop/SOUND/data/small_data/off/'
# wavs = librosa.util.find_files(dataPath)
# wavsoff = librosa.util.find_files(offpath)
# shuffle(wavs)
# shuffle(wavsoff)
# c=0
# m1=0
# m2=0
# for i,wav in enumerate(wavs):
#     print('Processing Record: ',str((i+1)/len(wavs)),wav,' ....')
#     y,sr=librosa.load(wav,sr=None,)
#     r = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85,center=False,n_fft=128,hop_length=None)
#     y = preprocess_wav(y)
#     #S = librosa.feature.melspectrogram(y, sr=sr, n_fft=128,hop_length=None,center=False)
#     #z = librosa.feature.zero_crossing_rate(y,frame_length=18,hop_length=(len(y)//18),center=False)
#     y2,sr2=librosa.load(wavsoff[i],sr=None,)
#     y2 = preprocess_wav(y2)
#     r2 = librosa.feature.spectral_rolloff(y=y2, sr=sr2, roll_percent=0.85,center=False,n_fft=128,hop_length=None)
#     print('Processing Record: ',str((i+1)/len(wavsoff[i])),wavsoff[i],' ....')
#     #z2 = librosa.feature.zero_crossing_rate(y2,frame_length=18,hop_length=(len(y2)//18),center=False)
#     # print(np.mean(z[0]) < np.mean(z2[0]))
#     # if (np.mean(z[0]) < np.mean(z2[0])): c+=1
#     m1+=np.sum(np.square(y[(len(y)//3)*2:]))
#     m2+=np.sum(np.square(y2[(len(y2)//3)*2:]))


# print(m1/len(wavs), m2/len(wavsoff))

# onf = './h_on.wav'
# off = './h_off.wav'

# y,sr=librosa.load(onf,sr=None)
# y2,sr=librosa.load(off,sr=None)
# num_frames = 5

# dataPath = 'C:/Users/Void/Desktop/SOUND/data/small_data/off/'
# wavs = librosa.util.find_files(dataPath)
# fm = 0
# for i,wav in enumerate(wavs):
#     print('Processing Record: ',str((i+1)),' ....')
#     y,sr=librosa.load(wav,sr=None,)
#     y=preprocess_wav(y)
#     #spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr,n_fft=128,center=False)[0]
#     #rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,n_fft=128,center=False)[0]
#     #tonnetz = librosa.feature.tonnetz(y=y, sr=sr,)[0]
#     contrast = librosa.feature.spectral_contrast(y=y, sr=sr,n_fft=512)
#     print(contrast.shape)
#     fm+=np.mean(contrast)
#     #fm+=np.mean(spec_bw)
    
# print(fm/len(wavs))



# p1 = np.abs(librosa.stft(y,n_fft=128))
# p2 = np.abs(librosa.stft(y2,n_fft=128,))




# print(p1.shape,p2.shape)
# rms = librosa.feature.rms(S=p1,)
# print(rms.shape)
# print(rms)


# rms = librosa.feature.rms(S=p2,)
# print(rms.shape)
# print(rms)
#play_audio(y,sr)

#y = y[y>2.0686559e-05]
#y = y*(np.max(y)/np.min(y))
#y,ind=librosa.effects.trim(y,top_db=20)
#play_audio(y,sr)
#plot_wave({'data':y,'sr':sr,'title':'ON'},{'data':y2,'sr':sr,'title':'OFF'})



########################################################################
# y,sr=librosa.load(data_path+video_path2,sr=None,res_type='scipy')
# y2,sr2=librosa.load(data_path+video_path,sr=None,res_type='scipy')

# p1 = np.abs(librosa.stft(y,n_fft=128))
# p2 = np.abs(librosa.stft(y2,n_fft=128))

# plot_wave({'data':y,'sr':sr,'title':'ON'},{'data':y2,'sr':sr2,'title':'OFF'})
# calculate_mel_spect(y,sr)
# #plot_spectrum(p2)
# calculate_mel_spect(y2,sr2)

# # #trim
# y,index = librosa.effects.trim(y,top_db=9)
# y2,index = librosa.effects.trim(y2,top_db=8)
# p1 = np.abs(librosa.stft(y,n_fft=128))
# p2 = np.abs(librosa.stft(y2,n_fft=128))
# plot_wave({'data':y,'sr':sr,'title':'ON'},{'data':y2,'sr':sr2,'title':'OFF'})
# #plot_spectrum(p1)
# calculate_mel_spect(y,sr)
# #plot_spectrum(p2)
# calculate_mel_spect(y2,sr2)

# # #sd.play(y2,samplerate=sr)
# # print(y.shape,y2.shape)

# # plot_wave({'data':y,'sr':sr,'title':'ON'},{'data':y2,'sr':sr2,'title':'OFF'})
