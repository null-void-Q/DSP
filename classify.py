import numpy as np
import sys
from RNN import build_model
from data import load_wav
import librosa
import sounddevice as sd
def recordFromMic(duration,fs,channels):
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels,)
    sd.wait()
    return myrecording

weights_path='./model/small_model.hdf5'
sr =16000
# audio_clip_path = 'h_off.wav'
# if len(sys.argv) > 1:
#     audio_clip_path=sys.argv[1]

class_list = 'label_list.txt'
labels = [x.strip() for x in open(class_list)]


model = build_model(len(labels),weights=weights_path)

while True:
    nb = input('press any key to record.')
    if str(nb) == 'q':
        break
    clip = recordFromMic(duration=1,fs=16000,channels=1).flatten()
    y = librosa.util.normalize(clip)
    mc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12,center=False)
    mc = librosa.util.normalize(mc,axis=1)
    clip = np.expand_dims(mc,axis=0)

    predictions = model.predict(clip)
    label_index = np.argmax(predictions,axis=1)[0]
    score = predictions[0][label_index]
    label = labels[label_index]
    print('Prediction: ',label,' Score',str(round(score*100,2))+'%')

























# import sys
# from scipy.spatial.distance import cosine,euclidean
# from classic_approach import calculate_energy,calculate_ZCR,spec_rolloff,preprocess_wav
# import librosa
# import numpy as np


# def predict(y,model_on,model_off,threshold=0.2,sr=16000, numOfFrames = 45): # zcr
        
#         y=preprocess_wav(y)
#         zcr = calculate_ZCR(y,sr,num_frames=numOfFrames)
#         label = 0
#         on_distance = cosine(zcr,model_on)
#         off_distance = cosine(zcr,model_off)
#         if on_distance > threshold or off_distance > threshold:
#             return -1 
#         print(on_distance,off_distance)
#         if  on_distance > off_distance:
#             label = 1
#         return label
# file_path = sys.argv[1]
# model_on = np.load('model_on.npy')
# model_off = np.load('model_off.npy')
# y,sr=librosa.load(file_path,sr=16000)

# prediction = predict(y,model_on[8:],model_off[8:],threshold=100)
# res = 'ON\n' 
# if prediction==1:res ='OFF\n' 
# elif prediction==-1: res ='UNKOWN\n'
# print(res)    
