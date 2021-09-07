import numpy as np
from librosa.feature import zero_crossing_rate
from scipy.spatial.distance import cosine,euclidean
import librosa

numOfsections = 50
numOfRolloffs = 4

def preprocess_wav(y):
    y = librosa.util.normalize(y)
    y-=np.mean(y)
    y,ind=librosa.effects.trim(y,top_db=32)
    return y

def train(train_path):
    wavs = librosa.util.find_files(train_path)
    energies = []
    zcrs = []
    spec_bw = []
    for i,wav in enumerate(wavs):
        print('Processing Record: ',str((i+1)),' ',wav,' ....')
        y,sr=librosa.load(wav,sr=16000)
        y=preprocess_wav(y)
        #energies.append(calculate_energy(y))
        spec_bw.append(spec_rolloff(y,sr,numOfRolloffs))
        zcrs.append(calculate_ZCR(y,sr,num_frames=numOfsections))
    return [*np.mean(spec_bw,axis=0),*np.median(zcrs,axis=0)]#np.mean(energies)


def test(test_path,model_on,model_off,true_label):
    wavs = librosa.util.find_files(test_path)
    true_count = 0
    for i,wav in enumerate(wavs):
        print('Processing Record: ',str((i+1)),' ....')
        y,sr=librosa.load(wav,sr=16000,)
        y=preprocess_wav(y)
        #energy = calculate_energy(y)
        spec_bw = spec_rolloff(y,sr,numOfRolloffs)
        zcr = calculate_ZCR(y,sr,num_frames=numOfsections)
        d = [*spec_bw,*zcr]
        label = 0
        if cosine(d,model_on) > cosine(d,model_off):
            label = 1

        if label == true_label:
            true_count+=1
        
    return true_count/len(wavs)


def test_rlf(test_path,model_on,model_off,true_label):
    wavs = librosa.util.find_files(test_path)
    true_count = 0
    for i,wav in enumerate(wavs):
        print('Processing Record: ',str((i+1)),' ....')
        y,sr=librosa.load(wav,sr=16000,)
        y=preprocess_wav(y)
        d = spec_rolloff(y,sr,numOfRolloffs)
        label = 0
        if cosine(d,model_on) > cosine(d,model_off):
            label = 1

        if label == true_label:
            true_count+=1
        
    return true_count/len(wavs)

def test_zcr(test_path,model_on,model_off,true_label):        
    wavs = librosa.util.find_files(test_path)
    true_count = 0
    for i,wav in enumerate(wavs):
        print('Processing Record: ',str((i+1)),' ....')
        y,sr=librosa.load(wav,sr=16000,)
        y=preprocess_wav(y)
        d = calculate_ZCR(y,sr,num_frames=numOfsections)
    
        label = 0
        if cosine(d,model_on) > cosine(d,model_off):
            label = 1

        if label == true_label:
            true_count+=1
        
    return true_count/len(wavs)

def calculate_energy(y):
    return np.sum(np.square(y))

def calculate_ZCR(y,sr,num_frames = 4):
     z = zero_crossing_rate(y,frame_length=num_frames,hop_length=(len(y)//num_frames),center=False)[0]
     return z
def spec_rolloff(y,sr,num_frames = 4):
    r = librosa.feature.spectral_rolloff(y=y, sr=sr,n_fft=1024,hop_length=512,roll_percent=0.5,center=False)[0]
    rolloff_rates = []
    base =(len(r)//num_frames)
    if base < 1 : 
        print('num of frames is greater than original samples')
        exit(1)
    for i in range(num_frames-1):
        rolloff_rates.append(np.mean(r[base*(i):base*(i+1)]))
    rolloff_rates.append(np.mean(r[base*(num_frames-1):])) 
    return rolloff_rates

if __name__ == "__main__":
        
    train_path_off= './data/small_data/off/'
    train_path_on= './data/small_data/on/'
    test_path_off= './data/small_data/off/'
    test_path_on= './data/small_data/on/'


    model_off = train(train_path_off)
    model_on = train(train_path_on)
    np.save('model_off',model_off)
    np.save('model_on',model_on)
    results = []

    acc_on = test(test_path_on,model_on,model_off,0)
    acc_off = test(test_path_off,model_on,model_off,1)

    results.append([acc_on,acc_off])
    
    acc_on = test_rlf(test_path_on,model_on[:numOfRolloffs],model_off[:numOfRolloffs],0)
    acc_off = test_rlf(test_path_off,model_on[:numOfRolloffs],model_off[:numOfRolloffs],1)

    results.append([acc_on,acc_off])


    acc_on = test_zcr(test_path_on,model_on[numOfRolloffs:],model_off[numOfRolloffs:],0)
    acc_off = test_zcr(test_path_off,model_on[numOfRolloffs:],model_off[numOfRolloffs:],1)

    results.append([acc_on,acc_off])

    print('\n\n**************************************\n')
    print('Results:')
    print('ZCR Accuracy: ',(results[2][0]+results[2][1])/2)
    print('RollOff Accuracy',(results[1][0]+results[1][1])/2)

