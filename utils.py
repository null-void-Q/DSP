import sounddevice as sd
import librosa
import numpy as np
import matplotlib.pyplot as plt


def maintain_size(y,size,center_crop=False):
    
    input_length = len(y)

    if input_length == size:
        return y

    if input_length > size:
        if center_crop:
            cropat = (input_length//2 - size//2  )
            y = y[cropat:((size - size//2 )+ input_length//2)]
        else: y = y[input_length-size:]    
    else:
        h = (size - len(y))//2
        b = np.zeros((h))
        e = np.zeros(((size-len(y))-h))
        y = np.concatenate((b,np.concatenate((y,e))))
        #y = np.pad(y, (0, max(0, size - len(y))), "constant")
    return y    


def calculate_fft(y):
    return np.abs(librosa.stft(y))
def calculate_mel_spect(samples,sample_rate):
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_fft=128,hop_length=1,)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()

def plot_spectrum(data):
    librosa.display.specshow(librosa.amplitude_to_db(data,ref=np.max),y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()


def plot_time_series(data):
    fig = plt.figure(figsize=(14, 8))
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(data)), data)
    plt.show()

def plot_wave(*plots):
    pn = len(plots)
    m = pn*2 - 1
    c = 0
    plt.figure()
    for pt in plots:
        if c == 0:
            plt.subplot(m,1,c+1)
        else:
            plt.subplot(m,1,c+2)
        librosa.display.waveplot(pt['data'],pt['sr'])
        plt.title(pt['title'])
        c+=1
    plt.show()

def clean_audio(samples,threshhold = 9):
    return librosa.effects.trim(samples,top_db=threshhold)
def read_audio_file(file_pth):
    return librosa.load(file_pth,sr=None,res_type='scipy',dtype=np.int16,mono=False)
def play_audio(samples,sr):
    sd.play(samples,samplerate=sr)
    sd.wait()
