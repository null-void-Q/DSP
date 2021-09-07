import librosa
import matplotlib.pyplot as plt
import numpy as np
from utils import play_audio
import time
import random


def add_noise(y,minIntensity= -0.05,maxIntensity = 0.05):
    noise = np.random.uniform(low=minIntensity,high=maxIntensity,size=y.shape)
    y_n = y + noise
    return y_n

def shift_signal(y,rate): 
    shift = int(rate*len(y))
    y_shifted = np.roll(y,shift)
    return y_shifted
def pitch_shift(y,sr,pitch):
    return librosa.effects.pitch_shift(y,sr,n_steps=pitch,center=False)
def play_speed(y, rate=1, keep_size=True):
    input_length = len(y)
    y = librosa.effects.time_stretch(y, rate)
    o_length = len(y)
    if keep_size:
        if o_length>input_length:
            cropat = (o_length//2 - input_length//2)
            y = y[cropat:((input_length - input_length//2 )+ o_length//2)]
        else:
            y = np.pad(y, (0, max(0, input_length - len(y))), "constant")

    return y

def amplify(y,rate = 0.2):
    y = y * rate
    if rate > 1:  
        y[y > +1] = +1
        y[y < -1] = -1
    return y

def padd_zeros(y,size=16000):
    pads = np.zeros((size-len(y)))
    return np.concatenate((y,pads),axis=0)

class Augmentor:
    def __init__(self,aug_chance = 0.8,noise_chance = 0.5,shift_chance = 0.5,playspeed_chance = 0.5,amplify_chance = 0.5
                ,noise_range = [0.01,0.10],shift_range = [-0.25,0.25],playspeed_range = [0.7,1.3],amplify_range = [0.2,1.6],
                pitch_range=[-5,5], sr = 16000):

                self.aug_chance = aug_chance
                self.noise_chance=noise_chance
                self.shift_chance=shift_chance
                self.playspeed_chance = playspeed_chance
                self.amplify_chance = amplify_chance

                self.noise_range = noise_range
                self.shift_range = shift_range
                self.playspeed_range = playspeed_range
                self. amplify_range = amplify_range
                self.pitch_range=pitch_range

                self.sr = sr

                random.seed(time.time())
                
    def augment(self,y):
        if (random.randint(0,101)/100) < self.shift_chance:
                selection_array = np.arange(start=self.shift_range[0],stop=self.shift_range[1],step=0.05)
                shift = selection_array[random.randint(0,len(selection_array)-1)]
                y = shift_signal(y,shift)
                
        if (random.randint(0,101)/100) < self.playspeed_chance:
                if (random.randint(0,101)/100) < 0.5:
                    selection_array = np.arange(start=self.playspeed_range[0],stop=self.playspeed_range[1],step=0.1)
                    speed = selection_array[random.randint(0,len(selection_array)-1)]
                    y = play_speed(y,speed, keep_size=True)
                    
                else:    
                    selection_array = np.arange(start=self.pitch_range[0],stop=self.pitch_range[1],step=1)
                    pitch = selection_array[random.randint(0,len(selection_array)-1)]
                    y = pitch_shift(y,self.sr,pitch)
                    
        if (random.randint(0,101)/100) < self.amplify_chance:
                selection_array = np.arange(start=self.amplify_range[0],stop=self.amplify_range[1],step=0.2)
                amp = selection_array[random.randint(0,len(selection_array)-1)]
                y = amplify(y,amp)
                

        if (random.randint(0,101)/100) < self.noise_chance:
                selection_array = np.arange(start=self.noise_range[0],stop=self.noise_range[1],step=0.01)
                noise_intensity = selection_array[random.randint(0,len(selection_array)-1)]
                y = add_noise(y,minIntensity=-1*noise_intensity,maxIntensity = noise_intensity)
                

        return y

        
if __name__ == "__main__":
    y,sr=librosa.load('h_off.wav',sr=None,)
    y = librosa.util.normalize(y)
    play_audio(y,sr)

    y_noise = add_noise(y,minIntensity=-0.1,maxIntensity = 0.1)
    play_audio(y_noise ,sr)

    y_shifted = shift_signal(y,-0.2)
    play_audio(y_shifted,sr)

    y_pitch = pitch_shift(y,sr,-2)
    play_audio(y_pitch,sr)

    y_stretched = play_speed(y,0.1,keep_size=False)
    play_audio(y_stretched,sr)

    y_amplified = amplify(y,2)
    play_audio(y_amplified,sr)

    figures = 6
    fig =plt.figure(figsize=(12,20))
    plt.subplot(figures,1,1)
    plt.title('Raw wave')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(y)), y)

    
    plt.subplot(figures,1,2)
    plt.title('Shiffted')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(y_shifted)), y_shifted)

    plt.subplot(figures,1,3)
    plt.title('Stretched')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(y_stretched)), y_stretched)

    plt.subplot(figures,1,4)
    plt.title('Amplified')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(y_amplified)), y_amplified)

    plt.subplot(figures,1,5)
    plt.title('NOISE')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(y_noise)), y_noise)


    plt.subplot(figures,1,6)
    plt.title('pitch')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(y_pitch)), y_pitch)
    fig.tight_layout()
    plt.show()