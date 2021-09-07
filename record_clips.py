import sounddevice as sd
import numpy as np
from librosa.output import write_wav

def recordFromMic(duration,fs,channels):
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels,)
    sd.wait()
    return myrecording

if __name__ == "__main__":
    c = 0
    while True:
        nb = input('press any key to record.')
        if str(nb) == 'q':
            break
        record = recordFromMic(1,16000,1)
        write_wav('rec'+str(c)+'.wav',record,16000)
        c+=1