import argparse
import tempfile
import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)
from scipy.spatial.distance import cosine,euclidean
from classic_approach import calculate_energy,calculate_ZCR,spec_rolloff,preprocess_wav
import librosa

# python app.py -r 16000 -c 1

NUM_INPUT_SAMPELS = 24000

def isSilence(y,threshold = 0.15):
    flatness = numpy.mean(librosa.feature.spectral_flatness(y=y,n_fft=128,center=False))
    print(flatness)
    return flatness > threshold

def predict(y,model_on,model_off,threshold=0.2,sr=16000, numOfFrames = 45): # zcr and rolloff
        
        y=preprocess_wav(y)
        zcr = calculate_ZCR(y,sr,num_frames=numOfFrames)
        label = 0
        on_distance = cosine(zcr,model_on)
        off_distance = cosine(zcr,model_off)
        print(on_distance,off_distance)
        if on_distance > threshold and off_distance > threshold:
            return -1 
        if  on_distance > off_distance:
            label = 1
        return label

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate',default=16000)
parser.add_argument(
    '-c', '--channels', type=int, default=1, help='number of input channels')
args = parser.parse_args(remaining)

q = queue.Queue() # main data holder


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    d =  [*(indata.copy())]
    for s in d:      
        q.put(s)

try:
    if args.samplerate is None:
        #device_info = sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        #args.samplerate = int(device_info['default_samplerate'])
        args.samplerate = 16000


    with sd.InputStream(samplerate=args.samplerate, device=args.device,
                        channels=args.channels, callback=callback,dtype=numpy.int16,blocksize=512):
        print('#' * 80)
        print('Session Started: press Ctrl+C to terminate.')
        print('#' * 80)

        model_on = numpy.load('model_on.npy')[8:]
        model_off = numpy.load('model_off.npy')[8:]

        while True:#do processing and prediction here
            if q.qsize() > NUM_INPUT_SAMPELS:
                frame = []
                for i in range(NUM_INPUT_SAMPELS):
                    frame.extend(q.get())
                y = numpy.asarray(frame,dtype=numpy.float32).flatten()
                #if not isSilence(y,threshold=0.05):
                prediction = predict(y,model_on,model_off,threshold=0.1)
                res = 'ON\n' 
                if prediction==1:res ='OFF\n' 
                elif prediction==-1: res =''#'UNKOWN\n'
                print(res)    
except KeyboardInterrupt:
    print('\n- Session Done. Processes terminated!')
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))