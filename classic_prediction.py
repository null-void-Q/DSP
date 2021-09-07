import argparse
import tempfile
import queue
import sys

import sounddevice as sd
import soundfile as sf
import numpy as np  
assert np  
import threading
import librosa
from scipy.spatial.distance import cosine
# python app.py -r 16000 -c 1

NUM_INPUT_SAMPELS = 16000
sr= 16000
class_list = 'label_list.txt'

thresholds = [0,1,0,0.0,0.0]

currentState =-1
resetTime = 5 #secs
leds = [0,0] #green red

def updateLEDs():
    global leds, currentState
    #while True:
    print('Green: ',leds[0],' Red: ',leds[1])

def updateLEDstate(led,state):
    global leds
    if led == 0:
        leds[0] = state
    else : leds[1] = state   

def handelCommand(label_index):
    global currentState

    if label_index == 0:
        updateLEDstate(1,1)
        updateLEDstate(0,1)
    else:
        updateLEDstate(1,0)
        updateLEDstate(0,0)    

    # if currentState < 0 :
    #     currentState = label_index
    # elif currentState == 2 or currentState == 3: #off/on first
    #     if label_index ==  0 or currentState == 4:
    #         updateLEDstate(led = label_index, state = currentState-2)
    #         resetState()
    #     else : currentState = label_index
    # else: #green/red first
    #     if label_index ==  2 or label_index == 3:
    #         updateLEDstate(led = currentState, state = label_index-2)
    #         resetState()
    #     else : currentState = label_index     

def predict(y,model_on,model_off):
        
    y = librosa.util.normalize(y)
    y-=np.mean(y)
    y,ind=librosa.effects.trim(y,top_db=32)
    z = librosa.feature.zero_crossing_rate(y,frame_length=50,hop_length=(len(y)//50),center=False)[0]

    label = 0
    if cosine(z,model_on) > cosine(z,model_off):
        label = 1
    
    return label
    
def isAcceptable(label_index,score):
    return score > thresholds[label_index]
def resetState():
    global currentState
    currentState = -1

def recordFromMic(duration,fs,channels):
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels,)
    sd.wait()
    return myrecording

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

        
    labels = [x.strip() for x in open(class_list)] # TODO useless in final implementation
       

    model_on=np.load('model_on.npy')[4:]
    model_off=np.load('model_off.npy')[4:]


    resetState() # set initial state
        
        # thread to keep updating LEDs
    led_handler = threading.Thread(target=updateLEDs,)
    led_handler.daemon = True
    led_handler.start()

    print('#' * 80)
    print('Session Started: press Ctrl+C to terminate.')
    print('#' * 80)
    resetCounter = 0
    while True:#do processing and prediction here
        #print(leds,' - ',currentState)
        nb = input('press any key to record.')
        if str(nb) == 'q':
            break
        y = recordFromMic(duration=1,fs=16000,channels=1).flatten()
        label_index = predict(y,model_on,model_off)
        handelCommand(label_index)
        print('Green: ',leds[0],' Red: ',leds[1])
            


except KeyboardInterrupt:
    print('\n- Session Done. Processes terminated!')
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))