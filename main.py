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
from RNN import build_model

# python app.py -r 16000 -c 1

NUM_INPUT_SAMPELS = 16000
weights_path='./model/small_model.hdf5'
sr= 16000
class_list = 'label_list.txt'

thresholds = [0.85,1,0.60,0.6,0.5]

currentState =-1
resetTime = 5 #secs
leds = [0,0] #green red

def updateLEDs():
    global leds, currentState
    while True:
        print('Green: ',leds[0],' Red: ',leds[1])

def updateLEDstate(led,state):
    global leds
    if led == 0:
        leds[0] = state
    else : leds[1] = state   

def handelCommand(label_index):
    global currentState
    if currentState < 0 :
        currentState = label_index
    elif currentState == 2 or currentState == 3: #off/on first
        if label_index ==  0 or currentState == 4:
            updateLEDstate(led = label_index, state = currentState-2)
            resetState()
        else : currentState = label_index
    else: #green/red first
        if label_index ==  2 or label_index == 3:
            updateLEDstate(led = currentState, state = label_index-2)
            resetState()
        else : currentState = label_index     

def predict(y,model):
        
    y = librosa.util.normalize(y)
    mc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12,center=False)
    mc = librosa.util.normalize(mc,axis=1)
    clip = np.expand_dims(mc,axis=0)

    predictions = model.predict(clip)
    label_index = np.argmax(predictions,axis=1)[0]
    score = predictions[0][label_index]

    return label_index,round(score,4)
    
def isAcceptable(label_index,score):
    return score > thresholds[label_index]
def resetState():
    global currentState
    currentState = -1
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
    global q
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
                        channels=args.channels, callback=callback,dtype=np.int16,blocksize=512):

        
        labels = [x.strip() for x in open(class_list)] # TODO useless in final implementation
        model = build_model(len(labels),weights=weights_path) # load model
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
            if q.qsize() > NUM_INPUT_SAMPELS:
                frame = []
                for i in range(NUM_INPUT_SAMPELS):
                    frame.extend(q.get())
                y = np.asarray(frame,dtype=np.float32).flatten()
                label_index,score = predict(y,model)
                if isAcceptable(label_index,score):
                    handelCommand(label_index)
                    label = labels[label_index]
                    resetCounter = 0
                    #print('Prediction: ',label,' Score',str(score*100)+'%')
                else:
                     resetCounter+=1
                     if resetCounter == resetTime:
                         resetState()     
except KeyboardInterrupt:
    print('\n- Session Done. Processes terminated!')
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))