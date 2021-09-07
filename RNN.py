from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,BatchNormalization , Activation,Conv1D
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data import generateAnnotationList,DataGenerator
from augmentation import Augmentor
from data import load_wav
import numpy as np
import json
import os
import librosa
def build_conv_lstm_model():
    model = Sequential()
    model.add(Conv1D(256, 10, strides=4,input_shape=(12,28)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu',dropout=0.2,return_sequences=True,))
    model.add(LSTM(128, activation='relu',dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(35, activation='softmax'))
    #model.load_weights('LSTM_benchmark_nodense_8.hdf5')
    adam =Adam(lr=0.0001,beta_1=0.9, beta_2=0.999, amsgrad=True)
    #sgd = SGD(lr=0.0005, decay = 1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy',])
    model._make_predict_function()
    print(model.summary())
    return model

def build_model(num_classes,input_shape=(12,28),weights = None, withTop=True):
    model = Sequential()
    model.add(LSTM(256,return_sequences=True,input_shape=input_shape))
    model.add(BatchNormalization(axis=2,))
    model.add(LSTM(256))
    model.add(BatchNormalization(axis=1,))
    model.add(Dense(512, activation='linear'))
    if not withTop:
        model.add(Dense(35, activation='softmax')) # TODO solve num of classes for NO TOP
    else: 
        model.add(Dense(num_classes, activation='softmax')) # TODO solve num of classes for NO TOP    
    if weights:
        model.load_weights(weights)
    if not withTop:
        model.pop()
        model.pop()    
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(512, activation='linear'))    
        model.add(Dense(num_classes, activation='softmax'))

    adam =Adam(lr=0.001,beta_1=0.9, beta_2=0.999, amsgrad=True)
    #sgd = SGD(lr=0.0005, decay = 1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy',])
    model._make_predict_function()
    print(model.summary())
    return model

def trainDefault(trainPath,validPath,class_list,inputDim=(12,28),batch_size=8,epochs=1):
    if class_list:
        labels = [x.strip() for x in open(class_list)]
    else :    labels = sorted(os.listdir(trainPath))
    
    augmentor = Augmentor(noise_chance=0.8,shift_chance=0.8,playspeed_chance=.8,amplify_chance=.0,sr=16000)
    print('\n\n\ngenerating Annotation List...')
    annoListT = generateAnnotationList(trainPath,labels)
    annoListV = generateAnnotationList(validPath,labels)

    print('creating data generator...')
    dataGeneratorT = DataGenerator(annoListT,augmentor=None,batch_size=311,n_classes=len(labels),shuffle=True,dim=(inputDim))
    dataGeneratorV = DataGenerator(annoListV,batch_size=125,n_classes=len(labels),shuffle=False,dim=(inputDim))

    print('Building Model...')
    model = build_model(num_classes=len(labels),input_shape=inputDim,withTop=True)
    
    print('starting...\n')

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('finetuned_model-{epoch:02d}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

    res = model.fit_generator(dataGeneratorT, epochs=epochs, 
                            verbose=1, callbacks=[earlystop, model_checkpoint],
                            validation_data=dataGeneratorV,
                            shuffle=False,)
    print(res.history)
    with open('training_history_finetuned.json', 'w') as f: 
        json.dump(res.history, f, indent=4)  
    model.save('final_model.hdf5')
def test(dataPath,model_weights):
    labels = sorted(os.listdir(dataPath))
    model = build_model(len(labels),weights=model_weights)
    wavs = librosa.util.find_files(dataPath)
    true_counter = 0
    for i,wav in enumerate(wavs):
        path= wav.replace('\\','/').split('/')
        true_label = path[-2]
        clip_name = path[-1]
        clip = load_wav(wav,augmentor=None)
        clip = np.expand_dims(clip,axis=0)

        predictions = model.predict(clip)
        label_index = np.argmax(predictions,axis=1)[0]
        label = labels[label_index]
        score = predictions[0][label_index]
        print('Record: ',clip_name,' Prediction: ',label,' Score',str(round(score*100,2))+'%','  - Progress: '+str(round((i+1)/len(wavs),2)*100)+'%',' ....')
        if label == true_label:
            true_counter+=1
    print('Accuracy: ',str(round(true_counter/len(wavs),4)*100)+'%')
if __name__ == "__main__":
    trainDefault('./data/benchmark/training/','./data/benchmark/validation/',None,epochs=5,batch_size=64)    
    #test('./data/finetune_data/validate/','./model/small_model.hdf5')    