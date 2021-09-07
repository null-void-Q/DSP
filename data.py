import librosa
import keras 
import numpy as np
    
def load_wav(wav,augmentor):
    y,sr=librosa.load(wav,sr=None,)
    y = librosa.util.normalize(y)
    if augmentor:
        y = augmentor.augment(y)
    mc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12,center=False)
    mc = librosa.util.normalize(mc,axis=1)
    return mc

def generateAnnotationList(dataPath,class_list):
    wavs = librosa.util.find_files(dataPath)
    alist = []
    for i,wav in enumerate(wavs):
        print('Generating Annotation List: ',str(round((i+1)/len(wavs),2),),' ....')
        path = wav
        label_name = path.replace('\\','/').split('/')[-2]
        label = class_list.index(label_name)
        alist.append({'file':path,'label_name':label_name,'label':label})
    return alist 


class DataGenerator(keras.utils.Sequence):
     
    def __init__(self, annotationList,augmentor=None,batch_size=32,dim=(12,28),
                 n_classes=35,shuffle=False):
       
        self.dim = dim
        self.augmentor = augmentor
        self.batch_size = batch_size
        self.annotationList = annotationList
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.annotationList) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        tmpAnnoList = [self.annotationList[k] for k in indexes]

        X, y = self.__data_generation(tmpAnnoList)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.annotationList))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tmpAnnoList):
         
        X = np.empty((self.batch_size,*self.dim), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)

        for i, WAV in enumerate(tmpAnnoList):
            X[i] = load_wav(WAV['file'],self.augmentor)

            y[i] = int(WAV['label'])
        
        return X, y#keras.utils.to_categorical(y, num_classes=self.n_classes)