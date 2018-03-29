
import os, subprocess
from pytube import YouTube
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

x_train = np.random.random((1000, 100))
y_train = np.random.randint(2, size=(100, 10))
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=500, batch_size=32)


class PredictionData( np.ndarray ):
    maxIndex = 0
    numElements = 0

    def __new__( cls, inputArray, maxIndex ):
        return

    def overloadedShape( self, ):
        t = np.shape( )
        

    def __getitem__( self, index ):
        ratio = index / maxIndex
        elementIndex = numElements * ratio
        elementIndex = int( elementIndex )
        return super( PredictionData, self ).__getitem__( elementIndex )

    def shape( self, index ):
        return




class DataHandler:
    dataPath = r"./data/"


    def __init__( self ):
        print( os.getcwd() )
        dataDir = os.path.dirname( self.dataPath );
        if not os.path.exists( dataDir ):
            os.makedirs( dataDir )
        return

    def extractData( self, videoDataPath, videoName ):
        videoPath = videoDataPath + videoName
        audioPath = videoDataPath + "audio.wav"
        command = 'ffmpeg -i "%s" -ab 160k -ac 2 -ar 44100 -vn %s' % ( videoPath, audioPath )
        print( command )
        subprocess.call( command, shell=True )


    def getNextData( self ):
        yt = YouTube( "https://www.youtube.com/watch?v=XOic6pVAN30" )
        yts =  yt.streams.filter( progressive=True )
        print( yts.all() )
        test = yts.first()
        videoDataPath = self.dataPath + yt.video_id + '/'
        if not os.path.exists( videoDataPath ):
            videoDir = os.path.dirname( videoDataPath )
            os.makedirs(  videoDir )
            test.download( videoDataPath )
            
        self.extractData( videoDataPath, test.default_filename )
        return

