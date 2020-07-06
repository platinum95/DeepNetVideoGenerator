#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:34:13 2020

@author: platinum95
"""

import os, subprocess, time, sys

import pytube
from pytube import YouTube, Playlist
import wave
import glob
import librosa
import librosa.display
import numpy as np
import pickle

imageRes = 224

class AudioLoader():
    audioPath = r""
    fftData = []
#    mfcData = []
    numFrames = 0
    windowLength = 1024
    samplerate = 44100
    numMelBands = 512
    
    def __init__( self, path, numFrames=None ):
        self.numFrames = numFrames
        audio = wave.open( path )
        numSamples = audio.getnframes()
        if( numFrames == None ):
            rate = audio.getframerate()
            duration = math.ceil( float(numSamples) / float(rate) )
            # Video will be at 25 fps
            numFrames = duration * 25
        
        assert numFrames > 0
        
        print( "Num audio samples: %i" % numSamples )
        samplesRaw = audio.readframes( audio.getnframes() )
        npSampleArray = np.frombuffer( samplesRaw, dtype=np.int16 ).astype( np.float32 )
        npSampleArray /= float( ( 2 ** 15 ) - 1 )
        npSampleArray -= np.mean( npSampleArray )
        npSampleArray /= np.std( npSampleArray )
        
        self.windowLength = numSamples // numFrames
        print( "Window Length: %d" % self.windowLength )
        windowsAligned = True
        if( numSamples % numFrames != 0 ):
            windowsAligned = False
            self.windowLength += 1
            numRequiredSamples = self.windowLength * numFrames
            assert( numRequiredSamples > numSamples )
            padSize = numRequiredSamples - numSamples
            print( "Padding with %d samples" % padSize )
            npSampleArray = np.pad( npSampleArray, ( 0, padSize ), 'constant' )
            print( "Now have %d samples" % npSampleArray.shape[ 0 ] )
            numSamples = npSampleArray.shape[ 0 ]
        
        pickledDataPath = path + ".pkl"
        if os.path.isfile( pickledDataPath ):
            dataDict = pickle.load( open( pickledDataPath,'rb' ) )           
            self.cqt = dataDict[ 'cqt' ]
            return
            
        numWindows = ( numSamples // self.windowLength )
        assert numWindows == self.numFrames, '{} != {}'.format( numWindows, self.numFrames )
        
        fftWindowSize = self.windowLength
        fftWindowHop = self.windowLength
#        mfc = librosa.feature.melspectrogram( npSampleArray, sr=self.samplerate,
#                                              n_fft=fftWindowSize,
#                                              hop_length=fftWindowHop,
 #                                             window=scipy.signal.windows.boxcar,
 #                                             center=False,
#                                              n_mels=self.numMelBands )
        
        # output should be (numMelBands, numFrames) so swap axes
#        mfc = np.swapaxes( mfc, 0, 1 )
#        for winIdx in range( mfc.shape[ 0 ] ):            
      #      mfc[ winIdx ] -= np.mean( mfc[ winIdx ] )
        #    mfc[ winIdx ] /= np.std( mfc[ winIdx ] )
        
       # mfc = mfc / mfcStd
        
 #       self.mfcData = mfc # np.empty( ( self.numFrames, 1024 )
#        self.fftData = librosa.core.stft( npSampleArray,
#                                          n_fft=fftWindowSize,
#                                          window=scipy.signal.windows.boxcar,
#                                          hop_length=fftWindowHop,
#                                          center=False )
        
#        self.fftData =  np.absolute( self.fftData )
#        self.fftData = np.swapaxes( self.fftData, 0, 1 )
#        self.fftData /= fftWindowSize
        
#        nyquistFreq = ( 1.0 / self.samplerate ) / 2.0
        numOctaves = 8
        binsPerOctave = 16
        numBins = numOctaves * binsPerOctave
        
        cqt = librosa.core.pseudo_cqt( npSampleArray,
                                       bins_per_octave=binsPerOctave,
                                       n_bins=numBins,
                                       sr=self.samplerate,
                                       window=scipy.signal.windows.boxcar,
                                       hop_length=fftWindowHop )
        cqt = np.swapaxes( cqt, 0, 1 )
        cqt /= np.std( cqt )
  #      if not windowsAligned:
  #          cqt = cqt[ :-1 ]
        self.cqt = cqt
        
        
        dataDict = { 'cqt' : self.cqt }
        pickle.dump( dataDict, open( pickledDataPath, "wb" ) )
        #self.fftData /= np.std( self.fftData )
       # self.fftData = np.log10( self.fftData + 1.0 )
        #for winIdx in range( self.fftData.shape[ 0 ] ):            
            #self.fftData[ winIdx ] -= np.mean( self.fftData[ winIdx ] )
            #self.fftData[ winIdx ] /= np.std( self.fftData[ winIdx ] )
#       fftInput = np.resize(npsamples,(self.numFrames * 1024))
#        lrFft = np.absolute(librosa.core.stft(fftInput, n_fft=2048, hop_length=1024))
#        lrFft = lrFft[1:]
#        lrFft = np.swapaxes(lrFft, 0, 1)
#        lrFft = lrFft + 1
#        lrFft = np.log10(lrFft) / np.log10(2.5)
#        maxFft = np.amax( lrFft )
#        lrFft = lrFft / maxFft
#        self.fftData = np.resize(lrFft, ( self.numFrames, lrFft.shape[ 1 ] ) )
    
    def getBlock( self, firstIndex, lastIndex ):
        dataList = self.cqt
        assert firstIndex >= 0 and lastIndex <= dataList.shape[ 0 ], "Failed: 0<={}, {}<={}".format( firstIndex, lastIndex, dataList.shape[ 0 ] )
        return dataList[ firstIndex : lastIndex ]
    
    def getAudioTimeBatches( self, i=0, timeSteps=10, batchSize=40, overlap=False ):
        dataList = self.cqt
        frameCount = self.cqt.shape[ 0 ]
        if( batchSize == 0 ):
            i = 0
            batchSize = frameCount
        assert( timeSteps <= batchSize )
        
        # Construct audio list, assuming the audio list is the same length as the video list (as it should be)
        firstOutputSampleIdx = i * batchSize
        firstRequiredSampleIdx = firstOutputSampleIdx - ( max( timeSteps, 1 ) - 1 )
        
        lastSampleIdx = ( firstOutputSampleIdx + batchSize ) if batchSize > 0 else frameCount 
        lastSampleIdx = min( lastSampleIdx, frameCount )
        batchSize = lastSampleIdx - firstOutputSampleIdx
        
        requiredDataArraySize = batchSize + ( max( timeSteps, 1 ) - 1 )
        totalNumRequiredSamples = ( lastSampleIdx - firstRequiredSampleIdx ) + 1
        audioData = self.getBlock( max( firstRequiredSampleIdx, 0 ), lastSampleIdx )
        
        if( timeSteps == 0 ):            
            return audioData
        
        featureLen = audioData.shape[ 1 ]
        requiredData = []
            
        if( firstRequiredSampleIdx < 0 ):
            requiredData = np.array( [ audioData[ i ] if i >=0 else np.zeros( featureLen ) for i in range( firstRequiredSampleIdx, lastSampleIdx ) ] )
        else:
            requiredData = audioData
        # Now create the overlapping windows
        outputData = np.zeros( ( batchSize, timeSteps, featureLen ) )
        for i in range( batchSize ):
            outputData[ i, : ] = requiredData[ i : i + timeSteps ]
        
        return outputData

class DataHandler:
    dataPath = r"./data/"
    resolution = imageRes

    audioLoader = None
    
    sourceDownloaded = False
    framesExtracted = False
    audioTrackExtracted = False
    
    ytStream = None
    datasetPath = None
    videoFramesPath = None
    audioTrackPath = None
    frameFileList = []
    frameCount = 0    
    title = ""
    
    def __init__( self, url ):
        dataDir = os.path.dirname( self.dataPath );
        if not os.path.exists( dataDir ):
            os.makedirs( dataDir )
        
        yt = YouTube( url )
        self.title = yt.title
        print( "Found YouTube video " + self.title )
        ytStreamList =  yt.streams.filter( progressive=True, res="360p" )
        if( len( ytStreamList ) < 1 ):
            print( "No valid video found" )
            return
        self.ytStream = ytStreamList.first()
        assert( self.ytStream )
        
        self.datasetPath = os.path.join( self.dataPath, yt.video_id )
        if not os.path.exists( self.datasetPath ):
            os.makedirs( self.datasetPath )
        self.sourceVideoPath = os.path.join( self.datasetPath, self.ytStream.default_filename )
        self.audioTrackPath = os.path.join( self.datasetPath, "audio.wav" )
        self.videoFramesPath = os.path.join( self.datasetPath, "frames" )
        self.tfRecordPath = os.path.join( self.datasetPath, "data.tfrecord")
        self.metadataPath = os.path.join( self.datasetPath, "metadata.pkl" )
        self.downloadSource()
                
    def downloadSource( self ):
        # Check if the video already exists
        if not os.path.isfile( self.sourceVideoPath ):
            print( "Downloading video " + self.title )
            self.ytStream.download( self.datasetPath )
        else:
            print( "Video " + self.title + " already downloaded" )
        self.sourceDownloaded = True
        
    def deleteSource( self ):
        if os.path.isfile( self.sourceVideoPath ):
            os.unlink( self.sourceVideoPath )
        self.sourceDownloaded = False
    
    def extractData( self ):
        print( "Extracting data for " + self.title )
        self.downloadSource()
        self.extractVideoFrames()
        self.extractAudioTrack()
        self.audioLoader = AudioLoader( self.audioTrackPath, self.frameCount )
        self.populateFrameContext( True )
    
    def populateFrameContext( self, keepFrames=False ):
        if not os.path.isfile( self.metadataPath ):
            self.downloadSource()
            self.extractVideoFrames()
            if not keepFrames:
                self.deleteVideoFrames()
            metadataDict = { 'framecount' : self.frameCount }
            pickle.dump( metadataDict, open( self.metadataPath, "wb" ) )
        else:
            metadataDict = pickle.load( open( self.metadataPath,'rb' ) )           
            self.frameCount = metadataDict[ 'framecount' ]
            self.frameFileList = [ os.path.join( self.videoFramesPath, "{0:05d}.bmp".format( i ) ) for i in range( 1, self.frameCount + 1 ) ]
        
    def clearData( self ):
        print( "Clearing data for " + self.title )
        self.deleteVideoFrames()
        self.deleteAudioTrack()
        self.audioLoader = None
        
    def extractAudioTrack( self ):
        if not os.path.isfile( self.audioTrackPath ):
            command = 'ffmpeg -i "%s" -vn -acodec pcm_s16le -ac 1 -ar 44100 -vn %s' % ( self.sourceVideoPath, self.audioTrackPath )
            # TODO - verify ffmpeg exit status
            subprocess.call( command, shell=True )
        audioTrackExtracted = True
        
    def deleteAudioTrack( self ):
        if os.path.isfile( self.audioTrackPath ):
            os.unlink( self.audioTrackPath )
        audioTrackExtracted = False
    
    def extractVideoFrames( self ):
        if not os.path.exists( self.videoFramesPath ):
            os.makedirs( self.videoFramesPath )
            outputImagePattern = os.path.join( self.videoFramesPath, "%05d.bmp" )
            command = 'ffmpeg -i "%s" -r 25 -vf scale=%i:%i "%s"' % \
                ( self.sourceVideoPath, self.resolution, self.resolution, outputImagePattern )
            # TODO - confirm this returned properly
            subprocess.call( command, shell=True )
            
        self.frameFileList = sorted( glob.glob( os.path.join( self.videoFramesPath, "*.bmp" ) ) )
        self.frameCount = len( self.frameFileList )
        self.framesExtracted = True
    
    def deleteVideoFrames( self ):
        if os.path.exists( self.videoFramesPath ):
            shutil.rmtree( self.videoFramesPath )
        self.framesExtracted = False
    
    def getBlockCount( self, blockSize ):
        if( blockSize == 0 ):
            return 1
        offset = 0
        if( self.frameCount % blockSize != 0 ):
            offset = 1
        return ( self.frameCount // blockSize ) + offset
    
    def getTimeBlockCount( self, blockSize, timeSteps ):
        return ( self.frameCount // blockSize ) - 1
    
    def getVideoTimeBatches( self, i=0, timeSteps=10, batchSize=40, overlap=False ):
        if( batchSize == 0 ):
            i = 0
            batchSize = self.frameCount
        assert( timeSteps <= batchSize )
        
        # Construct audio list, assuming the audio list is the same length as the video list (as it should be)
        firstOutputSampleIdx = i * batchSize
        firstRequiredSampleIdx = firstOutputSampleIdx - ( max( timeSteps, 1 ) - 1 )
                
        lastSampleIdx = ( firstOutputSampleIdx + batchSize ) if batchSize > 0 else self.frameCount
        lastSampleIdx = min( lastSampleIdx, self.frameCount )
        batchSize = lastSampleIdx - firstOutputSampleIdx
        
        requiredDataArraySize = batchSize + ( max( timeSteps, 1 ) - 1 )
        totalNumRequiredSamples = ( lastSampleIdx - firstRequiredSampleIdx ) + 1
        
        requiredFilesList = [ ( self.frameFileList[ i ] if i >=0 else "" ) for i in range( firstRequiredSampleIdx, lastSampleIdx ) ]
        
        requiredImages = np.array( [ ( np.array( Image.open( fname ) ) if fname else np.zeros( ( imageRes, imageRes, 3 ) ) ) for fname in requiredFilesList ] )
        requiredImages = requiredImages.astype( 'float32' ) / 255.0
        if( timeSteps == 0 ):            
            return requiredImages
        
        outputData = np.zeros( ( batchSize, timeSteps, imageRes, imageRes, 3 ) )
        for i in range( batchSize ):
            outputData[ i, : ] = requiredImages[ i : i + timeSteps ]
        return outputData
        
    def getAudioTimeBatches( self, i=0, timeSteps=10, batchSize=40, overlap=False ):
        if( batchSize == 0 ):
            i = 0
            batchSize = self.frameCount
        assert( timeSteps <= batchSize )
        
        # Construct audio list, assuming the audio list is the same length as the video list (as it should be)
        firstOutputSampleIdx = i * batchSize
        firstRequiredSampleIdx = firstOutputSampleIdx - ( max( timeSteps, 1 ) - 1 )
        
        lastSampleIdx = ( firstOutputSampleIdx + batchSize ) if batchSize > 0 else self.frameCount 
        lastSampleIdx = min( lastSampleIdx, self.frameCount )
        batchSize = lastSampleIdx - firstOutputSampleIdx
        
        requiredDataArraySize = batchSize + ( max( timeSteps, 1 ) - 1 )
        totalNumRequiredSamples = ( lastSampleIdx - firstRequiredSampleIdx ) + 1
        audioData = self.audioLoader.getBlock( max( firstRequiredSampleIdx, 0 ), lastSampleIdx )
        
        if( timeSteps == 0 ):            
            return audioData
        
        featureLen = audioData.shape[ 1 ]
        requiredData = []
            
        if( firstRequiredSampleIdx < 0 ):
            requiredData = np.array( [ audioData[ i ] if i >=0 else np.zeros( featureLen ) for i in range( firstRequiredSampleIdx, lastSampleIdx ) ] )
        else:
            requiredData = audioData
        # Now create the overlapping windows
        outputData = np.zeros( ( batchSize, timeSteps, featureLen ) )
        for i in range( batchSize ):
            outputData[ i, : ] = requiredData[ i : i + timeSteps ]
        
        return outputData
    
    def getAudioVideoData( self, i=0, audioTimesteps=10, videoTimesteps=10, batchSize=40 ):
        # If timesteps==0, return a singleton, else return a list
        assert( ( audioTimesteps >= 0 ) and ( videoTimesteps >= 0 ) and ( batchSize >= 0 ) and ( i >= 0 ) )
        
        audioData = self.getAudioTimeBatches( i, timeSteps=audioTimesteps, batchSize=batchSize )
        videoData = self.getVideoTimeBatches( i=i, timeSteps=videoTimesteps, batchSize=batchSize )
        assert( audioData.shape[ 0 ] == videoData.shape[ 0 ] )
        return audioData, videoData
            
    def getBlock( self, i=0, blockSize=40 ):
        assert( blockSize >= 0 )
        fileSlice = []
        if( blockSize > 0 ):
            index = i * blockSize
            assert( index < self.frameCount - 1 )
            lastIndex = min( index + blockSize, self.frameCount )
            
            fileSlice = self.frameFileList[ index : lastIndex ]
        else:
            fileSlice = self.frameFileList
            
        images = np.array( [ np.array( Image.open( fname ) ) for fname in fileSlice ] )
        images = images.astype( 'float32' ) / 255.0
        
        return images


class PlaylistHandler:
    dataHandlerList = []
    playlist = None
    allFileList = []
    def __init__( self, url ):
        for i in range( 3 ):
            self.playlist = Playlist( url )
            self.videoUrls = self.playlist.video_urls
            if( len( self.videoUrls ) ):
                break
            time.sleep( 1 )
        
        assert len( self.videoUrls ), "Could not load playlist"
        print( "Loading playlist with %i videos" % len( self.videoUrls ) )
        for videoUrl in self.videoUrls:
            self.dataHandlerList.append( DataHandler( videoUrl ) )
            self.dataHandlerList[ -1 ].extractData()
            
    
    def loadFileList( self ):
        self.allFileList = []
        for dataHandler in self.dataHandlerList:
            dataHandler.extractData()
            self.allFileList.extend( dataHandler.frameFileList )
            
        random.shuffle( self.allFileList )
        
    def numVideoBatches( self, batchSize ):
        return len( self.allFileList ) // batchSize
    
    def loadVideoBatch( self, idx, batchSize ):
        startIdx = idx * batchSize
        lastIdx = min( len( self.allFileList ), startIdx + batchSize )
        
        print( "From {} to {}".format( startIdx, lastIdx ) )
        fileSlice = self.allFileList[ startIdx : lastIdx ]
        requiredImages = np.array( [ ( np.array( Image.open( fname ) ) if fname else np.zeros( ( imageRes, imageRes, 3 ) ) ) for fname in fileSlice ] )
        requiredImages = requiredImages.astype( 'float32' ) / 255.0
        return requiredImages
            