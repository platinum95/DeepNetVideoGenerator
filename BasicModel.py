#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:44:01 2020

@author: platinum95
"""

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import datasets
from tensorflow.keras.initializers import glorot_uniform, Constant
from tensorflow.keras.layers import *
from tensorflow.keras.layers import PReLU, LeakyReLU, Conv2D, MaxPool2D, Lambda, Conv2DTranspose, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils import conv_utils


imageRes = 224

kernel_init = glorot_uniform()
bias_init = Constant( value=0.2 )

class InceptionModule( Layer ):
    def __init__( self, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None ):
        super( InceptionModule, self ).__init__( name="InceptionModule" )
        self.conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.conv_3x3_1 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.conv_3x3_2 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.conv_5x5_1 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.conv_5x5_2 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.pool_proj_1 = MaxPool2D((3, 3), strides=(1, 1), padding='same')
        self.pool_proj_2 = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.concat_1 = Concatenate( axis=-1, name=name )
    def call( self, x ):
    
        b0 = self.conv_1x1( x )

        b1 = self.conv_3x3_1( x )
        b1 = self.conv_3x3_2( b1 )

        b2 = self.conv_5x5_1( x )
        b2 = self.conv_5x5_2( b2 )

        b3 = self.pool_proj_1( x )
        b3 = self.pool_proj_2( b3 )

        x = self.concat_1( [ b0, b1, b2, b3 ] )

        return x

class FrameEncoder( Layer ):
    def __init__( self ):
        # Convs should give ( inX, inY, numFilters )
        # pools should divide by kernel size
        # Aiming for 4096 in dense layer
        # For input res 128, with 
        super( FrameEncoder, self ).__init__( name="Encoder" )
        
        self.conv1 = Conv2D( 32, 3, padding='same', activation='relu', input_shape=( imageRes, imageRes, 3 ) )
        self.maxPool1 = MaxPool2D( pool_size=( 2,2 ) )
        self.conv2 = Conv2D( 64, 4, padding='same', activation='relu' )
        self.maxPool2 = MaxPool2D( pool_size=( 2,2 ) )
        
        self.conv3 = Conv2D( 64, 5, padding='same', activation='relu' )
        self.maxPool3 = MaxPool2D( pool_size=( 2,2 ) )
        self.conv4 = Conv2D( 64, 6, padding='same', activation='relu' )
        self.maxPool4 = MaxPool2D( pool_size=( 2,2 ) )
      #  self.inceptionModule = InceptionModule( filters_1x1=192,
      #                                    filters_3x3_reduce=96,
      #                                    filters_3x3=208,
      #                                    filters_5x5_reduce=16,
      #                                    filters_5x5=48,
      #                                    filters_pool_proj=64,
      #                                    name='inception_4a' )

        # should be 8 x 8 x 64 here
        self.flatten = Flatten()
        self.d1 = Dense( 4096, activation='relu' )
        #self.d2 = Dense( 128, activation='sigmoid', use_bias=False )
        
    def call( self, encoderIn ):
        x = encoderIn
       # x = self.inceptionModule( x )
        
        x = self.conv1( x )
        x = self.maxPool1( x )
        x = self.conv2( x )
        x = self.maxPool2( x )
        
        x = self.conv3( x )
        x = self.maxPool3( x )
        
        x = self.conv4( x )
        x = self.maxPool4( x )
        
        x = self.flatten( x )
        x = self.d1( x )
        #x = self.d2( x )
        return x

        
    def compute_output_shape( self, input_shape ):
        #input shape = [None, 299, 299, 3]
        return [ None, 128 ]
    
class FrameDecoder( Layer ):
    def __init__( self ):
        super( FrameDecoder, self ).__init__( name="Decoder" )
        #self.d2 = Dense( 128, activation='relu' )
        self.d3 = Dense( 4096, activation='relu' )
        self.unflatten = Reshape( ( 8, 8, 64 ) )
        self.upSample1 = UpSampling2D( size=( 2, 2 ) )
        self.deconv1 = Conv2DTranspose( 32, 4, padding='same', activation='relu' )
        
        self.upSample2 = UpSampling2D( size=( 2, 2 ) )
        self.deconv2 = Conv2DTranspose( 3, 3, padding='same', activation='relu' )
        
        self.upSample3 = UpSampling2D( size=( 2, 2 ) )
        self.deconv3 = Conv2DTranspose( 3, 3, padding='same', activation='relu' )
        
        self.upSample4 = UpSampling2D( size=( 2, 2 ) )
        self.deconv4 = Conv2DTranspose( 3, 3, padding='same', activation='sigmoid', use_bias=False )
    
    def call( self, decoderIn ):
        x = decoderIn
        #x = self.d2( x )
        x = self.d3( x )
        x = self.unflatten( x )
        
        x = self.upSample1( x )
        x = self.deconv1( x )
        
        x = self.upSample2( x )
        x = self.deconv2( x )
        
        x = self.upSample3( x )
        x = self.deconv3( x )
        
        x = self.upSample4( x )
        x = self.deconv4( x )
        return x
    
    def compute_output_shape( self, input_shape ):
        #input shape = [None, 299, 299, 3]
        return [ 128, 128, 3 ]
      
        
class FrameAutoencoderModel( Model ):
    def __init__( self ):
        super( FrameAutoencoderModel, self ).__init__()
        self.encoder = FrameEncoder()
        #self.encoder = TimeDistributed( FrameEncoder() )
        #self.lstm1 = LSTM( 128 )
        self.decoder = FrameDecoder()
    
    def call( self, x ):
        x = self.encoder( x )
         #x = self.lstm1( x )
        x = self.decoder( x )
        return x