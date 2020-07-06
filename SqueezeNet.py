#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 18:00:04 2020

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


class FireModule( Layer ):
    fireIndex = -1
    s_1x1 = -1
    e_1x1 = -1
    e_3x3 = -1
    
    def __init__( self, s_1x1, e_1x1, e_3x3, fireIndex ):
        super( FireModule, self ).__init__( name="FireModule-{}".format( fireIndex ) )
        self.fireIndex = fireIndex
        self.s_1x1 = s_1x1
        self.e_1x1 = e_1x1
        self.e_3x3 = e_3x3
        
        activation = "relu"
        self.s_1x1Conv = Conv2D( self.s_1x1, 1, padding='valid', activation=activation, name="FireModule_{}_s1x1_Conv2D".format( self.fireIndex ) )
        self.e_1x1Conv = Conv2D( self.e_1x1, 1, padding='valid', activation=activation, name="FireModule_{}_e1x1_Conv2D".format( self.fireIndex ) )
        
        self.e_3x3Conv = Conv2D( self.e_3x3, 3, padding='same', activation=activation, name="FireModule_{}_e3x3_Conv2D".format( self.fireIndex ) )
        
        self.concatOutput = Concatenate( axis=-1, name="FireModule_{}_concatenate".format( self.fireIndex ) )
        
    def call( self, inputObj ):
        x = inputObj
        squeezeOutput = self.s_1x1Conv( x )
        
        e_1x1_Output = self.e_1x1Conv( squeezeOutput )
        e_3x3_Output = self.e_3x3Conv( squeezeOutput )
        
        output = self.concatOutput( [ e_1x1_Output, e_3x3_Output ] )
        
        return output
    
    def compute_output_shape( self, inputShape ):
        #input shape = [None, 299, 299, 3]
        # Input shape hopefully has 4 dimensions
        numOutputFilters = self.e_1x1 + self.e_3x3
        return [ inputShape[ 1 ], inputShape[ 5 ], numOutputFilters ]
    
    
class TransposeFireModule( Layer ):
    fireIndex = -1
    s_1x1 = -1
    e_1x1 = -1
    e_3x3 = -1
    
    def __init__( self, s_1x1, e_1x1, e_3x3, outputFilters, fireIndex ):
        super( TransposeFireModule, self ).__init__( name="TransposeFireModule-{}".format( fireIndex ) )
        self.fireIndex = fireIndex
        self.s_1x1 = s_1x1
        self.e_1x1 = e_1x1
        self.e_3x3 = e_3x3
        self.outputFilters = outputFilters
        
        activation = "relu"
        #self.split_e_1x1 = Lambda( lambda x: x[ :, :, 0:e_1x1 ] )
        #self.split_e_3x3 = Lambda( lambda x: x[ :, :, e_1x1: ] )
        
        self.e_3x3DeConv = Conv2DTranspose( self.s_1x1, 3, padding='same', activation=activation, name="FireModule_{}_e3x3_Conv2D".format( self.fireIndex ) )        
        self.e_1x1DeConv = Conv2DTranspose( self.s_1x1, 1, padding='same', activation=activation, name="FireModule_{}_e3x3_Conv2D".format( self.fireIndex ) )        
        
        self.concatExpand = Concatenate( axis=-1 )
        self.s_1x1DeConv = Conv2D( outputFilters, 1, padding='valid', activation=activation, name="FireModule_{}_s1x1_Conv2D".format( self.fireIndex ) )
       
        
    def call( self, inputObj ):
        x = inputObj
        
#        e_1x1_in = self.split_e_1x1( x )
#        e_3x3_in = self.split_e_3x3( x )
        
        e_1x1_Output = self.e_1x1DeConv( x )
        e_3x3_Output = self.e_3x3DeConv( x )
        
        s_1x1_in = self.concatExpand( [ e_1x1_Output, e_3x3_Output ] )
        
        output = self.s_1x1DeConv( s_1x1_in )
        
        return output
    
    def compute_output_shape( self, inputShape ):
        #input shape = [None, 299, 299, 3]
        # Input shape hopefully has 4 dimensions
        numOutputFilters = self.e_1x1 + self.e_3x3
        return [ inputShape[ 1 ], inputShape[ 2 ], outputFilters ]

class SqueezeNetEncoder( Layer ):
    def __init__( self, latentSize=1024 ):
        super( SqueezeNetEncoder, self ).__init__( name="Encoder" )
        self.latentSize = latentSize
        activation = 'relu'
        self.conv1 = Conv2D( 96, 3, strides=(2,2), padding="valid", activation=activation )
        self.maxpool1 = MaxPool2D( (3,3), strides=(2,2), padding="valid" )
        
        self.fire2 = FireModule( 16, 64, 64, 2 )
        self.fire3 = FireModule( 16, 64, 64, 3 )
        self.fire4 = FireModule( 32, 128, 128, 4 )
        
        self.maxpool4 = MaxPool2D( (3,3), strides=(2,2), padding="valid" )
        
        self.fire5 = FireModule( 32, 128, 128, 5 )
        self.fire6 = FireModule( 48, 192, 192, 6 )
        self.fire7 = FireModule( 48, 192, 192, 7 )
        self.fire8 = FireModule( 64, 256, 256, 8 )
        
        self.maxpool8 = MaxPool2D( (3,3), strides=(2,2), padding="valid" )
        
        self.fire9 = FireModule( 64, 256, 256, 9 )
        self.dropout9 = Dropout( 0.5 )
        
        self.conv10 = Conv2D( latentSize, 1, strides=(1,1), padding="same", activation=activation )
        self.avgPool10 = AveragePooling2D( (13,13), strides=(1,1), padding="valid" )
        self.flatten10 = Flatten()
        
        self.dense11 = Dense( latentSize, activation='sigmoid' )
        

    def call( self, encoderInput ):
        x = encoderInput
        x = self.conv1( x )
        x = self.maxpool1( x )
        
        x = self.fire2( x )
        x = self.fire3( x )
        x = self.fire4( x )
        x = self.maxpool4( x )
        
        x = self.fire5( x )
        x = self.fire6( x )
        x = self.fire7( x )
        x = self.fire8( x )
        x = self.maxpool8( x )
        
        x = self.fire9( x )
        x = self.dropout9( x )
        
        x = self.conv10( x )
        x = self.avgPool10( x )
        x = self.flatten10( x )
        
        x = self.dense11( x )
        return x
    
    def compute_output_shape( self, inputShape ):
        return [ self.latentSize ]
    
    
class SqueezeNetDecoder( Layer ):
    latentSize = -1
    def __init__( self, latentSize=1024 ):
        super( SqueezeNetDecoder, self ).__init__( name="Decoder" )
        self.latentSize = latentSize
        #input is (1024,) or whatever latent size is
        
        self.avgPoolDense = Dense( 7 * 7 * latentSize )
        self.avgPoolReshape = Reshape( (7,7,latentSize) )
        self.avgPoolUpsample = UpSampling2D( size=(2,2) )
        self.deconv10 = Conv2DTranspose( 512, 1, strides=(1,1), padding='same', activation='relu')
        
        self.dropout9 = Dropout( 0.5 )
        self.fire9Transpose = TransposeFireModule( 64, 256, 256, 512, 9 )
        
        # Want 14
        self.upsample8 = UpSampling2D( size=(2, 2) ) # Out - 28
        self.fire8Transpose = TransposeFireModule( 64, 256, 256, 384, 8 )
        self.fire7Transpose = TransposeFireModule( 48, 192, 192, 384, 7 )
        self.fire6Transpose = TransposeFireModule( 48, 192, 192, 256, 6 )
        self.fire5Transpose = TransposeFireModule( 32, 128, 128, 256, 5 )
        
        # Want 28
        self.upsample4 = UpSampling2D( size=(2, 2) ) # Out - 56
        self.fire4Transpose = TransposeFireModule( 32, 128, 128, 128, 4 )
        self.fire3Transpose = TransposeFireModule( 16, 64, 64, 128, 3 )
        self.fire2Transpose = TransposeFireModule( 16, 64, 64, 96, 2 )
        # Want 56
        self.upsample1 = UpSampling2D( size=(2, 2) ) # Out -112
        self.deconv1 = Conv2DTranspose( 3, 7, strides=(2,2), padding='same', activation='sigmoid' )
        # Want 224 out
        
    def call( self, decoderIn ):
        x = decoderIn
        
        x = self.avgPoolDense( x )
        
        x = self.avgPoolReshape( x )
        x = self.avgPoolUpsample( x )
        x = self.deconv10( x )
        
        x = self.dropout9( x )
        x = self.fire9Transpose( x )
        
        x = self.upsample8( x )
        x = self.fire8Transpose( x )
        x = self.fire7Transpose( x )
        x = self.fire6Transpose( x )
        x = self.fire5Transpose( x )
        
        x = self.upsample4( x )
        x = self.fire4Transpose( x )
        x = self.fire3Transpose( x )
        x = self.fire2Transpose( x )
        
        x = self.upsample1( x )
        x = self.deconv1( x )
                                        
        return x
    
    def compute_output_shape( self, input_shape ):
        #input shape = [None, 299, 299, 3]
        return [ 224, 224, 3 ]
    
    
class SqueezeNetAutoencoder( Model ):
    def __init__( self ):
        super( SqueezeNetAutoencoder, self ).__init__()
        self.encoder = SqueezeNetEncoder()
        self.decoder = SqueezeNetDecoder()
    
    def call( self, x ):
        x = self.encoder( x )
        x = self.decoder( x )
        return x
    
