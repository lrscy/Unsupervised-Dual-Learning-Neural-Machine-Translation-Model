# coding: utf-8

""" Abstract

The demo is a prototype of the project model. Codes and structure here could change in the future.
The main point of codes below is to run on local computer and test whether it works on small scale of data.

Now:
    Prove that original model is inaccessible. Start to implemente original RNN-LSTM and treat it as baseline.

TODO:
    1. Try new way to approach the unsupervised method.
    2. Try to understand how to control gradient in Tensorflow.
"""

import os
import sys
import random
import math
import h5py
import numpy as np
import tensorflow as tf
import keras as K
import keras.backend.tensorflow_backend as KTF
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, Lambda, concatenate, Multiply

K.backend.clear_session()
KTF.set_session( tf.Session( config = tf.ConfigProto( device_count = {'gpu':0} ) ) )

""" Get train data from path

Read train data from files for each language and save to a dictionary.

Args:
    dataPath: file path of train data. Default value is "../../Data/train/".
    langList: language list. Indicating which language data will be included in train data.
              Default value is ["Chinese", "English"].
    encoding: encoding of each file in train directory.
    ratio: propotion of train data, others will be treat as dev data. Default value is 0.98.
    sort: boolean value. If shuffle equals True, all data will be sorted according to their
          length from short to long. Otherwise, train sentences will be shuffled at the end.
          Default value is True.

Returns:
    trainData: a dictionary of train data sentences of each language. Its structure is:
    
               {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
    devData: a dictionary of dev data sentences of each language. Its structure is:
    
               {language A: [[word1, word2, ...], [...], ...], language B: ...}.
"""
def getTrainData( dataPath = "../Data/train/", lanList = ["chinese", "english"],
                  encoding = "UTF-8", ratio = 0.98, sort = True ):
    trainData = {}
    devData   = {}
    for lan in lanList:
        print( lan, end = " " )
        if lan not in trainData:
            trainData[lan] = []
        if lan not in devData:
            devData[lan] = []
        files = os.listdir( dataPath + lan + "/" )
        data = []
        for file in files:
            with open( dataPath + lan + "/" + file, encoding = encoding ) as f:
                line = f.readline()
                while line:
                    wordList = line.split()
                    data.append( ["<S>"] + wordList + ["</S>"] )
                    line = f.readline()
        # suffle here is to make sure that all data are random distributed
        random.shuffle( data )
        noOfSentences = len( data )
        print( noOfSentences, end = " " )
        noOfTrainData = int( noOfSentences * ratio )
        devData[lan]   = data[noOfTrainData:]
        trainData[lan] = data[:noOfTrainData]
        if sort == True:
            trainData[lan].sort( key = lambda x: len( x ) )
            devData[lan].sort( key = lambda x: len( x ) )
        print( len( trainData[lan] ), len( devData[lan] ) )
    return trainData, devData

""" Generate dictionary and preprocess setences for each language

Generate dictionary for each language and convert word to corresponding index.
Here we set two dictionaries to speed up the whole program.

Args:
    data: a dictionary contains sentences of each language.  Its structure is:
    
          {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
    threshold: a word will be replace with <UNK> if frequency of a word is
               less than threshold. If the value is less than 1, it means
               no need to replace any word to <UNK>. Default value is 0.

Returns:
    wordNumDict: a dictionary which can convert words to index in each language.
                 Its structure is:
                 
                 {language A: {word A: index, word B: ..., ...}, language B: ..., ...}.
    
    numWordDict: a dictionary which can convert index to word in each language.
                 Its structure is:
                 
                 {language A: {word A: index, word B: ..., ...}, language B: ..., ...}.
"""
def generateDict( data, threshold = 0 ):
    wordNumDict = {}
    numWordDict = {}
    for lan, sentences in data.items():
        wordCount = {}
        if lan not in wordNumDict:
            # Add special word to dictionary
            wordNumDict[lan] = {"<PAD>": 0, "<S>": 1, "</S>": 2, "<UNK>": 3}
        if lan not in numWordDict:
            # Add special word to dictionary
            numWordDict[lan] = {0: "<PAD>", 1: "<S>", 2: "</S>", 3: "<UNK>"}
        
        # Count word frequency
        for sentence in sentences:
            for i in range( len( sentence ) ):
                word = sentence[i]
                if word not in wordCount:
                    wordCount[word] = 0
                wordCount[word] += 1
        
        # Find and replace with <UNK>
        for sentence in sentences:
            for i in range( len( sentence ) ):
                word = sentence[i]
                if wordCount[word] < threshold:
                    word = "<UNK>"
                if word not in wordNumDict[lan]:
                    number = len( wordNumDict[lan] )
                    wordNumDict[lan][word] = number
                    numWordDict[lan][number] = word
                sentence[i] = wordNumDict[lan][word]
    return wordNumDict, numWordDict

"""Number to One-hot

Only convert sentences which length are small than 30.

Args:
    data: a dictionary contains sentences of each language.  Its structure is:
    
          {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
    wordNumDict: a dictionary which can convert words to index in each language.
                 Its structure is:
                 
                 {language A: {word A: index, word B: ..., ...}, language B: ..., ...}.

Returns:
    ndata: 
    td:
"""
def toCategory( data, wordNumDict, left, right ):
#     n = right - left
    maxlch = 0
    maxlen = 0
    n = 0
    for i in range( left, right ):
        if len( data["chinese"][i] ) <= 32:
            n += 1
            maxlch = np.max( [maxlch, len( data["chinese"][i] )] )
            maxlen = np.max( [maxlen, len( data["english"][i] )] )
    if n == 0:
        return False, [], []
    zh = np.zeros( ( n, maxlch ) )
    en = np.zeros( ( n, maxlen ) )
    td = np.zeros( ( n, maxlen, len( wordNumDict["english"] ) ) )
    n = 0
    for i in range( left, right ):
        if len( data["chinese"][i] ) <= 32:
            for j in range( len( data["chinese"][i] ) ):
                zh[n, j] = data["english"][i][j]
            for j in range( len( data["english"][i] ) ):
                en[n, j] = data["english"][i][j]
                if j:
                    w = data["english"][i][j]
                    td[n, j - 1, w] = 1
            n += 1
    ndata = {}
    ndata["chinese"] = zh
    ndata["english"] = en
    return True, ndata, td

""" Simple Seqence to Sequence Implementation

A simple implementation of Sequence to Sequence model. It works as baseline

Args:
    input_dim:  dimension of input word vector.
    output_dim: dimension of output word vector.
    hidden_dim: dimension of hidden states vector.
    output_vocab_size: size of output language vocabulary size.
    input_vocab_size:  size of input  language vocabulary size.
    word_vec_dim: dimension of word-vector.
    name: name of the model.

Returns:
    model: the whole model of simple Seq2Seq model.

"""
def simpleSeq2Seq( output_vocab_size, input_vocab_size, hidden_dim = 256,
                   word_vec_dim = 512, name = "demo" ):
    embedding_encoder  = Embedding( output_dim = word_vec_dim, input_dim = input_vocab_size,
                                 name = name + "_encoder_embedding", mask_zero = True ) # 
    embedding_decoder = Embedding( output_dim = word_vec_dim, input_dim = output_vocab_size,
                                 name = name + "_decoder_embedding", mask_zero = True ) # 
    # Encoder
    encoder_input     = Input( shape = ( None, ), name = name + "_encoder_input" )
    # change when using pre-trained embedding trainable= False
    encoder           = LSTM( hidden_dim, return_state = True )
    encoder_input_emb = embedding_encoder( encoder_input )
    _, state_h, state_c = encoder( encoder_input_emb )
    state_encoder     = [state_h, state_c]
    # Decoder
    decoder = LSTM( hidden_dim, return_sequences = True )

    decoder_input     = Input( shape = ( None, ), name = name + "_decoder_input" )
    decoder_input_emb = embedding_decoder( decoder_input )
    decoder_outputs   = decoder( decoder_input_emb, initial_state = state_encoder )
    decoder_dense     = Dense( output_vocab_size, activation = "softmax", name = name + "_decoder_output" )
    decoder_outputs   = decoder_dense( decoder_outputs )

    # Build model
    model = Model( inputs = [encoder_input, decoder_input], outputs = decoder_outputs, name = name )
    model.compile( optimizer = 'adam', loss = "categorical_crossentropy" )
    return model

trainData, devData = getTrainData( "../Data/train/" )
wordNumDict, numWordDict = generateDict( trainData, threshold = 5 )
ivs = len( wordNumDict["chinese"] )
ovs = len( wordNumDict["english"] )
print( ivs, ovs )

trainData["chinese"] = trainData["chinese"][::-1]
devData["chinese"] = devData["chinese"][::-1]
trainData["english"] = trainData["english"][::-1]
devData["english"] = devData["english"][::-1]

model = simpleSeq2Seq( output_vocab_size = ovs, input_vocab_size = ivs, name = "demo" )
model.summary()

batch_size = 64
losses = []
n = 0
total = len( trainData["chinese"] )
for i in range( 0, total + batch_size, batch_size ):
    status, newTrainData, td = toCategory( trainData, wordNumDict, i, min( i + batch_size, total ) )
    if status == False:
        continue
    loss = model.train_on_batch( [newTrainData["chinese"], newTrainData["english"]], td )
    n += 1
    print( n, loss )
    if n and n % 3000 == 0:
        model.save_weights("Models/model_weights_" + str( n ) + ".h5" ) 
    losses.append( loss )
model.save_weights("Models/model_weights_final" + str( n ) + ".h5" )
