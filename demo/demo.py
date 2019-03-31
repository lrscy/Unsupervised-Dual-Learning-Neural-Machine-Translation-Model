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
import multiprocessing
import h5py
import tqdm
import json
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

Read paralelled train data from files for each language and save to a dictionary.

Args:
    dataPath: file path of train data. Default value is "../../Data/train/".
    langList: language list. Indicating which language data will be included in train data.
              Default value is ["Chinese", "English"].
    encoding: encoding of each file in train directory.
    ratio: propotion of train data, others will be treat as dev data. Default value is 0.98.

Returns:
    trainData: a dictionary of train data sentences of each language. Its structure is:
    
               {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
    devData: a dictionary of dev data sentences of each language. Its structure is:
    
               {language A: [[word1, word2, ...], [...], ...], language B: ...}.
"""
def getTrainData( dataPath = "../../Data/train/", lanList = ["chinese", "english"],
                  encoding = "UTF-8", ratio = 0.98 ):
    trainData = {}
    devData   = {}
    data = {}
    for lan in lanList:
        if lan not in data:
            data[lan] = []
        print( "Reading " + lan + " files..." )
        files = os.listdir( dataPath + lan + "/" )
        for file in files:
            with open( dataPath + lan + "/" + file, encoding = encoding ) as f:
                line = f.readline()
                while line:
                    wordList = line.split()
                    data[lan].append( ["<S>"] + wordList + ["</S>"] )
                    line = f.readline()
    noOfSentences = len( data[lanList[0]] )
    arr = [i for i in range( noOfSentences )]
    random.shuffle( arr )
    for lan in lanList:
        if lan not in trainData:
            trainData[lan] = []
        if lan not in devData:
            devData[lan] = []
        data[lan] = np.array( data[lan] )[arr].tolist()
        noOfTrainData = int( noOfSentences * ratio )
        devData[lan]   = data[lan][noOfTrainData:]
        trainData[lan] = data[lan][:noOfTrainData]
        print( trainData[lan][:5] )
    return trainData, devData

""" Generate dictionary and preprocess setences for each language

Generate dictionary for each language and convert word to corresponding index.
Here we set two dictionaries to speed up the whole program.

Moreover, the function will replace word in sentences to index automatically.

Args:
    data: a dictionary contains sentences of each language.  Its structure is:
    
          {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
    threshold: a word will be replace with <UNK> if frequency of a word is
               less than threshold. If the value is less than 1, it means
               no need to replace any word to <UNK>. Default value is 0.

Returns:
    wordNumDict: a dictionary which can convert words to index in each language.
                 Its structure is:
                 
                 {language A: {word A: index A, word B: ..., ...}, language B: ..., ...}.
    
    numWordDict: a dictionary which can convert index to word in each language.
                 Its structure is:
                 
                 {language A: {index A: word A, index B: ..., ...}, language B: ..., ...}.
"""
def generateDict( data, threshold = 0 ):
    wordNumDict = {}
    numWordDict = {}
    for lan, sentences in data.items():
        if lan not in wordNumDict:
            # Add special word to dictionary
            wordNumDict[lan] = {"<PAD>": 0, "<S>": 1, "</S>": 2, "<UNK>": 3}
        if lan not in numWordDict:
            # Add special word to dictionary
            numWordDict[lan] = {0: "<PAD>", 1: "<S>", 2: "</S>", 3: "<UNK>"}
        
        # Count word frequency
        wordCount = {}
        for sentence in sentences:
            for word in sentence:
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

"""Save dictionary to file

Args:
    dt: the dictionary to save.
    fileName: the name of file to save.
    path: the path of file to save.

Returns:
    None.
"""
def saveDict( dt, fileName, path ):
    print( "Saving to " + path + fileName + "..." )
    fileName = path + fileName
    with open( fileName, "w" ) as f:
        jsn = json.dumps( dt )
        f.write( jsn )

"""Sort dictionary by length of original language

Args:
    data: a dictionary contains sentences of each language.  Its structure is:
    
          {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
    lan: a list of language. The first one is the original language, the second
         one is the target language. For example:

         [language A, language B].

Returns:
    None.
"""
def sortByOriLan( data, lan = ["chinese", "english"] ):
    print( "Sorting..." )
    tmp = list( zip( data[lan[0]], data[lan[1]] ) )
    tmp.sort( key = lambda x: len( x[0] ) )
    data[lan[0]], data[lan[1]] = zip( *tmp )

"""Number to One-hot

Only convert sentences which length are small than 30.

Args:
    data: a dictionary contains sentences of each language.  Its structure is:
    
          {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
    lan: a list of language. The first one is the original language, the second
         one is the target language. For example:

         [language A, language B].

    dictLength: dictionary length of language B.

Returns:
    status: boolean value. True represents successfully extract batch data. False
            represents extract nothing from original data.
    ndata: data like dictionary.
    lan1d: 3-D data contains one-hot format label data.
"""
def toCategory( data, lan, dictLength ):
    maxlLan0 = 0
    maxlLan1 = 0
    n = 0
    for i in range( len( data[lan[0]] ) ):
        if len( data[lan[0]][i] ) <= 32:
            n += 1
            maxlLan0 = max( maxlLan0, len( data[lan[0]][i] ) )
            maxlLan1 = max( maxlLan1, len( data[lan[1]][i] ) )
    if n == 0:
        return False, [], []
    lan0 = np.zeros( ( n, maxlLan0 ) )
    lan1 = np.zeros( ( n, maxlLan1 ) )
#    lan1d = np.zeros( ( n, maxlLan1, dictLength ) )
    n = 0
    for i in range( len( data[lan[0]] ) ):
        if len( data[lan[0]][i] ) <= 32:
            for j in range( len( data[lan[0]][i] ) ):
                lan0[n, j] = data[lan[0]][i][j]
            for j in range( len( data[lan[1]][i] ) ):
                lan1[n, j] = data[lan[1]][i][j]
#                if j:
#                    w = data[lan[1]][i][j]
#                    lan1d[n, j - 1, w] = 1
            n += 1
    data[lan[0]] = lan0
    data[lan[1]] = lan1
    return True, data#, lan1d

def toCategoryWrap( args ):
    return toCategory( *args )

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
def simpleSeq2Seq( output_vocab_size, input_vocab_size, hidden_dim = 128,
                   word_vec_dim = 300, name = "demo" ):
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

trainData, devData = getTrainData( "../../Data/train/" )
wordNumDict, numWordDict = generateDict( trainData, threshold = 5 )
saveDict( wordNumDict, "wordNumDict", "Dicts" )
saveDict( numWordDict, "numWordDict", "Dicts" )
sortByOriLan( trainData, ["chinese", "english"] )
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
length = len( wordNumDict["english"] )

print( "parallizly processing training data" )
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool( processes = cores )
params = []
for i in range( 0, total + batch_size, batch_size ):
    # Divide data into batch
    tdata = {}
    tdata["chinese"] = trainData["chinese"][i:i + batch_size]
    tdata["english"] = trainData["english"][i:i + batch_size]
    # Combine all params
    params.append( [tdata, ["chinese", "english"], length] )
print( "MAP" )
rets = []
for ret in tqdm.tqdm( pool.imap_unordered( toCategoryWrap, params ) ):
    rets.append( ret )
#rets = pool.map( toCategoryWrap, params )
pool.close()
pool.join()

n = 0
print( "Training..." )
for epoch in range( 2 ):
    for ret in rets:
    #    status, newTrainData, td = toCategory( trainData, ["chinese", "english"], length, i, min( i + batch_size, total ) )
        if ret[0] == False:
            continue
        label = K.utils.to_categorical( ret[1]["english"], length )
        loss = model.train_on_batch( [ret[1]["chinese"], ret[1]["english"]], label )
        n += 1
        print( n, loss )
        if n and n % 3000 == 0:
            model.save_weights("Models/model_weights_" + str( n ) + ".h5" ) 
        losses.append( loss )
    model.save_weights("Models/model_weights_" + str( n ) + ".h5" ) 
with open( "losses", "w" ) as f:
    for loss in losses:
        f.write( str( loss ) + "\n" )
