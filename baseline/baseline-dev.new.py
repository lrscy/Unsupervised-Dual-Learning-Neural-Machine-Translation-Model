#!/usr/bin/env python
# coding: utf-8

import os
import io
import sys
import copy
import random
import math
import multiprocessing
import h5py
import json
import time
import numpy as np

import numpy as np
import tensorflow as tf
import keras as K
import keras.backend.tensorflow_backend as KTF
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import *

#K.backend.clear_session()
#sess = tf.Session( config = tf.ConfigProto( device_count = {'gpu':0} ) )
#KTF.set_session( sess )

# Global debug parameter
EMB = True          # Open embedding layer or not
UNK = False         # Open UNK or not
MUTI_GPU = 1        # Train on multi GPU(1, 2, 3, ...)
CONTINUE = False    # Continuely train
TRAIN = True        # Train or not
GENERATION = False  # Generate or not
WORD2VEC = "self"   # self or bert

model_name = "baseline"
batch_size = 128
min_length = 5
max_length = 30
epoch      = 3000
loss_path  = "loss/"
dict_path  = "dicts/"
model_path = "models/"
generation_path = "generation/"
train_data_path = "../data/other/"
test_data_path  = "../data/other/"
word2vec_path   = "../data/w2v/"

def load_vectors( path, word2vec, language ):
    files = os.listdir( path + language + "/" )
    data = {}
    for file_name in files:
        fin = io.open( path + language + "/" + file_name, "r", encoding = "utf-8",
                       newline = "\n", errors = "ignore" )
#        n, d = map( int, fin.readline().split() )
        for line in fin:
            tokens = line.rstrip().split( ' ' )
            data[tokens[0]] = map( float, tokens[1:] )
    word2vec[language] = data

def load_word2vec( path, language_list = ["chinese", "english"] ):
    p = []
    manager = multiprocessing.Manager()
    word2vec = manager.dict()
    for language in language_list:
        # Only one file in each folder
        p_lan = multiprocessing.Process( target = load_vectors,
                                         args = ( path, word2vec, language ) )
        p.append( p_lan )
        p_lan.start()
    for p_lan in p:
        p_lan.join()
    word2vec = dict( word2vec )
    return word2vec

def load_language_data( path, language, encoding, lan_data = {} ):
    """Get data of one language from path

    Args:
        path: a string represents corpus path of each language.
        language: a string represents language.
        encoding: a string represents encoding of the language.
        lan_data: a dictionary contains sentences of each language. All changes
                  happens inside the dictionary. Its structure is:

                  {language A: [[word1, word2, ...], [...], ...],
                   language B: ...}
    """
    sentences = []
    files = os.listdir( path + language + "/" )
    for file_name in files:
        with open( path + language + "/" + file_name, "r", encoding = encoding ) as f:
            line = f.readline()
            while line:
                line = line.strip()
                if 0 == len( line ):
                    line = f.readline()
                    continue
                words = ["<START>"] + line.split() + ["<END>"]
                sentences.append( words )
                line = f.readline()
    lan_data[language] = sentences

def shuffle_list( x, shuf_order ):
    """Shuffle list with specific order

    Args:
        x: a list waiting to be shuffled.
        shuf_order: a list of distinct number represents the shuffle order.

    Returns:
        x: a list of shuffled data.
    """
    x = np.array( x )[shuf_order].tolist()
    return x

def load_data( path = "../data/", language_list = ["chinese", "english"],
               encoding_list = ["UTF-8", "UTF-8"], shuffle = True ):
    """Get data from path

    Args:
        path: a string represents corpus path of each language.
        language_list: a list of string represents languages.
        encoding_list: a list of string represents encoding of each language
                       corresponding to language list.
        shuffle: a boolean value. True for shuffle.

    Returns:
        lan_data: a dictionary contains sentences of each language.
                  Its structure is:

                  {language A: [[word1, word2, ...], [...], ...],
                   language B: ...}
    """
    assert len( language_list ) == len( encoding_list )
    p = []
    manager = multiprocessing.Manager()
    lan_data = manager.dict()
    # Read parallel corpus
    for lan, enc in zip( language_list, encoding_list ):
        # I don't know why I should pre-define it, but it works.
        # If I remove this line, some lines in data will be overwritten by unkown data.
        lan_data[lan] = {}
        print( "Reading " + lan + " language corpus..." )
        p_lan = multiprocessing.Process( target = load_language_data,
                                         args = ( path, lan, enc, lan_data ) )
        p.append( p_lan )
        p_lan.start()
    for p_lan in p:
        p_lan.join()
    lan_data = dict( lan_data )
    
    if shuffle == True:
        print( "Shuffling..." )
        
        # Decide shuffle order
        length = len( lan_data[language_list[0]] )
        shuf_order = [i for i in range( length )]
        random.shuffle( shuf_order )

        # Shuffle corpus
        pool = multiprocessing.Pool( processes = len( language_list ) )
        p = {}
        for language in language_list:
            p_lan = pool.apply_async( func = shuffle_list,
                                      args = ( lan_data[language], shuf_order ) )
            p[language] = p_lan
        pool.close()
        pool.join()
        for language in language_list:
            lan_data[language] = p[language].get()
    return lan_data

def build_language_dictionary( language, sentences,
                               word_to_idx_dict, idx_to_word_dict,
                               threshold = 0 ):
    """Build dictionary for one language

    Args:
        language: a string represents input sentences' language.
        sentence: a list contains sentences of one language.
                  Its structure is:

                  [[word1, word2, ...], [...], ...],

        word_to_idx_dict: a dictionary converts word to index. All changes about
                          mapping word to index happends here. Its structure is:

                          {language A: {word A: index A, word B: ..., ...},
                           language B: ...}.

        idx_to_word_dict: a dictionary converts index to word. All changes about
                          mapping index to word happends here. Its structure is:

                          {language A: {index A: word A, index B: ..., ...},
                           language B: ...}.

        threshold: a integer represents threshold. If the number of a word
                   is less than threshold, it will be replaced by <UNK>.
    """
    # Generate dictionary for each language
    ### TODO: special tokens ###
    word_to_idx_dict_lan = {"<PAD>": 0, "<S>": 1, "</S>": 2}
    idx_to_word_dict_lan = {0: "<PAD>", 1: "<S>", 2: "</S>"}
    if UNK:
        word_to_idx_dict_lan["<UNK>"] = 3
        idx_to_word_dict_lan[3] = "<UNK>"

    # Count words
    word_count = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1

    # Replace words to <UNK>
    for word, count in word_count.items():
        if count <= threshold:
            if not UNK:
                continue
            word = "<UNK>"
        if word not in word_to_idx_dict_lan:
            idx = len( word_to_idx_dict_lan )
            word_to_idx_dict_lan[word] = idx
            idx_to_word_dict_lan[idx] = word
   
    word_to_idx_dict[language] = word_to_idx_dict_lan
    idx_to_word_dict[language] = idx_to_word_dict_lan

def build_dictionary( language_data, threshold = 0 ):
    """Build dictionary for each language

    Args:
        language_data: a dictionary contains sentences of each language.
                       Its structure is:

                       {language A: [[word1, word2, ...], [...], ...],
                        language B: ...}

        threshold: a integer represents threshold. If the number of a word
                   is less than threshold, it will be replaced by <UNK>.

    Returns:
        word_to_idx_dict: a dictionary converts word to index. Its structure is:

                          {language A: {word A: index A, word B: ..., ...},
                           language B: ...}.

        idx_to_word_dict: a dictionary converts index to word. Its structure is:

                          {language A: {index A: word A, index B: ..., ...},
                           language B: ...}.
    """
    manager = multiprocessing.Manager()
    word_to_idx_dict = manager.dict()
    idx_to_word_dict = manager.dict()
    p = []
    for language, sentences in language_data.items():
        # Same as previous multiprocessing, I still don't know why.
        word_to_idx_dict[language] = {}
        idx_to_word_dict[language] = {}
        print( "Building " + language + " language dictionary..." )
        p_lan = multiprocessing.Process( target = build_language_dictionary,
                                         args = ( language, sentences,
                                                  word_to_idx_dict, idx_to_word_dict,
                                                  threshold ) )
        p.append( p_lan )
        p_lan.start()
    for p_lan in p:
        p_lan.join()
    word_to_idx_dict = dict( word_to_idx_dict )
    idx_to_word_dict = dict( idx_to_word_dict )
    return word_to_idx_dict, idx_to_word_dict    

def load_language_dictionary( language, word_to_idx_dict, idx_to_word_dict ):
    """Load dictionary for one language

    Args:
        word_to_idx_dict: a dictionary converts word to index. All changes about
                          mapping word to index happends here. Its structure is:

                          {language A: {word A: index A, word B: ..., ...},
                           language B: ...}.

        idx_to_word_dict: a dictionary converts index to word. All changes about
                          mapping index to word happends here. Its structure is:

                          {language A: {index A: word A, index B: ..., ...},
                           language B: ...}.
    """
    word_to_idx_dict_lan = {}
    idx_to_word_dict_lan = {}
    with open( "dicts/" + language + ".vab", encoding = "utf-8" ) as fp:
        tmp = json.load( fp )
    for idx, word in tmp.items():
        idx_to_word_dict_lan[int( idx )] = word
    for each in idx_to_word_dict_lan:
        word_to_idx_dict_lan.setdefault( idx_to_word_dict_lan[each], each )
    word_to_idx_dict[language] = word_to_idx_dict_lan
    idx_to_word_dict[language] = idx_to_word_dict_lan

def load_dictionary( language_list ):
    """Build dictionary for each language

    Args:
        language_data: a dictionary contains sentences of each language.
                       Its structure is:

                       {language A: [[word1, word2, ...], [...], ...],
                        language B: ...}

        threshold: a integer represents threshold. If the number of a word
                   is less than threshold, it will be replaced by <UNK>.

    Returns:
        word_to_idx_dict: a dictionary converts word to index. Its structure is:

                          {language A: {word A: index A, word B: ..., ...},
                           language B: ...}.

        idx_to_word_dict: a dictionary converts index to word. Its structure is:

                          {language A: {index A: word A, index B: ..., ...},
                           language B: ...}.
    """
    manager = multiprocessing.Manager()
    word_to_idx_dict = manager.dict()
    idx_to_word_dict = manager.dict()
    p = []
    for language in language_list:
        # Same as previous multiprocessing, I still don't know why.
        word_to_idx_dict[language] = {}
        idx_to_word_dict[language] = {}
        print( "Building " + language + " language dictionary..." )
        p_lan = multiprocessing.Process( target = load_language_dictionary,
                                         args = ( language,
                                                  word_to_idx_dict,
                                                  idx_to_word_dict ) )
        p.append( p_lan )
        p_lan.start()
    for p_lan in p:
        p_lan.join()
    word_to_idx_dict = dict( word_to_idx_dict )
    idx_to_word_dict = dict( idx_to_word_dict )
    return word_to_idx_dict, idx_to_word_dict    

def save_dict( dt, file_name, path = "dicts/" ):
    """Save dictionary to file

    Args:
        dt: the dictionary to save.
        file_name: the name of file to save.
        path: the path of file to save.
    """
    print( "Saving to " + path + file_name + "..." )
    file_name = path + file_name
    with open( file_name, "w" ) as f:
        jsn = json.dumps( dt )
        f.write( jsn )

def sort_by_ori_lan( data ):
    """Sort dictionary by length of original language

    Args:
        data: a dictionary contains sentences of each language.  Its structure is:

              {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    """
    print( "Sorting..." )
    lan = list( data.keys() )
    tmp = list( zip( data[lan[0]], data[lan[1]] ) )
    tmp.sort( key = lambda x: len( x[0] ) )
    data[lan[0]], data[lan[1]] = zip( *tmp )
    for language in lan:
        data[language] = list( map( list, data[language] ) )

def convert_to_index( data, word_to_idx_dict ):
    """Convert word to index

    Args:
        data: a dictionary contains sentences of each language.  Its structure is:

              {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
        word_to_idx_dict: a dictionary converts word to index. Its structure is:

                          {language A: {word A: index A, word B: ..., ...},
                           language B: ...}.
    """
    print( "Converting to one-hot..." )
    language_list = list( data.keys() )
    for language in language_list:
        for i in range( len( data[language] ) ):
            new_sentence = []
            for word in data[language][i]:
                if word not in word_to_idx_dict[language]:
                    if not UNK:
                        continue
                    word = "<UNK>"
                new_sentence.append( int( word_to_idx_dict[language][word] ) );
            data[language][i] = new_sentence

def convert_to_batch( data, language_list, batch_size = 64, min_length = 5, max_length = 32 ):
    """Pack individual data into batch

    Args:
        data: a dictionary contains sentences of each language.  Its structure is:

              {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
        language_list: a list of language, the first one is original language.
        batch_size: size of batch.
        min_length: minimum length for sentences in data set. Default is 5.
        max_length: maximum length for sentences in data set. Default is 32.
    """
    print( "Converting to batches..." )
    lan_list = language_list # Just for convenient
    data["label"] = copy.deepcopy( data[lan_list[-1]] )
    language_list.append( "label" )
    tdata = {}
    for lan in lan_list:
        if lan not in tdata:
            tdata[lan] = []
    for i in range( 0, len( data[lan_list[0]] ) + batch_size, batch_size ):
        lan = {}
        max_len = {}
        for language in lan_list:
            lan[language] = data[language][i: i + batch_size]
            max_len[language] = 0
        # Calculate max length of valid sentences
        n = 0
        for j in range( len( lan[lan_list[0]] ) ):
            length = len( lan[lan_list[0]][j] )
            if length < min_length or max_length < length:
                continue
            for language in lan_list:
                max_len[language] = max( max_len[language], len( lan[language][j] ) )
            n += 1
        # If there is no sentence valid, ignore the batch
        if n == 0:
            continue
        np_lan = {}
        for language in lan_list:
             np_lan[language] = np.zeros( ( n, max_len[language] ) )
        n = 0
        for j in range( len( lan[lan_list[0]] ) ):
            length = len( lan[lan_list[0]][j] )
            if length < min_length or max_length < length:
                continue
            for language in lan_list:
                length = len( lan[language][j] )
                for k in range( length ):
                    if language == "label":
                        if k:
                            np_lan[language][n, k - 1] = lan[language][j][k]
                    else:
                        np_lan[language][n, k] = lan[language][j][k]
            n += 1
        for language in lan_list:
            tdata[language].append( np_lan[language] )
    for language in lan_list:
        data[language] = tdata[language]

#def pretrained_embedding_layer( word_to_vec_map, word_to_index ):
def pretrained_embedding_layer( language ):
    """Creates a Keras Embedding() layer and loads in pre-trained word2vec 300-dimensional vectors.

    Args:
        word_to_vec_map: dictionary mapping words to their word2vec vector representation.
        word_to_index: dictionary mapping from words to their indices in the vocabulary

    Returns:
        embedding_layer: pretrained layer Keras instance
    """
    emb_matrix = np.load( word2vec_path + language + "/" + language )
    vocab_len = emb_matrix.shape[0] + 1
    emb_dim   = emb_matrix.shape[1]
    emb_matrix = np.append( emb_matrix, np.zeros( ( 1, emb_dim ) ), axis = 0 )
    print( "embedding: ", language, emb_matrix.shape )

    # Define Keras embedding layer with the correct output/input sizes, make it trainable.
    # Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding( vocab_len, emb_dim, trainable = False, mask_zero = True )

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    # Do not modify the "None".
    embedding_layer.build( ( None, ) )
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights( [emb_matrix] )
    
    return embedding_layer

def simple_seq2seq( vocab_size_list, language_list,
                    hidden_dim = 128, name = "baseline" ):
    ### Encoder-Decoder for train ###
    
    if EMB:
        encoder_embedding = pretrained_embedding_layer( language_list[0] )
        decoder_embedding = pretrained_embedding_layer( language_list[1] )

    # Encoder
    encoder = Bidirectional( LSTM( hidden_dim, return_state = True,
                                   return_sequences = True,
                                   kernel_initializer = "glorot_normal",
                                   name = name + "_encoder_lstm" ) )
    if EMB:
        encoder_input = Input( shape = ( None, ),
                               name = name + "_encoder_input" )
        encoder_input_emb = encoder_embedding( encoder_input )
    else:
        encoder_input = Input( shape = ( None, vocab_size_list[0] ),
                               name = name + "_encoder_input" )
        encoder_input_emb = encoder_input
    encoder_output, forward_state_h, forward_state_c, \
        backward_state_h, backward_state_c = encoder( encoder_input_emb )
    state_h = Concatenate()( [forward_state_h, backward_state_h] )
    state_c = Concatenate()( [forward_state_c, backward_state_c] )
    encoder_state = [state_h, state_c]
    
    # Decoder
    decoder = LSTM( hidden_dim * 2, return_state = True, return_sequences = True,
                    kernel_initializer = "glorot_normal",
                    name = name + "_decoder_lstm" )
    decoder_dense_pre = TimeDistributed( Dense( hidden_dim * 2,
                                                kernel_initializer = "glorot_normal",
                                                activation = 'tanh',
                                                name = name + "_decoder_output_pre" ) )
    decoder_dense     = TimeDistributed( Dense( vocab_size_list[1],
                                                activation = "softmax",
                                                kernel_initializer = "glorot_normal",
                                                name = name + "_decoder_output" ) )
    
    if EMB:
        decoder_input = Input( shape = ( None, ),
                               name = name + "_decoder_input" )
        decoder_input_emb = decoder_embedding( decoder_input )
    else:
        decoder_input = Input( shape = ( None, vocab_size_list[1] ),
                               name = name + "_decoder_input" )
        decoder_input_emb = decoder_input
    decoder_output, state_h, state_c = decoder( decoder_input_emb,
                                                initial_state = encoder_state )

    # Attention mechanism
    attention = Dot( axes = [2, 2] )( [decoder_output, encoder_output] )
    attention = Softmax()( attention )
    context   = Dot( axes = [2, 1] )( [attention, encoder_output] )
    decoder_with_context = Concatenate( axis = -1 )( [context, decoder_output] )
    decoder_output = decoder_dense_pre( decoder_with_context )

    decoder_output = decoder_dense( decoder_output )
    
    # Model
    model = Model( inputs = [encoder_input, decoder_input],
                   outputs = decoder_output,
                   name = name )
    if MUTI_GPU > 1:
        model = multi_gpu_model( model, gpus = MUTI_GPU )
    model.compile( optimizer = 'rmsprop', loss = "categorical_crossentropy" )
    
    ### Encoder-Decoder for generation ###
    
    # Encoder Model
    encoder_model   = Model( inputs  = encoder_input,
                             outputs = encoder_state,
                             name = name + "_encoder" )
    if MUTI_GPU > 1:
        encoder_model = multi_gpu_model( encoder_model, gpus = MUTI_GPU )
    
    # Decoder Model
    decoder_state_h = Input( shape = ( hidden_dim * 2, ), name = name + "_state_h" )
    decoder_state_c = Input( shape = ( hidden_dim * 2, ), name = name + "_state_c" )
    decoder_state_input = [decoder_state_h, decoder_state_c]
    decoder_output, state_h, state_c = decoder( decoder_input_emb,
                                                initial_state = decoder_state_input )
    decoder_state   = [state_h, state_c]
    decoder_output  = decoder_dense( decoder_output )
    decoder_model   = Model( inputs  = [decoder_input] + decoder_state_input,
                             outputs = [decoder_output] + decoder_state,
                             name = name + "_decoder" )
    if MUTI_GPU > 1:
        decoder_model = multi_gpu_model( decoder_model, gpus = MUTI_GPU )
    
    return model, encoder_model, decoder_model

def translate_sentences( data, encoder_model, decoder_model, max_len,
                         word_to_idx_dict, idx_to_word_dict, language_list,
                         path = "" ):
    """Generate sentences based on given sentences

    Args:
        data: a list of dev data sentences. Its structure is:

              [[word1, word2, ...], [...], ...]

        encoder_model: encoder part of seq2seq model.
        decoder_model: decoder (generate) part of seq2se1 model.
        max_len: a interger represents the max length of generated (translated)
                 sentence.
        word_to_idx_dict: a dictionary converts word to index. Its structure is:

                          {language A: {word A: index A, word B: ..., ...},
                           language B: ...}.

        idx_to_word_dict: a dictionary converts index to word. Its structure is:

                          {language A: {index A: word A, index B: ..., ...},
                           language B: ...}.

    Returns:
        sentences: a list of generated (translated) sentences.
    """
    f = open( path + "translated_sentence.txt", "w" )
    sentences = []
    lan_list = language_list
    input_vocab_size  = len( word_to_idx_dict[lan_list[0]] )
    output_vocab_size = len( word_to_idx_dict[lan_list[1]] )
    for sentence in data:
        init = "<START>"
        cnt = 0
        words = []
        if not EMB:
            sentence = K.utils.to_categorical( sentence, input_vocab_size )
        state = encoder_model.predict( sentence )
        while init != "<END>" and cnt <= max_len + 1:
            index = np.array( [word_to_idx_dict[lan_list[1]][init]] ).reshape( ( 1, 1 ) )
            if not EMB:
                index = [K.utils.to_categorical( index, output_vocab_size )]
            indeces, state_h, state_c = decoder_model.predict( [index] + state )
            index = np.argmax( indeces[0, -1, :] )
            init = idx_to_word_dict[lan_list[1]][index]
            state = [state_h, state_c]
            words.append( init )
            cnt += 1
        print( words[:-1] )
        f.write( ' '.join( words[:-1] ) + "\n" )
    f.close()

def one_hot_to_categorical( epoch, no_of_batch, category, no_of_category,
                            data, language_list, vocab_size_list, queue ):
    for ep in range( epoch ):
        for i in range( category, no_of_batch, no_of_category ):
            train_data = []
            if EMB:
                train_data.append( data[language_list[0]][i] )
                train_data.append( data[language_list[1]][i] )
            else:
                train_data.append( K.utils.to_categorical( data[language_list[0]][i], vocab_size_list[0] ) )
                train_data.append( K.utils.to_categorical( data[language_list[1]][i], vocab_size_list[1] ) )
            label = K.utils.to_categorical( data[language_list[2]][i], vocab_size_list[1] )
            queue.put( [train_data, label] )

def check_path( path ):
    if not os.path.exists( path ):
        os.mkdir( path )

if __name__ == "__main__":
    check_path( loss_path  )
    check_path( dict_path  )
    check_path( model_path )
    check_path( train_data_path )
    check_path( test_data_path  )
    check_path( generation_path )

    if TRAIN:
        language_list = ["chinese", "english"] # [ori_lan, tar_lan]
        data = load_data( train_data_path, language_list, shuffle = False )
        if CONTINUE:
            word_to_idx_dict, idx_to_word_dict = load_dictionary( language_list )
            # truncate all content in losses.txt
            f = open( loss_path + "losses.txt", "w" )
            f.close()
        else:
            word_to_idx_dict, idx_to_word_dict = load_dictionary( language_list )
#            word_to_idx_dict, idx_to_word_dict = build_dictionary( data, 0 )
#            save_dict( word_to_idx_dict, "word_to_idx.vab", dict_path )
#            save_dict( idx_to_word_dict, "idx_to_word.vab", dict_path )
        vocab_size_list = [len( word_to_idx_dict[language_list[0]] ),
                           len( word_to_idx_dict[language_list[1]] )]
        print( vocab_size_list )
        sort_by_ori_lan( data )
        convert_to_index( data, word_to_idx_dict )
        convert_to_batch( data, language_list, batch_size = batch_size,
                          min_length = min_length, max_length = max_length )
        print( language_list )
        print( len( data[language_list[0]] ) )

        print( "Building Model" )
        model, encoder_model, decoder_model = simple_seq2seq( vocab_size_list, language_list,
                                                              name = model_name )

        latest = 0
        if CONTINUE:
            with open( model_path + "checkpoint", "r" ) as f:
                latest = f.readline().strip()
            if len( latest ):
                model.load_weights( model_path + "model_weights_" + latest + ".h5" )
                latest = int( latest )
        no_of_batch = len( data[language_list[0]] )
        manager = multiprocessing.Manager()
        queue = manager.Queue( 10 )
        p = []
        no_of_generator = 3
        for i in range( no_of_generator ):
            p_i = multiprocessing.Process( target = one_hot_to_categorical,
                                           args = ( epoch, no_of_batch, i, no_of_generator,
                                                    data, language_list,
                                                    vocab_size_list,
                                                    queue ) )
            p.append( p_i )
            p_i.start()
   
        time.sleep( 5 )
        print( "Training Model" )
        losses = []
        n = 0
        while n < epoch * no_of_batch:
            train_data, label = queue.get_nowait()
            loss = model.train_on_batch( train_data, label )
            losses.append( loss )
            n += 1
            print( n, loss )
            if ( n + latest ) % 5000 == 0:
                model.save_weights( model_path + "model_weights_" + str( n + latest ) + ".h5" )
                with open( model_path + "checkpoint", "w" ) as f:
                    f.write( str( n + latest ) )
        model.save_weights( model_path + "model_weights_" + str( n + latest ) + ".h5" )
        with open( model_path + "checkpoint", "w" ) as f:
            f.write( str( n + latest ) )
        for p_i in p:
            p_i.join()
        with open( loss_path + "losses.txt", "w+" ) as f_loss:
            for loss in losses:
                f_loss.write( str( loss ) + "\n" )

    if GENERATION:
        language_list = ["chinese", "english"] # [ori_lan, tar_lan]
        if not TRAIN:
            word_to_idx_dict, idx_to_word_dict = load_dictionary( language_list )
            vocab_size_list = [len( word_to_idx_dict[language_list[0]] ),
                               len( word_to_idx_dict[language_list[1]] )]
            print( "Building Model" )
            model, encoder_model, decoder_model = simple_seq2seq( vocab_size_list, language_list,
                                                                  name = model_name )
            with open( model_path + "checkpoint", "r" ) as f:
                latest = f.readline().strip()
            model.load_weights( model_path + "model_weights_" + latest + ".h5" )

        data = load_data( test_data_path, language_list, shuffle = False )
        convert_to_index( data, word_to_idx_dict )
        convert_to_batch( data, language_list, batch_size = 1, min_length = 0, max_length = 10000 )
        print( "Generating sentences" )
        translated_sentences = translate_sentences( data[language_list[0]],
                                                    encoder_model, decoder_model,
                                                    max_length,
                                                    word_to_idx_dict, idx_to_word_dict,
                                                    language_list, generation_path )

