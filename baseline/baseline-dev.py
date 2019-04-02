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
import numpy as np

import numpy as np
import tensorflow as tf
import keras as K
import keras.backend.tensorflow_backend as KTF
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, \
                            Lambda, Concatenate, Bidirectional
K.backend.clear_session()
sess = tf.Session( config = tf.ConfigProto( device_count = {'gpu':0} ) )
KTF.set_session( sess )

# Global debug parameter
# Open embedding layer or not
EMB = False
# Open UNK or not
UNK = False
# Train on multi GPU(1, 2, 3, ...)
MUTI_GPU = 1

def load_vectors( path, word2vec, language ):
    files = os.listdir( path + language + "/" )
    data = {}
    for file_name in files:
        fin = io.open( path + language + "/" + file_name, "r", encoding = "utf-8",
                       newline = "\n", errors = "ignore" )
        n, d = map( int, fin.readline().split() )
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

def get_language_data( path, language, encoding, lan_data = {} ):
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
                words = ["<S>"] + line.split() + ["</S>"]
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

def get_data( path = "../data/", language_list = ["chinese", "english"],
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
        p_lan = multiprocessing.Process( target = get_language_data,
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
    print( "Converting to index..." )
    language_list = list( data.keys() )
    for language in language_list:
        for i in range( len( data[language] ) ):
            new_sentence = []
            for word in data[language][i]:
                if word not in word_to_idx_dict[language]:
                    if not UNK:
                        continue
                    word = "UNK"
                new_sentence.append( word_to_idx_dict[language][word] )
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
            if isinstance( data[lan_list[0]][0][0], str ):
                dtype = np.unicode_
                np_lan[language] = np.empty( ( n, max_len[language] ), dtype = dtype )
                np_lan[language][:] = ""
            else:
                np_lan[language] = np.zeros( ( n, max_len[language] ) )
        n = 0
        for j in range( len( lan[lan_list[0]] ) ):
            length = len( lan[lan_list[0]][j] )
            if length < min_length or max_length < length:
                continue
            for language in lan_list:
                for k in range( len( lan[language][j] ) ):
                    np_lan[language][n, k] = lan[language][j][k]
            n += 1
        for language in lan_list:
            tdata[language].append( np_lan[language] )
    for language in lan_list:
        data[language] = tdata[language]

def convert_to_pair( data, language_list ):
    """Pack sentence at the same index in each language into one pair

    Args:
        data: a dictionary contains sentences of each language.  Its structure is:

              {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
        language_list: a list of string represents languages.

    Returns:
        data: a list contains batches of all data set. Its structure is:

              [[[word1, word2], [...], ...], ...]
    """
    print( "Converting to pair..." )
    new_data = []
    for language in language_list:
        new_data.append( data[language] )
    new_data = list( zip( *new_data ) )
    new_data = list( map( list, new_data ) )
    return new_data

def pretrained_embedding_layer( word_to_vec_map, word_to_index ):
    """Creates a Keras Embedding() layer and loads in pre-trained word2vec 300-dimensional vectors.

    Args:
        word_to_vec_map: dictionary mapping words to their word2vec vector representation.
        word_to_index: dictionary mapping from words to their indices in the vocabulary

    Returns:
        embedding_layer: pretrained layer Keras instance
    """
    vocab_len = len(word_to_index) + 1 # adding 1 to fit Keras embedding (requirement)
    if EMB:
        emb_dim = word_to_vec_map["cucumber"].shape[0]
    else:
        emb_dim = 1
    
    # Initialize the embedding matrix as a numpy array of zeros of shape
    # (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros( ( vocab_len, emb_dim ) )
    
    # Set each row "index" of the embedding matrix to be the word vector
    # representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        if EMB:
            if word not in word_to_index:
                emb_matrix[index, :] = np.random.random( ( 1, 300 ) )
            else:
                emb_matrix[index, :] = word_to_vec_map[word]
        else:
            emb_matrix[index, :] = index

    # Define Keras embedding layer with the correct output/input sizes, make it trainable.
    # Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding( vocab_len, emb_dim, trainable = False, mask_zero = True )

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    # Do not modify the "None".
    embedding_layer.build( ( None, ) )
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights( [emb_matrix] )
    
    return embedding_layer

def simple_seq2seq( input_vocab_size, output_vocab_size,
                    encoder_embedding, decoder_embedding,
                    hidden_dim = 128, word_vec_dim = 300,
                    name = "baseline" ):
    ### Encoder-Decoder for train ###
    
    # Encoder
    encoder = Bidirectional( LSTM( hidden_dim, return_state = True,
                                   name = name + "_encoder_lstm" ) )
    encoder_input = Input( shape = ( None, ),
                           name = name + "_encoder_input" )
    
    encoder_input_emb   = encoder_embedding( encoder_input )
    _, forward_state_h, forward_state_c, \
        backward_state_h, backward_state_c = encoder( encoder_input_emb )
    state_h = Concatenate()( [forward_state_h, backward_state_h] )
    state_c = Concatenate()( [forward_state_c, backward_state_c] )
    encoder_state = [state_h, state_c]
    
    # Decoder
    decoder = LSTM( hidden_dim * 2, return_state = True, return_sequences = True,
                    name = name + "_decoder_lstm" )
    decoder_dense = Dense( output_vocab_size, activation = "softmax",
                           name = name + "_decoder_output" )
    decoder_input = Input( shape = ( None, ),
                           name = name + "_decoder_input" )
    
    decoder_input_emb = decoder_embedding( decoder_input )
    decoder_output, state_h, state_c = decoder( decoder_input_emb,
                                                initial_state = encoder_state )
    decoder_output    = decoder_dense( decoder_output )
    
    # Model
    model = Model( inputs = [encoder_input, decoder_input],
                   outputs = decoder_output,
                   name = name )
    if MUTI_GPU > 1:
        model = multi_gpu_model( model, gpus = MUTI_GPU )
    model.compile( optimizer = 'adam', loss = "categorical_crossentropy" )
    
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
    for sentence in data:
        init = "<S>"
        cnt = 0
        words = []
        state = encoder_model.predict( sentence )
        while init != "</S>" and cnt <= max_len + 1:
            index = np.array( [word_to_idx_dict[lan_list[1]][init]] ).reshape( ( 1, 1 ) )
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
                            data, language, output_vocab_size, label_queue ):
    for ep in range( epoch ):
        for i in range( category, no_of_batch, no_of_category ):
            label = K.utils.to_categorical( data[i][language], output_vocab_size )
            label_queue.put( [i, label] )

if __name__ == "__main__":
    model_name = "baseline"
    language_list = ["chinese", "english"] # [ori_lan, tar_lan]
    batch_size = 128
    max_length = 32

    data = get_data( "../data/test/", language_list, shuffle = False )
    word_to_idx_dict, idx_to_word_dict = build_dictionary( data, 5 )
    if EMB:
        word_to_vec = load_word2vec( "../data/word2vec/", language_list )
    else:
        word_to_vec = word_to_idx_dict
    print( len( word_to_idx_dict[language_list[0]] ), len( word_to_idx_dict[language_list[1]] ) )
    save_dict( word_to_idx_dict, "word_to_idx.json" )
    save_dict( idx_to_word_dict, "idx_to_word.json" )
    input_vocab_size = len( word_to_idx_dict[language_list[0]] )
    output_vocab_size = len( word_to_idx_dict[language_list[1]] )
    sort_by_ori_lan( data )
    convert_to_index( data, word_to_idx_dict )
    convert_to_batch( data, language_list, batch_size = batch_size, max_length = max_length )
    data = convert_to_pair( data, language_list )
    print( len( data ) )

    epoch = 350
    no_of_batch = len( data )
    manager = multiprocessing.Manager()
    labels = manager.Queue( 20 )
    p = []
    no_of_generator = 2
    for i in range( no_of_generator ):
        p_i = multiprocessing.Process( target = one_hot_to_categorical,
                                       args = ( epoch, no_of_batch, i, no_of_generator,
                                                data, 1,
                                                output_vocab_size,
                                                labels ) )
        p.append( p_i )
        p_i.start()

    print( "Building Model" )
    encoder_embedding = pretrained_embedding_layer(
                            word_to_vec[language_list[0]], word_to_idx_dict[language_list[0]] )
    decoder_embedding = pretrained_embedding_layer(
                            word_to_vec[language_list[1]], word_to_idx_dict[language_list[1]] )
    model, encoder_model, decoder_model = simple_seq2seq( input_vocab_size,  output_vocab_size,
                                                          encoder_embedding, decoder_embedding,
                                                          name = model_name )

    print( "Training Model" )
    losses = []
    n = 0
    while n < epoch * no_of_batch:
        tmp = labels.get_nowait()
        idx, label = tmp
#    for _ in range( epoch ):
#        for idx in range( no_of_batch ):
#           label = K.utils.to_categorical( data[idx][1], output_vocab_size )
        loss = model.train_on_batch( data[idx], label )
        losses.append( loss )
        n += 1
        print( n, loss )
        if n % 5000 == 0:
            model.save_weights( "models/model_weights_" + str( n ) + ".h5" )
    labels.task_done()

    for p_i in p:
        p_i.join()
    losses = list( losses )

    with open( "losses.txt", "w" ) as f:
        for loss in losses:
            f.write( str( loss ) + "\n" )

    data = get_data( "../data/test/", language_list, shuffle = False )
    convert_to_index( data, word_to_idx_dict )
    convert_to_batch( data, language_list, batch_size = 1, min_length = 0, max_length = 10000 )
    print( "Generating sentences" )
    translated_sentences = translate_sentences( data[language_list[0]],
                                                encoder_model, decoder_model,
                                                max_length,
                                                word_to_idx_dict, idx_to_word_dict,
                                                language_list )

