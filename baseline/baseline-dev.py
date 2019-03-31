#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import tensorflow as tf
import keras as K
import keras.backend.tensorflow_backend as KTF
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, Lambda


# In[ ]:


# K.backend.clear_session()
# sess = tf.Session( config = tf.ConfigProto( device_count = {'gpu':0} ) )
# KTF.set_session( sess )


# In[ ]:


# Global debug parameter

# open embedding layer or not
EMB = False


# In[ ]:


def load_vectors( path, word2vec, language ):
    files = os.listdir( path + language + "/" )
    data = {}
    for file in files:
        fin = io.open( path + file, "r", encoding = "utf-8",
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


# In[ ]:


def get_language_data( path, language, encoding, lan_data = {} ):
    sentences = []
    files = os.listdir( path + language + "/" )
    for file in files:
        with open( path + language + "/" + file, "r", encoding = encoding ) as f:
            line = f.readline()
            while line:
                line = line.strip()
                if len( line ) == 0:
                    line = f.readline()
                    continue
                words = ["<S>"] + line.split() + ["</S>"]
                sentences.append( words )
                line = f.readline()
    lan_data[language] = sentences

def shuffle_list( x, shuf_order ):
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


# In[ ]:


def build_language_dictionary( language, sentences,
                               word_to_idx_dict, idx_to_word_dict,
                               threshold = 0 ):
    # Generate dictionary for each language
    word_to_idx_dict_lan = {"<PAD>": 0, "<S>": 1, "</S>": 2, "<UNK>": 3}
    idx_to_word_dict_lan = {0: "<PAD>", 1: "<S>", 2: "</S>", 3: "<UNK>"}

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


# In[ ]:


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

def convert_to_index( data, word_to_idx_dict ):
    """Convert word to index

    Args:
        data: a dictionary contains sentences of each language.  Its structure is:

              {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
        word_to_idx_dict: a dictionary converts word to index. Its structure is:

                          {language A: {word A: index A, word B: ..., ...},
                           language B: ...}.
    """
    print( "Coverting to index..." )
    lan_list = list( data.keys() )
    for lan in lan_list:
        for sentence in data[lan]:
            for i in range( len( sentence ) ):
                word = sentence[i]
                if word not in word_to_idx_dict[lan]:
                    word = "<UNK>"
                sentence[i] = word_to_idx_dict[lan][word]


# In[ ]:


def pretrained_embedding_layer( word_to_vec_map, word_to_index ):
    """
    Creates a Keras Embedding() layer and loads in pre-trained word2vec 300-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their word2vec vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
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


# In[1]:


def convert_to_batch( data, batch_size = 64, min_length = 3, max_length = 32 ):
    print( "Converting to batches..." )
    lan_list = list( data.keys() )
    tdata = {}
    for lan in lan_list:
        if lan not in tdata:
            tdata[lan] = []
    print( type( data[lan_list[0]][0][0] ),
           isinstance( data[lan_list[0]][0][0], str ) )
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


# In[ ]:


def simple_seq2seq( input_vocab_size, output_vocab_size,
                    encoder_embedding, decoder_embedding,
                    hidden_dim = 128, word_vec_dim = 300,
                    name = "baseline" ):
    ### Encoder-Decoder for train ###
    
    # Encoder
    encoder = LSTM( hidden_dim, return_state = True,
                    name = name + "_encoder_lstm" )
    encoder_input = Input( shape = ( None, ),
                           name = name + "_encoder_input" )
    
    encoder_input_emb   = encoder_embedding( encoder_input )
    _, state_h, state_c = encoder( encoder_input_emb )
    encoder_state       = [state_h, state_c]
    
    # Decoder
    decoder = LSTM( hidden_dim, return_state = True, return_sequences = True,
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
    model.compile( optimizer = 'adam', loss = "categorical_crossentropy" )
    
    ### Encoder-Decoder for generation ###
    
    # Encoder Model
    encoder_model   = Model( inputs  = encoder_input,
                             outputs = encoder_state,
                             name = name + "_encoder" )
    
    # Decoder Model
    decoder_state_h = Input( shape = ( hidden_dim, ), name = name + "_state_h" )
    decoder_state_c = Input( shape = ( hidden_dim, ), name = name + "_state_c" )
    decoder_state_input = [decoder_state_h, decoder_state_c]
    decoder_output, state_h, state_c = decoder( decoder_input_emb,
                                                initial_state = decoder_state_input )
    decoder_state   = [state_h, state_c]
    decoder_output  = decoder_dense( decoder_output )
    decoder_model   = Model( inputs  = [decoder_input] + decoder_state_input,
                             outputs = [decoder_output] + decoder_state,
                             name = name + "_decoder" )
    
    return model, encoder_model, decoder_model


# In[ ]:


def translate_sentences( data, encoder_model, decoder_model, max_len,
                         word_to_idx_dict, idx_to_word_dict, language_list ):
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
        sentences.append( words[:-1] )
    return sentences


# In[ ]:


if __name__ == "__main__":
    model_name = "baseline"
    language_list = ["chinese", "english"] # [ori_lan, tar_lan]
    batch_size = 64
    max_length = 32

    data = get_data( "../data/test/", language_list, shuffle = False )
    word_to_idx_dict, idx_to_word_dict = build_dictionary( data, 15 )
    if EMB:
        word_to_vec = load_word2vec( "../data/word2vec/", language_list )
    else:
        word_to_vec = word_to_idx_dict
    print( len( word_to_idx_dict[language_list[0]] ), len( word_to_idx_dict[language_list[1]] ) )
    save_dict( word_to_idx_dict, "word_to_idx.json" )
    save_dict( idx_to_word_dict, "idx_to_word.json" )
    input_vocab_size = len( word_to_idx_dict[language_list[0]] )
    output_vocab_size = len( word_to_idx_dict[language_list[1]] )

    print( "Building Model" )
    encoder_embedding = pretrained_embedding_layer(
                            word_to_vec[language_list[0]], word_to_idx_dict[language_list[0]] )
    decoder_embedding = pretrained_embedding_layer(
                            word_to_vec[language_list[1]], word_to_idx_dict[language_list[1]] )
    model, encoder_model, decoder_model = simple_seq2seq( input_vocab_size,  output_vocab_size,
                                                          encoder_embedding, decoder_embedding,
                                                          name = model_name )

    sort_by_ori_lan( data )
    tdata = copy.deepcopy( data )
    for lan in language_list:
        print( len( data[lan] ), len( tdata[lan] ) )
    convert_to_index( tdata, word_to_idx_dict )
    convert_to_batch( data )
    convert_to_batch( tdata )
    for lan in language_list:
        print( len( data[lan] ) )

    print( "Training Model" )
    n = 0
    losses = []
    for epoch in range( 0 ):
        for i in range( len( data[language_list[0]] ) ):
            label = K.utils.to_categorical( tdata[language_list[1]][i],
                                            output_vocab_size )
            loss = model.train_on_batch( [data[language_list[0]][i],
                                          data[language_list[1]][i]],
                                         label )
            losses.append( loss )
            print( n, loss )
            n += 1
            if n % 5000 == 0:
                model.save_weights( "models/model_weights_" + str( n ) + ".h5" )
    with open( "losses.txt", "w" ) as f:
        for loss in losses:
            f.write( str( loss ) + "\n" )

    data = get_data( "../data/test/", language_list, shuffle = False )
    convert_to_index( data, word_to_idx_dict )
    convert_to_batch( data, batch_size = 1, min_length = 0, max_length = 10000 )
    print( "Generating sentences" )
    translated_sentences = translate_sentences( data[language_list[0]],
                                                encoder_model, decoder_model,
                                                max_length,
                                                word_to_idx_dict, idx_to_word_dict,
                                                language_list )
    with open( "translated_sentence.txt", "w" ) as f:
        for sentence in translated_sentences:
            f.write( sentence + "\n" )
