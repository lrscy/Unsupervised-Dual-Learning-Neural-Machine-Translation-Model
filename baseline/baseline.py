#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import random
import math
import multiprocessing
import h5py
import json
import numpy as np
import gensim
import tensorflow as tf
import keras as K
import keras.backend.tensorflow_backend as KTF
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, Lambda


# In[2]:


# K.backend.clear_session()
# sess = tf.Session( config = tf.ConfigProto( device_count = {'gpu':0} ) )
# KTF.set_session( sess )


# In[4]:


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
def get_data( path = "../data/", language_list = ["chinese", "english"],
              encoding_list = ["UTF-8", "UTF-8"], shuffle = True ):
    assert len( language_list ) == len( encoding_list )
    # Just for my convenient in the following
    lan_list, enc_list = language_list, encoding_list
    
    # Read parallel corpus
    lan_data = {}
    for i in range( len( lan_list ) ):
        lan = lan_list[i]
        print( "Reading " + lan + " language corpus..." )
        if lan not in lan_data:
            lan_data[lan] = []
        files = os.listdir( path + lan + "/" )
        for file in files:
            with open( path + lan + "/" + file, "r", encoding = enc_list[i] ) as f:
                line = f.readline()
                while line:
                    line = line.strip()
                    if len( line ) == 0:
                        line = f.readline()
                        continue
                    words = ["<S>"] + line.split() + ["</S>"]
                    lan_data[lan].append( words )
                    line = f.readline()
    
    if shuffle == True:
        print( "Shuffling..." )
        
        # Decide shuffle order
        length = len( lan_data[lan_list[0]] )
        shuf_list = [i for i in range( length )]
        random.shuffle( shuf_list )

        # Shuffle corpus
        for lan in lan_list:
            lan_data[lan] = np.array( lan_data[lan] )[shuf_list].tolist()
    
    return lan_data


# In[5]:


"""Save dictionary to file

Args:
    dt: the dictionary to save.
    file_name: the name of file to save.
    path: the path of file to save.

Returns:
    None.
"""
def save_dict( dt, file_name, path = "dicts/" ):
    print( "Saving to " + path + file_name + "..." )
    file_name = path + file_name
    with open( file_name, "w" ) as f:
        jsn = json.dumps( dt )
        f.write( jsn )


# In[6]:


"""Sort dictionary by length of original language

Args:
    data: a dictionary contains sentences of each language.  Its structure is:
    
          {language A: [[word1, word2, ...], [...], ...], language B: ...}.
    
Returns:
    None.
"""
def sort_by_ori_lan( data ):
    print( "Sorting..." )
    lan = list( data.keys() )
    tmp = list( zip( data[lan[0]], data[lan[1]] ) )
    tmp.sort( key = lambda x: len( x[0] ) )
    data[lan[0]], data[lan[1]] = zip( *tmp )


# In[7]:


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
def build_dictionary( language_data, threshold = 0 ):
    lan_data = language_data
    word_to_idx_dict = {}
    idx_to_word_dict = {}
    for lan, sentences in lan_data.items():
        # Generate dictionary for each language
        if lan not in word_to_idx_dict:
            word_to_idx_dict[lan] = {"<PAD>": 0, "<S>": 1, "</S>": 2, "<UNK>": 3}
        if lan not in idx_to_word_dict:
            idx_to_word_dict[lan] = {0: "<PAD>", 1: "<S>", 2: "</S>", 3: "<UNK>"}
        
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
            if word not in word_to_idx_dict[lan]:
                idx = len( word_to_idx_dict[lan] )
                word_to_idx_dict[lan][word] = idx
                idx_to_word_dict[lan][idx] = word
                
    return word_to_idx_dict, idx_to_word_dict


# In[8]:


def convert_to_index( data, word_to_idx_dict ):
    print( "Coverting to index..." )
    lan_list = list( data.keys() )
    for lan in lan_list:
        for sentence in data[lan]:
            for i in range( len( sentence ) ):
                word = sentence[i]
                if word not in word_to_idx_dict[lan]:
                    word = "<UNK>"
                sentence[i] = word_to_idx_dict[lan][word]


# In[9]:


"""
Creates a Keras Embedding() layer and loads in pre-trained word2vec 300-dimensional vectors.

Arguments:
word_to_vec_map -- dictionary mapping words to their word2vec vector representation.
word_to_index -- dictionary mapping from words to their indices in the vocabulary

Returns:
embedding_layer -- pretrained layer Keras instance
"""
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
#     w2v_dict = word_to_vec_map.vocab.keys()
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = 1
#     emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros( ( vocab_len, emb_dim ) )
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = index
#         if word not in word_to_index:
#             emb_matrix[index, :] = np.random.random( ( 1, 300 ) )
#         else:
#             emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable.
    # Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding( vocab_len, emb_dim, trainable = False, mask_zero = True, )
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


# In[10]:


def convert_to_batch( data, batch_size = 64, min_length = 3, max_length = 32 ):
    print( "Converting to batches..." )
    tdata = {}
    lan_list = list( data.keys() )
    for lan in lan_list:
        if lan not in tdata:
            tdata[lan] = []
    for i in range( 0, len( data[lan_list[0]] ) + batch_size, batch_size ):
        lan0 = data[lan_list[0]][i: i + batch_size]
        lan1 = data[lan_list[1]][i: i + batch_size]
        # Calculate max length of valid sentences
        n = 0
        max_len0, max_len1 = 0, 0
        for j in range( len( lan0 ) ):
            len0 = len( lan0[j] )
            len1 = len( lan1[j] )
            if min_length <= len0 and len0 <= max_length:
                max_len0 = max( max_len0, len0 )
                max_len1 = max( max_len1, len1 )
                n += 1
        # If there is no sentence valid, ignore the batch
        if n == 0:
            continue
        # Convert to batch
        if isinstance( data[lan_list[0]], str ):
            dtype = np.unicode_
        else:
            dtype = np.int
        np_lan0 = np.empty( ( n, max_len0 ), dtype = dtype )
        np_lan1 = np.empty( ( n, max_len1 ), dtype = dtype )
        if isinstance( data[lan_list[0]], str ):
            np_lan0[:] = ""
            np_lan1[:] = ""
        else:
            np_lan0[:] = 0
            np_lan1[:] = 0
        n = 0
        for j in range( len( lan0 ) ):
            len0 = len( lan0[j] )
            if min_length <= len0 and len0 <= max_length:
                for k in range( len( lan0[j] ) ):
                    np_lan0[n, k] = lan0[j][k]
                for k in range( len( lan1[j] ) ):
                    np_lan1[n, k] = lan1[j][k]
                n += 1
        tdata[lan_list[0]].append( np_lan0 )
        tdata[lan_list[1]].append( np_lan1 )
    return tdata


# In[11]:


def simple_seq2seq( input_vocab_size, output_vocab_size,
                    encoder_embedding, decoder_embedding,
                    hidden_dim = 128, word_vec_dim = 300,
                    name = "baseline" ):
    ### Encoder-Decoder for train ###
    
    # Encoder
#     encoder_embedding = Embedding( output_dim = word_vec_dim,
#                                    input_dim  = input_vocab_size,
#                                    mask_zero  = True,
#                                    name = name + "_encoder_embedding")
    encoder = LSTM( hidden_dim, return_state = True,
                    name = name + "_encoder_lstm" )
    encoder_input = Input( shape = ( None, ),
                           name = name + "_encoder_input" )
    
    encoder_input_emb   = encoder_embedding( encoder_input )
    _, state_h, state_c = encoder( encoder_input_emb )
    encoder_state       = [state_h, state_c]
    
    # Decoder
#     decoder_embedding = Embedding( output_dim = word_vec_dim,
#                                    input_dim = output_vocab_size,
#                                    mask_zero = True,
#                                    name = name + "_decoder_embedding")
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


# In[12]:


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
def translate_sentences(data, encoder_model, decoder_model, max_len,
                        word_to_idx_dict, idx_to_word_dict, language_list):
    sentences = []
    lan_list = language_list
    for sentence in data:
        init = "<S>"
        cnt = 0
        words = []
        sentence_ = [word_to_idx_dict[lan_list[0]][x]
                     if x in word_to_idx_dict[lan_list[0]] else 3
                     for x in sentence]
        state = encoder_model.predict(sentence_)
        while init != "</S>" and cnt <= max_len + 1:
            index = np.array([word_to_idx_dict[lan_list[1]][init]]).reshape( ( 1, 1 ) )
            indeces, state_h, state_c = decoder_model.predict([index] + state)
            index = np.argmax(indeces[0, -1, :])
            init = idx_to_word_dict[lan_list[1]][index]
            state = [state_h, state_c]
            words.append(init)
            cnt += 1
        print( words[:-1] )
        sentences.append(words[:-1])
    return sentences


# In[18]:


model_name = "baseline"
language_list = ["chinese", "english"] # [ori_lan, tar_lan]
batch_size = 64
max_length = 32

data = get_data( "../data/test/" )
word_to_idx_dict, idx_to_word_dict = build_dictionary( data, 15 )
print( len( word_to_idx_dict[language_list[0]] ), len( word_to_idx_dict[language_list[1]] ) )
save_dict( word_to_idx_dict, "word_to_idx.json" )
save_dict( idx_to_word_dict, "idx_to_word.json" )
input_vocab_size = len( word_to_idx_dict[language_list[0]] )
output_vocab_size = len( word_to_idx_dict[language_list[1]] )

print( "Building Model" )
encoder_embedding = pretrained_embedding_layer(
                        word_to_idx_dict[language_list[0]], word_to_idx_dict[language_list[0]] )
decoder_embedding = pretrained_embedding_layer(
                        word_to_idx_dict[language_list[1]], word_to_idx_dict[language_list[1]] )
model, encoder_model, decoder_model = simple_seq2seq( input_vocab_size,
                                                      output_vocab_size,
                                                      encoder_embedding,
                                                      decoder_embedding,
                                                      name = model_name )

sort_by_ori_lan( data )
tdata = data.copy()
for lan in language_list:
    print( len( data[lan] ), len( tdata[lan] ) )
convert_to_index( tdata, word_to_idx_dict )
# convert_to_index( data, word_to_idx_dict )
data = convert_to_batch( data )
tdata = convert_to_batch( tdata )
for lan in language_list:
    print( len( data[lan] ) )

# print( "Training Model" )
# n = 0
# losses = []
# for epoch in range( 1 ):
#     for i in range( len( data[language_list[0]] ) ):
#         label = K.utils.to_categorical( tdata[language_list[1]][i],
#                                         output_vocab_size )
#         loss = model.train_on_batch( [data[language_list[0]][i],
#                                       data[language_list[1]][i]],
#                                      label )
#         losses.append( loss )
#         print( n, loss )
#         n += 1
#         if n % 5000 == 0:
#             model.save_weights( "models/model_weights_" + str( n ) + ".h5" )
# with open( "losses.txt", "w" ) as f:
#     for loss in losses:
#         f.write( str( loss ) + "\n" )

# data = get_data( "../data/test/" )
# # convert_to_index( data, word_to_idx_dict )
# print( "Generating sentences" )
# translated_sentences = translate_sentences( data[language_list[0]],
#                                             encoder_model, decoder_model,
#                                             max_length,
#                                             word_to_idx_dict, idx_to_word_dict,
#                                             language_list )
# with open( "translated_sentence.txt", "w" ) as f:
#     for sentence in translated_sentences:
#         f.write( sentence + "\n" )

