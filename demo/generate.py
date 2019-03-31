import os
import sys
import random
import json
import numpy as np
import tensorflow as tf
import keras as K
import keras.backend.tensorflow_backend as KTF
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Lambda

K.backend.clear_session()
KTF.set_session( tf.Session( config = tf.ConfigProto( device_count = {'gpu':0} ) ) )

"""Import dictionaries from files

Read word->idx dictionary and idx->word dictionary from files.
Formats are all listed in returns.

Args:
    path: a string contains path which saves all dictionaries.
    languages: a list of languages that all dictionary contains.

Returns:
    word_to_idx_dict: a dictionary converts word to index. Its structure is:

                      {language A: {word A: index A, word B: ..., ...},
                       language B: ..., ...}.

    idx_to_word_dict: a dictionary converts index to word. Its structure is:

                      {language A: {index A: word A, index B: ..., ...},
                       language B: ..., ...}.
"""
def import_dictionaries( path, languages = ["chinese", "english"] ):
    print( "Importing data from " + path + "..." )
    idx_to_word_dict_path = path + "idx_word_dict.json"
    word_to_idx_dict_path = path + "word_idx_dict.json"
    idx_to_word_dict = {languages[0]:{}, languages[1]:{}}
    word_to_idx_dict = {languages[0]:{}, languages[1]:{}}

    with open( idx_to_word_dict_path, "r" ) as f:
        t_idx_to_word_dict = json.load( f )
        for lan in languages:
            for ( k, v ) in t_idx_to_word_dict[lan].items():
                idx_to_word_dict[lan][int( k )] = v

    with open( word_to_idx_dict_path, "r" ) as f:
        t_word_to_idx_dict = json.load( f )
        for lan in languages:
            for ( k, v ) in t_word_to_idx_dict[lan].items():
                word_to_idx_dict[lan][k] = int( v )

    return word_to_idx_dict, idx_to_word_dict

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
    encoder           = LSTM( hidden_dim, return_state = True, name = name + "_encoder_lstm" )
    encoder_input     = Input( shape = ( None, ), name = name + "_encoder_input" )
    # change when using pre-trained embedding trainable= False
    encoder_input_emb = embedding_encoder( encoder_input )
    _, state_h, state_c = encoder( encoder_input_emb )
    state_encoder     = [state_h, state_c]

    # Decoder
    decoder           = LSTM( hidden_dim, return_state = True,
                              return_sequences = True, name = name + "_decoder_lstm" )
    decoder_input     = Input( shape = ( None, ), name = name + "_decoder_input" )
    decoder_input_emb = embedding_decoder( decoder_input )
    decoder_outputs, _, _   = decoder( decoder_input_emb, initial_state = state_encoder )
    decoder_dense     = Dense( output_vocab_size, activation = "softmax", name = name + "_decoder_output" )
    decoder_outputs   = decoder_dense( decoder_outputs )

    # Build model
    model = Model( inputs = [encoder_input, decoder_input], outputs = decoder_outputs, name = name )
    model.compile( optimizer = 'adam', loss = "categorical_crossentropy" )

    # Build encoder_model
    encoder_model = Model( inputs = encoder_input,
                           outputs = state_encoder )

    decoder_state_h = Input( shape = ( 128, ), name = name + "_state_h" )
    decoder_state_c = Input( shape = ( 128, ), name = name + "_state_c" )
    decoder_state_input = [decoder_state_h, decoder_state_c]
    decoder_output, decoder_state_h, decoder_state_c = decoder( decoder_input_emb,
                                                                initial_state = decoder_state_input )
    decoder_state  = [decoder_state_h, decoder_state_c]
    decoder_output = decoder_dense( decoder_output )
    decoder_model  = Model( inputs = [decoder_input] + decoder_state_input,
                            outputs = [decoder_output] + decoder_state )

    return model, encoder_model, decoder_model

"""Convert word to index

Results are written into original data.

Args:
    data: a dictionary of data sentences of each language. Its structure is:
    
          {language A: [[word1, word2, ...], [...], ...], language B: ...}.

    word_to_idx_dict: a dictionary converts word to index. Its structure is:

                      {language A: {word A: index A, word B: ..., ...},
                       language B: ..., ...}.

    language: a list of language indicating which language is in data and
              dictionary. Default value is ["chinese", "english"]

Returns:
    None.
"""
def to_index( data, word_to_idx_dict, language = ["chinese", "english"] ):
    print( "Converting words to indexes..." )
    for lan in language:
        for sentence in data[lan]:
            for i in range( len( sentence ) ):
                if sentence[i] in word_to_idx_dict[lan]:
                    sentence[i] = word_to_idx_dict[lan][sentence[i]]
                else:
                    sentence[i] = word_to_idx_dict[lan]["<UNK>"]

"""Generate sentences based on given sentences

Args:
    data: a dictionary of dev data sentences of each language. Its structure is:
    
          {language A: [[word1, word2, ...], [...], ...], language B: ...}.

    encoder_model: encoder part of seq2seq model.
    decoder_model: decoder (generate) part of seq2se1 model.
    max_len: a interger represents the max length of generated (translated)
             sentence.
    idx_to_word_dict: a dictionary converts index to word. Its structure is:

                      {language A: {index A: word A, index B: ..., ...},
                       language B: ..., ...}.

    language: a list of language indicating which language is in data and
              dictionary. Default value is ["chinese", "english"]

Returns:
    ret_sentences: a list of generated (translated) sentences.
"""
def generate_sentences( data, encoder_model, decoder_model, max_len,
                        idx_to_word_dict, languages = ["chinese", "english"] ):
    print( "Generating sentences..." )
    ret_sentences = []
    cnt = 0
    for sentence in data[languages[0]]:
        decoder_state = encoder_model.predict( sentence )
        decoder_input = np.zeros( ( 1, 1 ) )
        decoder_input[0, 0] = 1 # <S>:1
        gen_sentence = []
        stop_sign = False
        while not stop_sign:
            decoder_output, decoder_state_h, decoder_state_c = \
                decoder_model.predict( [decoder_input] + decoder_state )
            token = np.argmax( decoder_output[0, -1, :] )
            while token < 4:
                decoder_output[0, -1, token] = -1
                token = np.argmax( decoder_output[0, -1, :] )
            word = idx_to_word_dict[languages[1]][token]
            gen_sentence.append( word )
            if word == "</S>" or len( gen_sentence ) > max_len:
                stop_sign = True
            decoder_state = [decoder_state_h, decoder_state_c]
            decoder_input = np.zeros( ( 1, 1 ) )
            decoder_input[0, 0] = token
        ori_sentence = [idx_to_word_dict[languages[0]][wi] for wi in sentence]
        ret_sentences.append( [' '.join( ori_sentence[1:-1] ), ' '.join( gen_sentence[1:-1] )] )
        cnt += 1
        print( "No. of translated sentences: ", cnt, end = "\r" )
    print( "" )
    return ret_sentences

model_name = "demo"
language_list = ["chinese", "english"] # [original, target]

_, devData = getTrainData()
# 100 sentences
devData[language_list[0]] = devData[language_list[0]][:100]
devData[language_list[1]] = devData[language_list[1]][:100]
word_to_idx_dict, idx_to_word_dict = import_dictionaries( "Dicts/" )
ivs = len( word_to_idx_dict[language_list[0]] )
ovs = len( word_to_idx_dict[language_list[1]] )

print( "Bulding model and  predition model..." )
model, encoder_model, decoder_model = simpleSeq2Seq( output_vocab_size = ovs, input_vocab_size = ivs, name = model_name )
model.summary()
print( "Loading weights..." )
model.load_weights( "Models/model_weights_24000.h5" )

#encoder_model, decoder_model = pred_model( model, name = model_name )

to_index( devData, word_to_idx_dict, language = language_list )
g_sentences = generate_sentences( devData, encoder_model, decoder_model, 50,
                                  idx_to_word_dict, language_list )
print( "Writing to file..." )
with open( "translated_sentences.txt", "w" ) as f:
    for g_sentence in g_sentences:
        f.write( g_sentence[0] + "\n" + g_sentence[1] + "\n" )
