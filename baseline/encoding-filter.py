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

UNK = False

def filter_vectors( path, language, word_dict ):
    print( "Filtering " + language + "..." )
    files = os.listdir( path + language + "/" )
    for file_name in files:
        fin = io.open( path + language + "/" + file_name, "r", encoding = "utf-8",
                       newline = "\n", errors = "ignore" )
        fout = open( "../data/wv/" + language + "/" + language, "w", encoding = "utf-8" )
        n, d = map( int, fin.readline().split() )
        for line in fin:
            tokens = line.rstrip().split( ' ' )
            if tokens[0] in word_dict:
                fout.write( tokens[0] + ' ' + ' '.join( tokens[1:] ) + '\n' )

def filter_word2vec( path, language_list = ["chinese", "english"], word_dict = None ):
    p = []
    manager = multiprocessing.Manager()
    for language in language_list:
        # Only one file in each folder
        p_lan = multiprocessing.Process( target = filter_vectors,
                                         args = ( path, language, word_dict[language] ) )
        p.append( p_lan )
        p_lan.start()
    for p_lan in p:
        p_lan.join()

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

if __name__ == "__main__":
    language_list = ["chinese", "english"] # [ori_lan, tar_lan]
    data = get_data( "../data/other/", language_list, shuffle = False )
    word_to_idx_dict, idx_to_word_dict = build_dictionary( data, 5 )
    filter_word2vec( "../data/word2vec/", language_list, word_to_idx_dict )
#    filter_vectors( "../data/word2vec/", language_list[0], word_to_idx_dict[language_list[0]] )
#    filter_vectors( "../data/word2vec/", language_list[1], word_to_idx_dict[language_list[1]] )
