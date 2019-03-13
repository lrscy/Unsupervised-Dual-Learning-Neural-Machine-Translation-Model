
# coding: utf-8

# In[ ]:


# Word-Level Language Modeling
# -*- coding: utf-8 -*-

import os
import re
import sys
import math
import random
import argparse


# In[ ]:


"""Get all files name under path

Args:
    path: folder path to retrieve files' name.
    ratio: propotion of training data. Default value is 1 (100%).
    shuffle: a boolean value. TRUE: shuffle list; False: order list.

Returns:
    filesName[:train]: a list of all files end with ".txt" for training set. For example:

    ["dir/a.txt", "dir/b.txt"].

    filesName[train:]: a list of all files end with ".txt" for held-out set. For example:

    ["dir/a.txt", "dir/b.txt"].
"""
def getFilesName( path, ratio = 1, shuffle = False ):
    print( "Retrieving files name from folder %s..." % ( path ) )
    filesName = []
    files = os.listdir( path )
    for file in files:
        name = '/'.join( [path, file] )
        filesName.append( name )
    if shuffle:
        random.shuffle( filesName )
    else:
        filesName.sort()
    total = len( filesName )
    train = int( total * ratio )
    return filesName[:train], filesName[train:]

"""Preprocess data

Find words in training set that appear ≤ 5 times as “UNK”.

Note: The function will figure out all words which are need to be replaced by "UNK"
      and they will be replaced when building n-gram word-level language model.

Args:
    contents: a list of content to be processed. Content is also a list.

Returns:
    repc: a list of words that need to be replaced with "<UNK>". For example:
    
    "[word a, word b, word c]"
"""
def unk( contents ):
    d = {}
    for content in contents:
        for w in content:
            if w not in d:
                d[w] = 0
            d[w] += 1
    repw = {}
    for ( k, v ) in d.items():
        if v <= 3:
            if k not in repw:
                repw[k] = 0
            repw[k] += 1
    return repw


# In[ ]:


# Build Word-Level Language Model
"""Generate n-gram dictionary.

Generate n-gram dictionary based on fed string and n.

Args:
    contents: a list of content used to calculate ngrams.
    n: n-gram.
    d: language model corresponds to n-gram.

Returns:
    None.
"""
def ngrams( contents, n, d ):
    for content in contents:
        for i in range( n - 1, len( content ) - 1 ):
            k = ' '.join( content[i - n + 1:i + 1] )
            if k not in d["c"]:
                d["c"][k] = 0
            d["c"][k] += 1
            d["t"] += 1

"""Build language model

Preprocess files and across all files in the directory (counted together), report the 
unigram, bigram, and trigram words counts.

Args:
    content: a list contains content needed to be processed.

Returns:
    lm: a dictionary of language model when savePath equals empty string. Its structure is:
    
    {"unigram": {"c": unigram, "t": total unigram words},
     "bigram" : {"c": bigram,  "t": total bigram  words},
     "trigram": {"c": trigram, "t": total trigram words}}.
"""
def LM( contents ):
    print( "Building language modeling..." )
    lm = {"unigram": {"c": {}, "t": 0},
          "bigram" : {"c": {}, "t": 0},
          "trigram": {"c": {}, "t": 0}}
    ngram = ["unigram", "bigram", "trigram"]
    
    # Calculate unigram, bigram, and trigram
    print( "Calculating n-grams..." )
    ngrams( contents, 1, lm["unigram"] )
    ngrams( contents, 2, lm["bigram" ] )
    ngrams( contents, 3, lm["trigram"] )
    return lm

"""Build Language Model

Across all files in the directory (counted together), report the unigram, bigram, and trigram
words counts and save them in seperate files.

Args:
    trainDataPath: train data path.
    encoding: train data files' encoding
    savePath: path to save language model.
    ratio: the proportion of the real training set comparing to whole training set

Returns:
    None
"""
def buildLM( trainDataPath = "./train", encoding = "Latin-1", savePath = "./lm", ratio = 1 ):
    ngram = ["unigram", "bigram", "trigram"]
    trainFiles, heldOutFiles = getFilesName( trainDataPath )
    # preprocess data and find UNK
    print( "Counting for finding UNK.")
    contents = []
    for fileName in trainFiles:
        with open( fileName, 'r', encoding = encoding ) as f:
            content = []
            line = f.readline()
            while line:
                contents.append( line.split() )
                line = f.readline()
    repw = unk( contents )
    if len( repw ):
        for content in contents:
            for i in range( len( content ) ):
                if content[i] in repw:
                    content[i] = "<UNK>"
    lm = LM( contents )
    if not os.path.isdir( savePath ):
        os.makedirs( savePath )
    for name in ngram:
        print( name )
        with open( savePath + "/" + name, "w", encoding = encoding ) as f:
            f.write( str( lm[name]["t"] ) + "\n" )
            for ( k, v ) in lm[name]["c"].items():
                f.write( k + " " + str( v ) + "\n" )


# In[ ]:


# buildLM( trainDataPath = "../../Data/train/english", encoding = "UTF-8", savePath = "./lm/english" )


# In[ ]:


# Apply Add-Lambda smoothing function on language model.
"""Load language model

Load language model from folder "lm" and save them into dictionary "lm".

Args:
    loadPath: language model load path.
    encoding: language model files' encoding

Returns:
    lm: a dictionary of language model. Its structure is:
    
    {"unigram": {"c": unigram, "t": total unigram words},
     "bigram" : {"c": bigram,  "t": total bigram  words},
     "trigram": {"c": trigram, "t": total trigram words}}.
"""
def loadLM( loadPath = "./lm", encoding = "utf-8" ):
    lm = {}
    ngram = ["unigram", "bigram", "trigram"]
    # load unigram, bigram, and trigram
    for name in ngram:
        with open( loadPath + "/" + name, "r", encoding = encoding ) as f:
            ngram = {}
            total = 0
            line = f.readline()
            while line:
                kv = line.split( ' ' )
                if len( kv ) > 1:
                    k = ' '.join( kv[:-1] )
                    v = kv[-1]
                    ngram[k] = int( v )
                else:
                    total = int( kv[0] )
                line = f.readline()
            lm[name] = {"c": ngram, "t": total}
    return lm

"""Calculate perplexity

PP(W) = P(w_1w_2 ... w_n)^(-1/n)
      = 2^{-1 / n * sum_{i=1:n}(log2(LM(w_i|w_{i-2}w_{i-1})))}

Note: Since here is no <SOS> and <EOS> in language model, n would be the length of
      the content - 2.

Args:
    content: a list of words in a sentence.
    lm: a dictionary contains language model. Its structure is:
    
    {"unigram": {"c": unigram, "t": total unigram words},
     "bigram" : {"c": bigram,  "t": total bigram  words},
     "trigram": {"c": trigram, "t": total trigram words}}.

    **kwargs:
        func: smoothing function name on calculating P(w_i|w_{i-2}w_{i-1}), including
              func = "Interplotation" and func = "AddLambda".
        
        lambdas: a dictionary of lambda for interplotation or addLambda. Its structure is:
    
        {1: lambdaForUnigram, 2:lambdaForBigram, 3:lambdaForTrigram}
        
        When using addLambda function, only need to feed one specific lambda.
    
Returns:
    ppw: a double number represents the perplexity of the content.

Raise:
    KeyError: an error when trying to find smoothing function.
"""
def perplexity( content, lm, **kwargs ):
    length = len( content )
    log2p = 0
    if( length <= 2 ):
        raise Exception( "Too short content." )
    if "func" in kwargs:
        if kwargs["func"] == "AddLambda":
            for i in range( length - 2 ):
                p = addLambda( lm, kwargs["lambdas"], content[i:i + 3] )
                log2p += math.log2( p )
        else:
            raise Exception( "Cannot find the smoothing function." )
    else:
        raise Exception( "No smoothing function." )
    log2p *= -1 / ( length - 2 )
    ppw = 2 ** log2p
    return ppw

"""Add lambda smoothing

P(w_{n}|w_{n-1}w_{n-2}) = ( c(w_{n-1}w_{n-2}, w_{n}) + lambda ) /
                            ( c(w_{n-1}w_{n-1}) + lambda * V )

Args:
    lm: a dictionary of language model. Its structure is:
    
    {"unigram": {"c": unigram, "t": total unigram words},
     "bigram" : {"c": bigram,  "t": total bigram  words},
     "trigram": {"c": trigram, "t": total trigram words}}.
    
    lambdas: a generator of lambdas which generate a dictionary of lambdas
             for unigram, bigram, trigram each time. For example:
    
    {1:0.1, 2:0.1, 3:0.8}
    
    s: a list of words wating for calculating unigram, bigram, and trigram.

Returns:
    p: a double number represents the final probability of P(w_{n}|w_{n-2}w_{n-1}).
"""
def addLambda( lm, lambdas, s ):
    s = s[-1]
    if s not in lm["trigram"]["c"]:
        cnt1 = 0
    else:
        cnt1 = lm["trigram"]["c"][s]
    s = ' '.join( s[:-1] )
    if s not in lm["bigram"]["c"]:
        cnt2 = 0
    else:
        cnt2 = lm["bigram"]["c"][s]
    p = ( cnt1 + lambdas[3] ) / ( cnt2 + len( lm["bigram"]["c"] ) * lambdas[3] )
    return p

"""Main function for problem 3.3

Calculate the perplexity for each 
le in the test set using linear interpolation smoothing
method.

Args:
    trainDataPath: train data path.
    encoding: train data files' encoding
    savePath: path to save language model. If it equals to empty string, the function returns
              language model.
    testDataPath: test data path.
    ratio: the proportion of the real training set comparing to whole training set.

Returns:
    None.
"""
def addLambdaPPW( lmPath = "./lm", encoding = "Latin-1", savePath = "./3_3",
                  testDataPath = "./test" ):
    # Get new language model
    lm = loadLM( lmPath, encoding )
    lambdas = {3: 0.1}
    # File-PPW pair dictionary
    dfp = {}
    filesName, _ = getFilesName( testDataPath )
    for fileName in filesName:
        with open( fileName, 'r', encoding = encoding ) as f:
            line = f.readline()
            n = 0
            while line:
                content = line.strip().split()
                n += 1
                ppw += perplexity( content, lm, func = "AddLambda", lambdas = lambdas )
                line = f.readline()
            ppw /= n
        dfp[fileName] = ppw
    fps = sorted( dfp.items(), key = lambda x: x[1], reverse = True )
    with open( savePath + "/" + "filesPerplexity-addLambda.txt", 'w' ) as f:
        for fp in fps:
            f.write( fp[0].split( '/' )[-1] + ", " + str( fp[1] ) + "\n" )


# In[ ]:


# Test
def test():
    # Build Word-Level Language Model
    print( "Building Word-Level Language Model..." )
    buildLM( trainDataPath = "./train", encoding = "Latin-1", savePath = "./lm",
             ratio = 1 )
    # Apply Add-Lambda smoothing function on language model
    print( "Applying Add-Lambda smoothing function on language model..." )
    addLambdaPPW( lmPath = "./lm", encoding = "Latin-1", savePath = "./save",
                  testDataPath = "./test" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    groupsmoothing = parser.add_mutually_exclusive_group()
    groupsmoothing.add_argument( "-t", "--test", help = "Test on all funcitions", action="store_true" )
    groupsmoothing.add_argument( "-b", "--build", help = "Build a Word-level n-gram language model", 
                                 action="store_true" )
    groupsmoothing.add_argument( "-a", "--addLambda",
                                 help = "Apply Add-Lambda smoothing function on language model",
                                 action="store_true" )
    parser.add_argument( "-e", "--encoding", type = str,
                         help = "Encoding of files", default = "Latin-1" )
    parser.add_argument( "-r", "--ratio", type = float,
                         help = "proportion of real train data files in train data path",
                         default = 1.0 )
    parser.add_argument( "--trainPath", type = str, help = "Path that train data stores",
                         default = "./train" )
    parser.add_argument( "--testPath",  type = str, help = "Path that test data stores",
                         default = "./test" )
    parser.add_argument( "--savePath",  type = str, help = "Path that function result will save at",
                         default = "./save" )
    parser.add_argument( "--lmPath",    type = str, help = "Path that language model stores",
                         default = "./lm" )
    args = parser.parse_args()

    if args.test:
        test()
    if args.build:
        buildLM( args.trainPath, args.encoding, args.savePath, args.ratio )
    if args.addLambda:
        addLambdaPPW( args.lmPath, args.encoding, args.savePath, args.testPath )

