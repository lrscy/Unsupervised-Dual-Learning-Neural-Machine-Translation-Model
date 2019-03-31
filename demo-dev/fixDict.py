import os
import sys
import json
import random

with open( "Dicts/numWordDict1", "r" ) as f:
    nwd = json.load( f )

print( list( nwd["chinese"].items() )[:10] )

#nnwd = {"chinese": {}, "english": {}}
#for ( k, v ) in nwd["chinese"].items():
#    nnwd["chinese"][int( k )] = v
#for ( k, v ) in nwd["english"].items():
#    nnwd["english"][int( k )] = v
#
#print( list( nnwd["chinese"].items() )[:10] )
#print( list( nnwd["english"].items() )[:10] )
#
#with open( "Dicts/numWordDict1", "w" ) as f:
#    json.dump( nnwd, f )

#nwd = {"chinese": {}, "english": {}}
#for ( k, v ) in wnd["chinese"].items():
#    nwd["chinese"][v] = k
#for ( k, v ) in wnd["english"].items():
#    nwd["english"][v] = k
#
#with open( "Dicts/wordNumDict1", "w" ) as f:
#    json.dump( wnd, f )
#
#with open( "Dicts/numWordDict1", "w" ) as f:
#    json.dump( nwd, f )
