import time
import tqdm
import multiprocessing
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial


def add(x, y):
    time.sleep( 2 )
    return x, y

def add_wrap(args):
    return add(*args)

if __name__ == "__main__":
    a = [i for i in range( 20 )]
    b = [i for i in range( 20, 40 )]
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = cores)
    c = []
    for y in tqdm.tqdm( pool.imap_unordered( add_wrap, zip( a, b ) ) ):
        c.append( y )
    #print(pool.map(add, [a, b]))
    #close the pool and wait for the worker to exit
    pool.close()
    pool.join()
    print( c )
