import threading
import time
import queue

a = 0
b = 0
que = queue.Queue( 5 )

def f():
    global a, b, que
    while a < 10:
        if not que.full():
            a += 1
            b += 2
            time.sleep( 1 )
            print( "1-", a, b )
            que.put( b, False )

def g():
    global a, b, que
    while a < 10 or not que.empty():
        if not que.empty():
            c = que.get( False )
            time.sleep( 1 )
            que.task_done()
            print( "2-", c )

print( a, b )
t1 = threading.Thread( target = f )
t2 = threading.Thread( target = g )
t1.start()
t2.start()
t1.join()
t2.join()
