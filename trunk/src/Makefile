#all: _align.so
#	gcc -shared align.o align_wrap.o -o _align.so

# Debug ops
#OPTS = -g
# Optimizer ops

OPTS = -O3

all: _cluster_wrap.so

_cluster_wrap.so cluster_wrap.o: cluster.o cluster_wrap.c
	gcc $(OPTS) -c -fpic -I/usr/include/python2.4 \
          -I/usr/lib/python2.4/site-packages/numpy/core/include \
          cluster_wrap.c
	gcc -shared cluster.o cluster_wrap.o -o _cluster_wrap.so

cluster.o:	cluster.c cluster.h
	gcc $(OPTS) -c -fpic cluster.c -o cluster.o

clean:
	rm -f cluster.o cluster_wrap.o cluster.pyc _cluster_wrap.so
