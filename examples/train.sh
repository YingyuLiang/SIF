cd ../src
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python2.7 train.py $@