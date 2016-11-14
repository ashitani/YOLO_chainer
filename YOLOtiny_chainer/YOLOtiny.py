#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

class YOLOtiny(Chain):
    def __init__(self):
        super(YOLOtiny, self).__init__(
            c1  = L.Convolution2D(3,16, ksize=3, pad=1),
            c3  = L.Convolution2D(None,32, ksize=3, pad=1),
            c5  = L.Convolution2D(None,64, ksize=3, pad=1),
            c7  = L.Convolution2D(None,128, ksize=3, pad=1),
            c9  = L.Convolution2D(None,256, ksize=3, pad=1),
            c11 = L.Convolution2D(None,512, ksize=3, pad=1),
            c13 = L.Convolution2D(None,1024, ksize=3, pad=1),
            c14 = L.Convolution2D(None,1024, ksize=3, pad=1),
            c15 = L.Convolution2D(None,1024, ksize=3, pad=1),
            l16 = L.Linear(50176,256),
            l17 = L.Linear(None,4096),
            l19 = L.Linear(None,1470),
        )

    def __call__(self,x):
       return self.predict(x)

    def  predict(self,x):
        h = F.leaky_relu(self.c1(x),slope=0.1)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.leaky_relu(self.c3(h),slope=0.1)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.leaky_relu(self.c5(h),slope=0.1)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.leaky_relu(self.c7(h),slope=0.1)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.leaky_relu(self.c9(h),slope=0.1)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.leaky_relu(self.c11(h),slope=0.1)
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.leaky_relu(self.c13(h),slope=0.1)
        h = F.leaky_relu(self.c14(h),slope=0.1)
        h = F.leaky_relu(self.c15(h),slope=0.1)
        h = F.leaky_relu(self.l16(h),slope=0.1)
        h = F.leaky_relu(self.l17(h),slope=0.1)
        # skip dropout
        h = self.l19(h)

        return h

def loadCoef(filename, shape):
    print "loading",filename
    x=np.loadtxt(filename)
    return x.reshape(shape)

def loadNPY(filename):
    print "loading",filename
    ary=np.load(filename)
    nd=np.ndim(ary)
    if nd==4: #conv weight
        return ary.transpose(3,2,0,1) # Tensorflow/chainer
    if nd==2: # linear weight
        return ary.transpose(1,0)
    else:
        return ary # biases

if __name__ == '__main__':

    c=YOLOtiny()
    im=np.zeros((1,3,448,448),dtype=np.float32)
    c.predict(im)

    COEF_ROOT="./convert/npy/"
    c.c1.W.data  = loadNPY(COEF_ROOT+"Variable:0.npy")
    c.c1.b.data  = loadNPY(COEF_ROOT+"Variable_1:0.npy")
    c.c3.W.data  = loadNPY(COEF_ROOT+"Variable_2:0.npy")
    c.c3.b.data  = loadNPY(COEF_ROOT+"Variable_3:0.npy")
    c.c5.W.data  = loadNPY(COEF_ROOT+"Variable_4:0.npy")
    c.c5.b.data  = loadNPY(COEF_ROOT+"Variable_5:0.npy")
    c.c7.W.data  = loadNPY(COEF_ROOT+"Variable_6:0.npy")
    c.c7.b.data  = loadNPY(COEF_ROOT+"Variable_7:0.npy")
    c.c9.W.data  = loadNPY(COEF_ROOT+"Variable_8:0.npy")
    c.c9.b.data  = loadNPY(COEF_ROOT+"Variable_9:0.npy")
    c.c11.W.data = loadNPY(COEF_ROOT+"Variable_10:0.npy")
    c.c11.b.data = loadNPY(COEF_ROOT+"Variable_11:0.npy")
    c.c13.W.data = loadNPY(COEF_ROOT+"Variable_12:0.npy")
    c.c13.b.data = loadNPY(COEF_ROOT+"Variable_13:0.npy")
    c.c14.W.data = loadNPY(COEF_ROOT+"Variable_14:0.npy")
    c.c14.b.data = loadNPY(COEF_ROOT+"Variable_15:0.npy")
    c.c15.W.data = loadNPY(COEF_ROOT+"Variable_16:0.npy")
    c.c15.b.data = loadNPY(COEF_ROOT+"Variable_17:0.npy")
    c.l16.W.data = loadNPY(COEF_ROOT+"Variable_18:0.npy")
    c.l16.b.data = loadNPY(COEF_ROOT+"Variable_19:0.npy")
    c.l17.W.data = loadNPY(COEF_ROOT+"Variable_20:0.npy")
    c.l17.b.data = loadNPY(COEF_ROOT+"Variable_21:0.npy")
    c.l19.W.data = loadNPY(COEF_ROOT+"Variable_22:0.npy")
    c.l19.b.data = loadNPY(COEF_ROOT+"Variable_23:0.npy")

    serializers.save_npz('YOLOtiny.model',c)
