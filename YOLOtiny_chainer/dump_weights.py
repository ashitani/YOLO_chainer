#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from YOLO_tiny_tf import *

c=YOLO_TF()
with c.sess.as_default():
    for v in tf.trainable_variables():
        name = v.name
        ary = v.eval()
        print name, np.shape(ary)
        np.save("npy/"+name, ary)

