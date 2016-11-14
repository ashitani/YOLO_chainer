#!/usr/bin/env python

from YOLOtiny_chainer import *
import cv2
import sys
import os

if len(sys.argv)!=2:
    print "Usage: python replay.py filename"
    exit(-1)
filename=sys.argv[1]

print "Loading model"
model=YOLOtiny()
serializers.load_npz('YOLOtiny_chainer/YOLOtiny.model',model)

im_org=cv2.imread(filename)
im_marked = predict(model,im_org)

path, ext = os.path.splitext(os.path.basename(filename))
outfile=path+"_marked.png"
cv2.imwrite(outfile,im_marked)
print "Result was written to ",outfile
