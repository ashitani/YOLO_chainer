import cv2
import numpy as np

def predict(model,im_org):
  im0=cv2.cvtColor(im_org, cv2.COLOR_BGR2RGB)
  im_size=np.shape(im0)
  im0=cv2.resize(im0,(448,448))
  im=np.asarray(im0,dtype=np.float32)/255.0
  im=im*2.0-1.0

  ans=model.predict( im.transpose(2,0,1).reshape(1,3,448,448)).data[0]

  probs=ans[0:980].reshape((7,7,20))     # class probabilities
  confs=ans[980:1078].reshape((7,7,2))   # confidence score for Bounding Boxes
  boxes = ans[1078:].reshape((7,7,2,4))  # Bounding Boxes positions (x,y,w,h)

  p=np.zeros((7,7,2,20))
  for i in range(20):
      for j in range(2):
          p[:,:,j,i]=np.multiply(probs[:,:,i],confs[:,:,j])

  th=0.1

  im_h=im_size[0]
  im_w=im_size[1]

  im_marked=im_org

  classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
             "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person",
             "pottedplant", "sheep", "sofa", "train","tvmonitor"]

  for z in np.argwhere(p>th):
      by,bx,j,i = z
      box=boxes[by,bx,j,:]
      x= (bx+box[0])*(im_w/7)
      y = (by+box[1])*(im_h/7)
      w = box[2]**2.0*im_w
      h = box[3]**2.0*im_h
      im_marked=cv2.rectangle(im_marked, (int(x-w/2), int( y-h/2)),( int(x+w/2), int(y+h/2)),[0,0,255],thickness=2)
      im_marked=cv2.rectangle(im_marked, (int(x-w/2), int( y-h/2)),( int(x-w/2+100), int(y-h/2+20)),[0,0,255],thickness=-1)
      cv2.putText(im_marked, classes[i],(int(x-w/2+5),int(y-h/2+15)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),thickness=2)

  return im_marked