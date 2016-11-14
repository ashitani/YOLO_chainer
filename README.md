# YOLO_chainer

Chainer implementation of [YOLO](http://pjreddie.com/darknet/yolo/).

Actually, CNN part is only a transportation from [YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow) to chainer.

![example_image](https://github.com/ashitani/YOLO_chainer/sample/bicycle_marked.png)

# Usage

Download pretrained chainer model from [here](https://docs.google.com/uc?id=0B0SWRybIpnRFU1ZpbXJZRjIydGM&export=download) and copy it to YOLOtiny_chainer folder.

After that:

```
python replay.py image_file
```

# Converting tensorflow model to chainer

See YOLOtiny_chainer folder.

# License

MIT
