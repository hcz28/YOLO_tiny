# YOLO tiny

Implementation of YOLO tiny (version 1) with tensorflow. This project only supports test with image or video. Please use darknet for training.

## Usage

- model: tiny yolo checkpoint
- filetype: 'image' or 'video'
- testdir: dictionary for test files
- prob_threshold: probability threshold for bounding boxes, default 0.1
- NMS_threshold: IOU threshold for NMS, default 0.5
- show: show the output image/video
- output_txt: write detection results into text
- output_img: write detection results into image/video (only .avi is supported)

e.g.
```
python run_test --model='weigts/YOLO_tiny.ckpt' --filetype='image' --testdir='test_imgs' --output_txt --output_img --show
```

## Results

![dog](https://github.com/hcz28/YOLO_tiny/blob/master/result_imgs/dog.jpg?raw=true)

## Requirements

- Python 3.6
- Tensorflow 1.6
- opencv-python 3.4

## References

- [gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)

More details in the future...
