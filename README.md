# YOLO tiny

Implementation of YOLO tiny (version 1) with tensorflow. This project only supports test with image or video. Please use darknet for training.

## Usage

- model: tiny yolo checkpoint
- filetype: 'image' or 'video'
- filename: filename
- show: whether to show the output image/video
- output_txt: filename for output txt
- output_img: filename for output image/video (only .avi is supported for video)

e.g.
```
python run_test --model='weigts/YOLO_tiny.ckpt' --filetype='image' --filename='test/dog.jpg' --output_txt='results/dog.txt' --output_img='results/dog.jpg' --show
```

## Results

![dog](https://github.com/hcz28/YOLO_tiny/blob/master/results/dog.jpg?raw=true)

## Requirements

- Python 3.6
- Tensorflow 1.6
- opencv-python 3.4

## References

- Most of the code comes from [gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)

More details in the future...
