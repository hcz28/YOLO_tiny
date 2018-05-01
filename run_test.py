import tensorflow as tf
import numpy as np
import pdb, cv2, os, time
from argparse import ArgumentParser 
import YOLO_tiny

# default configuration
MODEL = 'weights/YOLO_tiny.ckpt'
PROB_THRESHOLD = 0.1
NMS_THRESHOLD = 0.5

# default configuration for VOC dataset
LABELS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train","tvmonitor"]
ALPHA = 0.1
S = 7
B = 2
C = 20

def build_parser():
    """
    Build parser. 
    """
    parser = ArgumentParser()

    parser.add_argument('--model', type = str, dest = 'model',
            help = 'tiny yolo checkpoint file', metavar = 'MODEL',
            default = MODEL)

    parser.add_argument('--filetype', type = str, dest = 'filetype',
            help = 'filetype: image or video', metavar = 'FILETYPE',
            required = True)

    parser.add_argument('--testdir', type = str, dest = 'testdir', 
            help = 'dictionary for test files', metavar = 'TESTDIR', required = True)

    parser.add_argument('--prob_threshold', type = float, dest = 'prob_threshold',
            help = 'probability threshold for boxes', metavar = 'PROB_THRESHOLD',
            default = PROB_THRESHOLD)

    parser.add_argument('--NMS_threshold', type = float, dest = 'nms_threshold',
            help = 'iou threshold for nms', metavar = 'NMS_THRESHOLD',
            default = NMS_THRESHOLD)

    parser.add_argument('--show', dest = 'show', action = 'store_true', 
            help = 'show result image/video')

    #parser.add_argument('--output_txt', type = str, dest = 'output_txt',
    #        help = 'the filename of output txt', metavar = 'OUTPUT_TXT',
    #        default = None)

    #parser.add_argument('--output_img', type = str, dest = 'output_img',
    #        help = 'the filename of output image/video(only support .avi)', 
    #        metavar = 'OUTPUT_IMG', default = None)

    parser.add_argument('--output_txt', dest = 'output_txt', action = 'store_true',
            help = 'write detection text results')

    parser.add_argument('--output_img', dest = 'output_img', action = 'store_true',
            help = 'write detection image/video(only .avi is supported) results')

    return parser 

def exists(path, msg):
    assert os.path.exists(path), msg 

def list_files(path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    return [os.path.join(path,x) for x in files]

def check_opts(opts):
    #pdb.set_trace()
    exists(opts.model, "checkpoint not found!")
    exists(opts.testdir, "test file not found!")
    assert(opts.filetype == 'image' or opts.filetype == 'video')
    assert opts.prob_threshold >= 0
    assert opts.nms_threshold >= 0
    #if opts.filetype == 'video' and opts.output_img != None:
    #    assert(opts.output_img[-3:] == 'avi')

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    with tf.Session() as sess:
        args = [
                opts.model,
                sess,
                LABELS 
                ]
        kwargs = {
                "prob_threshold": opts.prob_threshold,
                "nms_threshold": opts.nms_threshold,
                "output_txt": None,
                "alpha": ALPHA,
                "S": S,
                "B": B,
                "C": C
                 }
        # build graph
        yolo = YOLO_tiny.YOLOTiny(*args, **kwargs)
        if opts.filetype == 'image':
            test_imgs = list_files(opts.testdir)
            for i, filename in enumerate(test_imgs):
                image = cv2.imread(filename)
                start_time = time.time()
                #image = cv2.imread(opts.filename)
                output, output_txt = yolo.test(image)
                end_time = time.time()
                if opts.show:
                    cv2.imshow('YOLO tiny detection: ' + opts.filename, output)
                    cv2.waitKey(10000)
                tmp = filename.split('/')[-1]
                tmp = tmp.split('.')[0]
                if opts.output_img:
                    cv2.imwrite('result_imgs/'+tmp+'.jpg', output)
                if opts.output_txt:
                    txt = open('result_txts/'+tmp+'.txt','w')
                    for lists in output_txt:
                        txt.write(lists[0])
                    txt.close()
                print("\n----------------YOLO tiny test for " + filename + "-------------------\n")
                print('The total test time is {0}s'.format(end_time - start_time))
                    
        if opts.filetype == 'video':
            start_time = time.time()
            cap = cv2.VideoCapture(opts.testdir)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            if opts.output_img:
                out = cv2.VideoWriter('result_videos/output.avi', fourcc, 25.0, 
                        (int(cap.get(3)), int(cap.get(4))))
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    output,_ = yolo.test(frame)
                    output = np.asarray(output, dtype = np.uint8)
                    if opts.output_img:
                        out.write(output)
                    if opts.show:
                        cv2.imshow('frame', output)
                        cv2.waitKey(1)
                else:
                    break
            cap.release()
            if opts.output_img:
                out.release()
            end_time = time.time()
            print("\n----------------YOLO tiny test for " + filename + "-------------------\n")
            print('The total test time is {0}s'.format(end_time - start_time))

    #if opts.output_img:
    #    print('The result file was saved to ' + opts.output_img)
    #if opts.output_txt:
    #    print('The result file was saved to ' + opts.output_txt)

if __name__ == '__main__':
    main()
