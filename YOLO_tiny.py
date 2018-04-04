import numpy as np
import tensorflow as tf
import pdb, cv2

class YOLOTiny:
    """
    Define the network for tiny YOLO, which has 9 convolution layers and 3 fully connected
    layers.

    Args:
        model_path: Path for checkpoint file.
        session: Session. 
        labels: Labels for each class.
        output_txt: Path to save the result in txt.
        alpha: Parameter for leaky_relu. Default 0.1. 
        S: Number of cells in each direction. Default 7. 
        B: Number of bounding boxes for each cell. Default 2.
        C: Number of classes. Default 20 for VOC dataset. 
        prob_threshold: The thresold used to filter probilities for each box.
        nms_threshold: The threshold used in non-maximum suppression.

    """
    def __init__(self, model_path, session, labels, prob_threshold = 0.1, 
            nms_threshold = 0.5, output_txt = None, alpha = 0.1, S = 7, B = 2, C = 20):
        """
        Initialization for arrtibutes.

        """
        self.model_path = model_path
        self.sess = session 
        self.labels = labels
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold 
        self.output_txt = output_txt
        self.alpha = alpha 
        self.S = S
        self.B = B
        self.C = C
        self._build_network()

    def _build_network(self):
        """
        Build YOLO tiny network.
        """
        self.x = tf.placeholder('float32',[None, 448, 448, 3])
        conv_1 = self._conv2d(self.x, 16, 3, 1)
        pool_1 = self._pool(conv_1, 2, 2)
        conv_2 = self._conv2d(pool_1, 32, 3, 1)
        pool_2 = self._pool(conv_2, 2, 2)
        conv_3 = self._conv2d(pool_2, 64, 3, 1)
        pool_3 = self._pool(conv_3, 2, 2)
        conv_4 = self._conv2d(pool_3, 128, 3, 1)
        pool_4 = self._pool(conv_4, 2, 2)
        conv_5 = self._conv2d(pool_4, 256, 3, 1)
        pool_5 = self._pool(conv_5, 2, 2)
        conv_6 = self._conv2d(pool_5, 512, 3, 1)
        pool_6 = self._pool(conv_6, 2, 2)
        conv_7 = self._conv2d(pool_6, 1024, 3, 1)
        conv_8 = self._conv2d(conv_7, 1024, 3, 1)
        conv_9 = self._conv2d(conv_8, 1024, 3, 1)
        fc_1 = self._fc(conv_9, 256, flat = True, linear = False)
        fc_2 = self._fc(fc_1, 4096, flat = False, linear = False)
        num_outputs = self.S * self.S * (self.C + self.B * 5)
        fc_3 = self._fc(fc_2, num_outputs, flat = False, linear = True)
        self.pred = fc_3 

    def _conv2d(self, x, out_channels, filter_size, stride_size):
        """
        Convolutional layers.
        """
        in_channels = int(x.shape[3])
        weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels,
            out_channels]))
        biases = tf.Variable(tf.constant(0.1, shape=[out_channels]))
        output = tf.nn.conv2d(x, weights, strides = [1, stride_size, stride_size, 1],
                padding = 'SAME')
        output = tf.add(output, biases)
        output = tf.nn.leaky_relu(output, self.alpha)
        return output
    
    def _pool(self, x, filter_size, stride_size):
        """
        Pooling Layers.
        """
        output = tf.nn.max_pool(x, ksize = [1, filter_size, filter_size, 1], strides
                = [1, stride_size, stride_size, 1], padding = 'SAME')
        return output

    def _fc(self, x, num_neurons, flat = False, linear = False):
        """
        Fully connected layers.

        Args:
            flat: True if the input x is a 4-D tensor, i.e. the output of a convolutional
            layer. Default: False. 
            linear: False if add a leaky relu layer. Default: False. 
        """
        x_shape = [i.value for i in x.shape]
        if flat:
            dim = x_shape[1] * x_shape[2] * x_shape[3]
            # need a transpose before resize
            x_transposed = tf.transpose(x, [0, 3, 1, 2])
            x_processed = tf.reshape(x_transposed, [-1, dim])
        else:
            dim = x_shape[1]
            x_processed = x
        weights = tf.Variable(tf.truncated_normal([dim, num_neurons], stddev = 0.1))
        biases = tf.Variable(tf.constant(0.1, shape = [num_neurons]))
        output = tf.add(tf.matmul(x_processed, weights), biases)
        if not linear:
            output = tf.nn.leaky_relu(output, self.alpha)
        return output

    def test(self, image):
        """
        Test.

        Args:
            image: 3-D array.

        Returns:
            The result image. 3-D array. 
        """
        # get the output of network, i.e. the array with shape (1, 1470)
        self.img = image
        self.height, self.width, _ = self.img.shape 
        self._preprocess_image()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.model_path)
        output = self.sess.run(self.pred, feed_dict = {self.x: self.input})
        results = self._interpret_output(output)
        result_img = self._draw_rectangle(results)
        return result_img

    def _preprocess_image(self):
        """
        Preprocess of image, resize, normalization, etc. Return the inputs of the network. 
    
        Args:
            image: 3-D array.  
    
        Returns:
            4-D array. 
        """
        img = cv2.resize(self.img, (448, 448))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv uses the BGR channel
        img = np.asarray(img, dtype = np.float32)
        img = img / 255.0 * 2 - 1;
        self.input = np.reshape(img, (1,) + img.shape)
        
    def _draw_rectangle(self, results):
        """
        Draw rectangles on the file.
        """
        
        colors = [
                    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], 
                    [0,255,255], [255, 125, 0], [125, 255,0], [0, 255,125], [235,245,223],
                    [186,212,170],[212,212,170],[237,180,88],[232,135,30],[232,226,136],
                    [125,206,130],[60,219,211],[143,54,134],[144,156,194],[239,208,31]
                ]
        img_cp = self.img.copy()
        if self.output_txt != None:
            txt = open(self.output_txt, 'w')
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])//2
            h = int(results[i][4])//2
            label = self.labels[results[i][0]]
            cv2.rectangle(img_cp, (x-w, y-h), (x+w, y+h), colors[results[i][0]], 2)
            cv2.rectangle(img_cp, (x-w-1, y-h-20), (x-w+12*len(label), y-h),
                    colors[results[i][0]], -1)
            cv2.putText(img_cp, label, (x-w+2, y-h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 2)
            if self.output_txt != None:
                txt.write(label + ',' + str(x) + ',' + str(y) + ',' + str(w) + ','
                        + str(h) + ',' + str(results[i][5]) + '\n')
        #if self.save_img:
        #    cv2.imwrite(self.result_img, img_cp)
        #if self.show_img:
        #    cv2.imshow('YOLO tiny detection', img_cp)
        #    cv2.waitKey(10000)
        if self.output_txt != None:
            txt.close()
        return img_cp
    
    def _interpret_output(self, output):
        """
        Interpret the output of network.
    
        Args:

            output: The output of tiny yolo. 2-D numpy array.
                    number of grids: 7 * 7
                    number of bounding box for each grid: 2
                    number of classes: 20
                    number of parameters for each bounding box: 5
                    The total number of output is 7 * 7 * (20 + 2 * 5) = 1470
    
                    More about the bounding box location:
                    (x, y): the center of the box relative to the bounds of the grid cell
                    (w, h): the width and height of the box relative to the whole image
    
        Returns:
            A 2-D array. Each row contains the class idx, 4 bounding box locations and 1
            probability.
        """
    
        #pdb.set_trace()
        
        output = output[0]
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        confidence = np.reshape(output[980: 1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7,7,2,4))
        
        # offset for boxes
        offset = np.arange(7) # (7, )
        offset = np.tile(offset, (7, 1)) # (7, 7)
        offset = np.reshape(offset, (7, 7, 1)) # (7, 7, 1)
        offset = np.tile(offset, (1, 1, 2)) # (7, 7, 2)
        
        # get the location of the box relative to the whole image
        boxes[:,:,:,0] += offset
        boxes[:,:,:,1] += np.transpose(offset, (1, 0, 2))
        boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
        boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2], boxes[:,:,:,2]) # sqrt in loss function
        boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3], boxes[:,:,:,3]) 
        
        boxes[:,:,:,0] *= self.width # relative to original image
        boxes[:,:,:,1] *= self.height 
        boxes[:,:,:,2] *= self.width
        boxes[:,:,:,3] *= self.height
        
        # calculate the probability for each bounding box
        for i in range(2):
            for j in range(20):
                probs[:,:,i,j] = np.multiply(class_probs[:,:,j], confidence[:,:,i])
        
        # threshold the probability
        #pdb.set_trace()
        filter_mat_probs = np.array(probs > self.prob_threshold, dtype='bool') # (7, 7, 2, 20)
        filter_mat_boxes = np.nonzero(filter_mat_probs) # tuple
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], 
                filter_mat_boxes[2]] # (# of remaining boxes ,4)
        probs_filtered = probs[filter_mat_probs] # (# of remaining boxes, )
        # get the class with maximum probability for each bounding box
        classes_num_filtered = np.argmax(filter_mat_probs, axis = 3)[filter_mat_boxes[0],
                filter_mat_boxes[1], filter_mat_boxes[2]] # shape (# of remaining boxes, ).
    
        # non-maximum suppression 
        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
    
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i+1, len(boxes_filtered)):
                if self._iou(boxes_filtered[i], boxes_filtered[j]) > self.nms_threshold:
                    probs_filtered[j] = 0.0
    
        filter_iou = np.array(probs_filtered>0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]
    
        results = []
        for i in range(len(boxes_filtered)):
            results.append([classes_num_filtered[i], boxes_filtered[i][0], boxes_filtered[i][1],
                boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])
        #pdb.set_trace() 
        return results
    
    def _iou(self, box1, box2):
        """
        更改IOU计算方式
        Calculate the intersection of union.
        """
         
        box1_x1 = box1[0] - 0.5*box1[2]
        box1_x2 = box1[0] + 0.5*box1[2]
        box1_y1 = box1[1] - 0.5*box1[3]
        box1_y2 = box1[1] + 0.5*box1[3]
        
        box2_x1 = box2[0] - 0.5*box2[2]
        box2_x2 = box2[0] + 0.5*box2[2]
        box2_y1 = box2[1] - 0.5*box2[3]
        box2_y2 = box2[1] + 0.5*box2[3]
    
        width = min(box1_x2, box2_x2) - max(box1_x1, box2_x1)
        height = min(box1_y2, box2_y2) - max(box1_y1, box2_y1)
    
        # if the two boxes are nested, set iou = 1
        if box1_x1 >= box2_x1 and box1_x2 <= box2_x2 and box1_y1 >= box2_y1 and box1_y2 <=\
                box2_y2:
            return 1
        if box2_x1 >= box1_x1 and box2_x2 <= box1_x2 and box2_y1 >= box1_y1 and box2_y2 <=\
                box1_y2:
            return 1
    
        # calculate the area of intersection
        if width < 0 or height < 0:
            intersection = 0
        else:
            intersection = width * height
        iou = intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)
        return iou
