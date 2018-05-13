1. Download YOLO-tiny code
    ```
    git clone https://github.com/hcz28/YOLO_tiny.git
    ```
2. Download VOC2007 test dataset in test_imgs
    ```
    wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar
    ```
3. Run YOLO-tiny
    ```
    python run_test.py --filetype='image' --testdir='test_imgs/VOC/VOCdevkit/VOC2007/JPEGImages' --output_img --output_txt
    ```
4. Calculate mAP
    ```
    git clone https://github.com/Cartucho/mAP.git
    ```
5.  ```
    pip install opencv-python
    sudo apt-get install libsm6
    sudo apt-get install libgtk2.0-dev
    ```

