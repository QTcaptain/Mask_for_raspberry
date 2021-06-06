# -*- coding:utf-8 -*-
import cv2
import sys
import time
import argparse
import os
import numpy as np
from PIL import Image
from anchor_generator import generate_anchors
from anchor_decode import decode_bbox
from nms import single_class_non_max_suppression
from loader import load_tf_model, tf_inference
import PRi.GPIO as GPIO
from mxl import MLX90614
sess, graph = load_tf_model('face_mask_detection.pb')
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)




id2class = {0: 'Mask:Open the door', 1: 'NoMask:Dont open the door'}
#树莓派部署设置
GPIO.setmode(GPIO.BOARD)
GPIO.SETUP(Ping, GPIO.OUT)
GPIO.output(PING, GPIO.LOW)
class_select = []

thermometer_address = 0x5a
path = '/home/pi/tensorflow_kz'
cmdilne = 'cd '+path
thermometer = MLX90614(thermometer_address)
#树莓派函数
    #播报语音
def km():
    os.system(cmdline)
    song = 'omxplayer km.wav'
    os.system(song)
def kz():
    os.system(cmdline)
    song = 'omxplayer km.wav'
    os.system(song)
def wd():
    os.system(cmdline)
    song = 'omxplayer km.wav'
    os.system(song)
#树莓派判断开门条件函数
def select(class_id):

    class_select.append(class_id)
    print(len(class_select))
    while len(class_select)==10:
        if sum(class_select)==0 and thermometer_address.get_obj_temp<=30:
            single_ = 1
            km()
            print("Open the door")
        elif sum(class_select)!=0 and thermometer_address.get_obj_temp<=30:
            single_ = 0
            print("Don't open the door")
            kz()
        elif thermometer_address.get_obj_temp>30:
            single_ = 0
            print("Don't open the door")
            wd()
        class_select.clear()
        return single_
def liuliang(single_):
    people = 0
    if single_ == 1:
        people =people + 1
        return people
def people_creat(people)
    f = open("people", 'w')
    f.write("今日人流量为:{}".format(people))
#控制继电器函数
def Door(single_):
    if single_ == 1:
        GPIO.output(Ping, GPIO.HIGH)
        time.sleep(5)
        GPIO.output(Ping, GPIO.LOW)
    else:
        GPIO.output(Ping, GPIO.LOW)
# def select(class_id):
#     # class_select = []
#     class_select.append(class_id)
#     print(class_select)
#     if sum(class_select) != len(class_select):
#         single = "Open the door"
#     else:
#         single = "Don't open the door"
#     # for i in class_select:
#     #     if i==0:
#     #         single="Open door"
#     #     else:
#     #         single="Don't open door"
#     print(single)
#     return single

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              ):

    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    #删除batch维度，因为batch总是1用于推断。
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        #检测置信度
        conf = float(bbox_max_scores[idx])
        #预测结果返回
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            #设置显示框颜色
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)  #框取出人脸

            cv2.putText(image, "%s: %.2f" % (id2class[class_id], thermometer.get_obj_temp()+8), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        #class_id 为id判断是否戴口罩
        #conf为置信度
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])


    return output_info
def infer():
    run_on_video(video_path=0, output_video_name='',conf_thresh=0.5)
def break_():
    sys.exit()
def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        #按帧读取视频
        status, img_raw = cap.read()    #seatus为读取是否成功的判断，img_raw为读取成功返回的图片

        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.6,
                      target_shape=(260, 260),
                      draw_result=True,
                     )
            cv2.imshow('image', img_raw[:, :, ::-1])
            cv2.waitKey(120)              #延时 60ms 切换到下一帧图像


infer()

