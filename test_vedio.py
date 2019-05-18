#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
caffe_root = '/mnt/home/nvidia/ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')
import caffe
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--colours', type=str, required=True, help='label colours')
    #parser.add_argument('--input_image', type=str, required=True, help='input image path')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                   'should be stored')
    parser.add_argument('--gpu', type=str, default='0', help='0: gpu mode active, else gpu mode inactive')
    parser.add_argument('--input_video', type=str, required=True, help='input vedio path')
    
    return parser
# initialize the video stream and pointer to output video file
#------------------------------------------------------------------
class time():
    def __init__(self):
        self.init_count=cv2.getTickCount()
        self.frame=0
        self.time=0
    def get_fps(self):
        #cv2.getTickFrequency()
        self.time = (cv2.getTickCount() - self.init_count)/cv2.getTickFrequency()
        self.frame = self.frame + 1
        if self.time>1 :
            print('FPS:%d'%self.frame)
            self.frame=0
            self.init_count=cv2.getTickCount()
#-------------------初始化视频和视频读取----------------------------
cap = cv2.VideoCapture("/home/nvidia/sample_720p.mp4")
#cap = cv2.VideoCapture(1)
fps=cap.get(cv2.CAP_PROP_FPS)
print('fps=%d'%fps)
# try to determine the total number of frames in the video file
prop =  cv2.CAP_PROP_FRAME_COUNT
total = int(cap.get(prop))
print('total %d'%total)
writer = None


if __name__ == '__main__':
	parser1 = make_parser()
	args = parser1.parse_args()
        caffe.set_mode_gpu()
	#if args.gpu == 0:
	#	caffe.set_mode_gpu()
	#else:
	#	caffe.set_mode_cpu()

	net = caffe.Net(args.model, args.weights, caffe.TEST)
        input_shape = net.blobs['deconv6_0_0'].data.shape
	#input_shape = net.blobs['data'].data.shape
	output_shape = net.blobs['deconv6_0_0'].data.shape
        label_colours = cv2.imread(args.colours, 1).astype(np.uint8)
#----------------------循环主体-------------------------------------------
time1 = time()
while True:
	ret,frame = cap.read()
        #ret是否有读取到----------frame帧图像--------
	       
	#input_image = cap.read(args.input_vedio, 1).astype(np.float32)
	input_image = frame
	input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
	input_image = input_image.transpose((2, 0, 1))
	input_image = np.asarray([input_image])
	out = net.forward_all(**{net.inputs[0]: input_image})
	prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)
	prediction = np.squeeze(prediction)
	prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
	prediction = prediction.transpose(1, 2, 0).astype(np.uint8)
	prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
	label_colours_bgr = label_colours[..., ::-1]
	cv2.LUT(prediction, label_colours_bgr, prediction_rgb)
	frame=cv2.resize(frame, (1280,720),interpolation = cv2.INTER_CUBIC)#修改大小
        prediction_rgb=cv2.resize(prediction_rgb , (1280,720),interpolation = cv2.INTER_CUBIC)
        output = ((0.3 * frame) + (0.7 * prediction_rgb )).astype("uint8")
        #cv2.imshow("input",frame)	
	#cv2.imshow("ENet", prediction_rgb)
	cv2.imshow("ENet", output)
#        if writer is None:
	# initialize our video writer
#-------		fourcc = cv2.VideoWriter_fourcc(*"MPEG")
#-------		writer = cv2.VideoWriter('output.mp4', fourcc, 30,
#-------			(input_shape[2], input_shape[3]), True)		
	# write the output frame to disk
	#writer.write(prediction_rgb)
#-------        writer.write(output)
	cv2.waitKey(10)
	time1.get_fps()
        if(cv2.waitKey(60) == 27):
                        break  
cap.release()
cv2.destroyAllWindows()


'''

#	    if args.out_dir is not None:
#		input_path_ext = args.input_video.split(".")[-1]
#		input_image_name = args.input_video.split("/")[-1:][0].replace('.' + input_path_ext, '')
#		out_path_im = args.out_dir + input_image_name + '_enet' + '.' + input_path_ext
#		out_path_gt = args.out_dir + input_image_name + '_enet_gt' + '.' + input_path_ext
#
#		cv2.imwrite(out_path_im, prediction_rgb)


		# cv2.imwrite(out_path_gt, prediction) #  label images, where each pixel has an ID that represents the class

'''





