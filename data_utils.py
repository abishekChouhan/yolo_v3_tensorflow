import os
import numpy as np
import cv2
import tensorflow as tf


def parse_line(line):
	if not isinstance(line, str):
		line = line.decode()
	line = line.split(' ')
	line_index = line[0]
	img_path = line[1]
	width, height = line[2], line[3] 
	num_obj = len(line[4:])//5
	boxes, labels = [], []
	for obj in range(num_obj):
		x_min, y_min, x_max, y_max, label = line[4+(obj*5):9+(obj*5)]
		boxes.append([x_min, y_min, x_max, y_max])
		labels.append(label)
	boxes = np.array(boxes, dtype='float32')
	labels = np.array(labels, dtype='int32')
	return line_index, img_path, width, height, boxes, labels

def get_true_output(boxes, lables, img_size, clas_num, anchors):

	box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
	box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

	y_true_13 = np.zeros((img_size[1]//32, img_size[0]//32, 3, 5 + class_num), dtype=np.float32)
	y_true_26 = np.zeros((img_size[1]//16, img_size[0]//16, 3, 5 + class_num), dtype=np.float32)
	y_true_52 = np.zeros((img_size[1]//8, img_size[0]//8, 3, 5 + class_num), dtype=np.float32)


def get_line_data(line, num_classes):
	line_index, img_path, width, height, boxes, labels = parse_line(line)
	img = cv2.imread(img_path)
	boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype='float32')), axis=-1)
	
	# BGR -> RGB
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # the input of yolo_v3 should be in range 0~1
    img = img / 255.

    y_true_13, y_true_26, y_true_52 = get_true_output(boxes, labels, img_size, class_num, anchors)

    return img_idx, img, y_true_13, y_true_26, y_true_52


# def get_batch_data(batch_lines, num_classes, img_size, mode, anchors, letter_box_size):

# 	for line in batch_lines:
# 		img_index, img, y_true_13, y_true_26, y_true_52 = parse_line(line)

x = get_line_data('0 abc_0.jpg 416 416 10 10 20 20 0 40 50 70 90 1 10 10 20 20 0 40 50 70 90 1', 5)
print(x)