import os
import numpy as np
import cv2
import tensorflow as tf

batch_size = 32
train_file = 'data/train.txt'

with open(train_file, 'r') as f:
	data = f.readlines()
	num_images = len(data)

train_dataset = tf.data.TestLineDataset(train_file)
train_dataset = train_dataset.suffle(num_images)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_file.map(lambda x: tf.py_func(get_batch_data,
													))