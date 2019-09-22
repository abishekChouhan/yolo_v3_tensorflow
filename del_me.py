import numpy as np

anchors = '10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90,  156,198,  373,326'


x = np.reshape(np.asarray(anchors.split(','), np.float32), [-1, 2])
y = np.expand_dims(x*2,1)

print(np.minimum(-y/2,-x/2))
