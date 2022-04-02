import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # 使用gpu
# 查看使用设备
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    print('init 被执行')
    self.num_outputs = num_outputs
    self.i = 0
    print('Init:This is i',self.i)
    self.i = self.i +1
  def build(self,input_shape):
    print('build 被执行')
    print('input_shape',input_shape)
    print('Build:This is i',self.i)
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
  
  def call(self, input):
    print('call 被执行')
    return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
_ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.
print([var.name for var in layer.trainable_variables])
_ = layer(tf.ones([10, 5]))
print([var.name for var in layer.trainable_variables])
