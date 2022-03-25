import tensorflow as tf
print(tf.__version__)
x = tf.constant(range(12))
print(x, 'x')
print(x.shape, 'x.shape')
print(tf.shape(x), 'tf.shape(x)')
print(len(x), 'len(x)')

X = tf.reshape(x,(3,4))
print(X, 'X')

y = tf.zeros((2,3,4))
print(y, 'y')
print(y.shape, 'y.shape')
print(tf.shape(y), 'tf.shape(y)')
print(len(y), 'len(y)')

Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(Y, 'Y')
print(X / Y, 'X / Y')
print(Y[1:3], 'Y[1:3]')
print(Y[1,2], 'Y[1,2]')