import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(0)
tf.set_random_seed(123)

# =====================
#  Data Gen
# =====================
mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 10000
train_size = 0.8
indices = np.random.permutation(range(n))[:N]

X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test =train_test_split(X, Y, train_size=train_size)

# =====================
#  Model
# =====================
n_in = len(X[0])  # 784
n_hidden = 200
n_out = len(Y[0])  # 10


def prelu(x, alpha):
    return tf.maximum(tf.zeros(tf.shape(x)), x) \
        + alpha * tf.minimum(tf.zeros(tf.shape(x)), x)


x = tf.placeholder(tf.float32, shape=[None, n_in])
t = tf.placeholder(tf.float32, shape=[None, n_out])

# Input Layer - Hidden Layer
W0 = tf.Variable(tf.truncated_normal([n_in, n_hidden], stddev=0.01))
b0 = tf.Variable(tf.zeros([n_hidden]))
alpha0 = tf.Variable(tf.zeros([n_hidden]))
h0 = prelu(tf.matmul(x, W0) + b0, alpha0)

# Hidden Layer - Hidden Layer
W1 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b1 = tf.Variable(tf.zeros([n_hidden]))
alpha1 = tf.Variable(tf.zeros([n_hidden]))
h1 = prelu(tf.matmul(h0, W1) + b1, alpha1)

W2 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b2 = tf.Variable(tf.zeros([n_hidden]))
alpha2 = tf.Variable(tf.zeros([n_hidden]))
h2 = prelu(tf.matmul(h1, W2) + b2, alpha2)

W3 = tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=0.01))
b3 = tf.Variable(tf.zeros([n_hidden]))
alpha3 = tf.Variable(tf.zeros([n_hidden]))
h3 = prelu(tf.matmul(h2, W3) + b3, alpha3)

# Hidden Layer - Output Layer
W4 = tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.01))
b4 = tf.Variable(tf.zeros([n_out]))
y = tf.nn.softmax(tf.matmul(h3, W4) + b4)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# =====================
#  Restore
# =====================
saver = tf.train.Saver()
sess = tf.Session()

model_dir = os.path.join(os.getcwd(), 'model')
saver.restore(sess, model_dir + '/model_49.ckpt')

#X0 = mnist.data[2]
#result0 = y.eval(session=sess, feed_dict={ x: [X0] })

X3 = mnist.data[20000]
result3 = y.eval(session=sess, feed_dict={ x: [X3] })

print(result3)

#plt.ion()
#mnist = datasets.fetch_mldata('MNIST original', data_home='.')
#pixels = mnist.data[20000]
#pixels = pixels.reshape((28, 28))
#plt.imshow(pixels, cmap='gray')
#plt.show()
#plt.close()




