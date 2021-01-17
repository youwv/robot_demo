# import tensorflow as tf
# import numpy as np


# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
# b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
#
# # Notice: init = tf.global_variables_initializer() is unnecessary
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('folder_for_nn'))
#     print("weights:", sess.run(W))
#     print("biases:", sess.run(b))


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore", ".*GUI is implemented.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

x_t = np.linspace(-10.0, 10.0, 100,)[:, np.newaxis]
noise = np.random.normal(x_t, 5.0)
y_t = x_t ** 2 + 2.0 + noise
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x_input')


def net(inp):
    w1 = tf.get_variable(name='w1', shape=[1, 5], )
    b1 = tf.get_variable(name='b1', shape=[5], )
    h1 = tf.nn.swish(tf.matmul(inp, w1)) + b1

    w2 = tf.get_variable(name='w2', shape=[5, 1], )
    b2 = tf.get_variable(name='b2', shape=[1], )
    y_p = tf.nn.swish(tf.matmul(h1, w2)) + b2
    return y_p


y_pre = net(x_input)
loss = tf.losses.mean_squared_error(y_t, y_pre)
opt = tf.train.AdamOptimizer(0.5).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
plt.ion()
for step in range(100):
    _, l, pred = sess.run([opt, loss, y_pre], {x_input: x_t})
    if step % 5 == 0:
        plt.cla()
        plt.scatter(x_t, y_t, )
        plt.plot(x_t, pred, 'r-', lw=5)
        plt.text(-10, 120, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
saver.save(sess, 'folder_for_nn/save_net.ckpt', global_step=100)
plt.ioff()
plt.show()
