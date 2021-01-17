# import tensorflow as tf
#
# # Save to file
# # remember to define the same dtype and shape when restore
# W = tf.Variable([[2, 5, 7], [11, 13, 19]], dtype=tf.float32, name='weights')
# b = tf.Variable([[23, 29, 31]], dtype=tf.float32, name='biases')
#
# # initialization
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "folder_for_nn/save_net.ckpt")
#     print("Save to path: ", save_path)

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
warnings.filterwarnings("ignore", ".*GUI is implemented.*")

x_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x_input')

x_t = np.linspace(-10.0, 10.0, 100, )[:, np.newaxis]
noise = np.random.normal(x_t, 5.0)
y_t = x_t ** 2 + 2.0 + noise


def net(inp):
    w1 = tf.get_variable(name='w1', shape=[1, 5], )
    b1 = tf.get_variable(name='b1', shape=[5], )
    h1 = tf.nn.swish(tf.matmul(inp, w1)) + b1

    w2 = tf.get_variable(name='w2', shape=[5, 1], )
    b2 = tf.get_variable(name='b2', shape=[1], )
    y_p = tf.nn.swish(tf.matmul(h1, w2)) + b2
    return y_p


y = net(x_input)
# Notice: init = tf.global_variables_initializer() is unnecessary
loss = tf.losses.mean_squared_error(y_t, y)
opt = tf.train.AdamOptimizer(0.5).minimize(loss)
saver = tf.train.Saver()
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('folder_for_nn'))
    print("success!")
    plt.ion()
    for step in range(10):
        _, l, pred = sess.run([opt, loss, y], {x_input: x_t})
        if step % 5 == 0:
            plt.cla()
            plt.scatter(x_t, y_t, )
            plt.plot(x_t, pred, 'r-', lw=5)
            plt.text(-10, 120, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
    saver.save(sess, 'folder_for_nn/save_net.ckpt', global_step=100)
    plt.ioff()
    plt.show()
