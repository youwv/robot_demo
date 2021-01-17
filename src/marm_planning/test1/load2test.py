import tensorflow as tf
import numpy as np
import env
import virtual_env
import time
a_bound = np.divide([2.9, 1.90, 2.2], 10)
offset = np.divide([0.0, 0.2, -1.1], 10)
state = tf.placeholder(dtype=tf.float32, shape=[None, 6], name='state')

env1 = virtual_env.abb_env()
env2 = env.Env()


def build_actor_network(s, bound, trainable):
    with tf.variable_scope('Actor'):
        with tf.variable_scope('target'):

            layer_1 = tf.layers.dense(s, 256, activation=tf.nn.tanh, name='layer_1', trainable=trainable)
            layer_2 = tf.layers.dense(layer_1, 256, activation=tf.nn.tanh, name='layer_2', trainable=trainable)
            a = tf.layers.dense(layer_2, 3, activation=tf.nn.tanh, name='actor', trainable=trainable)
            return tf.multiply(a, bound, name="limit_a")


actor = build_actor_network(state, bound=a_bound, trainable=True)
# Notice: init = tf.global_variables_initializer() is unnecessary
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('folder_for_nn'))
    print 'load successfully!'
    # print sess.run(actor, feed_dict={state: [[]]})
    r = []
    for num in range(50):
        env1.get_random_pos()
    # print env1.random_pos
    env2.random_pos = env1.random_pos
    for j in range(50):
        env2.set_target_position()
        action = [0.0] * 3
        s = env2.reset(action)
        ep_r = 0.0
        for i in range(20):
            s = np.reshape(s, (6,))
            a_temp = sess.run(actor, feed_dict={state: s[np.newaxis, :]})[0]
            action += a_temp
            next_state, reward, done, dist = env2.step(action)
            ep_r += reward
            time.sleep(0.2)
            if done:
                break
            s = next_state
        print ep_r, i, dist
        r.append(ep_r)
    import matplotlib.pyplot as plt
    plt.plot(r)
    plt.show()
    plt.savefig("reward.png")
