import tensorflow._api.v2.compat.v1 as tf

# simple tensorflow 1 style code to compute graphs using
tf.disable_v2_behavior()

v_1 = tf.constant([1, 2, 3, 4])
v_2 = tf.constant([2, 1, 5, 3])
v_add = tf.add(v_1, v_2)

with tf.Session() as sess:
    print(sess.run(v_add))

# zeros
print(tf.zeros([2, 3], tf.float32))

# ones
print(tf.ones([3, 4], tf.int32))

# random
print(tf.random_uniform([2, 3], maxval=4, seed=12))

# evaluating y = 2x for a 4 by 5 matrix tensor
x = tf.placeholder("float")
y = 2 * x

with tf.Session() as sess:
    data = tf.random_uniform([4, 5], 10)
    x_data = sess.run(data)
    print(sess.run(y, feed_dict={x: x_data}))

in_a = tf.placeholder(dtype=tf.float32, shape=(2))

# checking out puts for tensorboards


def model(x):
    with tf.variable_scope("matmul"):
        W = tf.get_variable("W", initializer=tf.ones(shape=(2, 2)))
        b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
        return x * W + b


out_a = model(in_a)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outs = sess.run([out_a], feed_dict={in_a: [1, 0]})
    writer = tf.summary.FileWriter("./logs/example", sess.graph)
