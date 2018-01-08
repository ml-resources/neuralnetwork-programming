import tensorflow as tf
import numpy as np

tf.InteractiveSession()

# tensorflow operations
a = tf.zeros((3,3))
b = tf.ones((3,3))

print(tf.reduce_sum(b, reduction_indices=1).eval())
print(a.get_shape())

# numpy operations
a = np.zeros((3, 3))
b = np.ones((3, 3))

print(np.sum(b, axis=1))
print(a.shape)

# session objects
a = tf.constant(6.0)
b = tf.constant(7.0)

c = a * b
with tf.Session() as sess:
	print(sess.run(c))
	print(c.eval())

# tensor variables
W1 = tf.ones((3,3))
W2 = tf.Variable(tf.zeros((3,3)), name="weights")

with tf.Session() as sess:
	print(sess.run(W1))
	sess.run(tf.global_variables_initializer())
	print(sess.run(W2))

# Variable objects can be initialized from constants or random values
W = tf.Variable(tf.zeros((2,2)), name="weights")
R = tf.Variable(tf.random_normal((2,2)), name="random_weights")

with tf.Session() as sess:
 	# Initializes all variables with specified values.
	sess.run(tf.initialize_all_variables())
	print(sess.run(W))
	print(sess.run(R))

state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(state))
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))

input1 = tf.constant(5.0)
input2 = tf.constant(6.0)
input3 = tf.constant(7.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

# Calling sess.run(var) on a tf.Session() object retrieves its value. Can retrieve multiple variables simultaneously with sess.run([var1, var2])
with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print(result)

a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
	print(sess.run(ta))

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
	print(sess.run([output], feed_dict={input1:[5.], input2:[6.]}))

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
assert v.name == "foo/bar/v:0"

#Variable scopes control variable (re)use
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v

#reuse is false
with tf.variable_scope("foo"):
    n = tf.get_variable("n", [1])
assert v.name == "foo/n:0"

#Reuse is true
with tf.variable_scope("foo"):
    n = tf.get_variable("n", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("n", [1])
assert v1 == n




