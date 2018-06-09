import tensorflow as tf
import seaborn
import numpy as np
import matplotlib.pyplot as plt

# input data
xData = np.arange(100, step=.1)
yData = xData + 20 * np.sin(xData/10)

#plot the input
plt.scatter(xData, yData)
plt.show()

# define the data size and batch size
nSamples = 1000
batchSize = 100

# resize input for tensorflow
xData = np.reshape(xData, (nSamples, 1))
yData = np.reshape(yData, (nSamples, 1))

X = tf.placeholder(tf.float32, shape=(batchSize, 1))
y = tf.placeholder(tf.float32, shape=(batchSize, 1))

# define the variables to be learned
with tf.variable_scope("linear-regression-pipeline"):
    W = tf.get_variable("weights", (1,1), initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1, ), initializer=tf.constant_initializer(0.0))

    # model
    yPred = tf.matmul(X, W) + b
    # loss function
    loss = tf.reduce_sum((y - yPred)**2/nSamples)

# set the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
#optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(loss)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate=.001).minimize(loss)
#optimizer = tf.train.AdagradOptimizer(learning_rate=.001).minimize(loss)
#optimizer = tf.train.MomentumOptimizer(learning_rate=.001, momentum=0.9).minimize(loss)
#optimizer = tf.train.FtrlOptimizer(learning_rate=.001).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=.001).minimize(loss)

errors = []
with tf.Session() as sess:
    # init variables
    sess.run(tf.global_variables_initializer())

    for _ in range(1000):
        # select mini batch
        indices = np.random.choice(nSamples, batchSize)
        xBatch, yBatch = xData[indices], yData[indices]
        # run optimizer
        _, lossVal = sess.run([optimizer, loss], feed_dict={X: xBatch, y: yBatch})
        errors.append(lossVal)

plt.plot([np.mean(errors[i-50:i]) for i in range(len(errors))])
plt.show()
plt.savefig("errors.png")
