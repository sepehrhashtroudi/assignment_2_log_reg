import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


n_epoch = 1
batch_size = 400
train_loss_list =[]
valid_loss_list = []
test_accuracy_list =[]
train_accuracy_list =[]
valid_accuracy_list =[]
learning_rates = [1e-5, 1e-3, 1e-3]
loss_selector = 1




def plot_loss_acc(loss_acc_list):
    plt.plot(loss_acc_list)
    plt.show()

def accuracy(test_data, test_target, w , b):
    error = 0
    for i in range(len(test_target)):
        predicted_target = np.round(1.0 / (1 + np.exp(np.matmul(test_data[i], w) + b)))
        if(predicted_target != test_target[i] ):
            error += 1
    return 100 - 100*error/len(test_target)

def predict(data,w,b):
    predicted_target =[]
    for i in range(len(data[:,0])):
        predicted_target.append( np.round(1.0 / (1 + np.exp(np.matmul(data[i], w) + b))) )
    return predicted_target

def predict_prob(data,w,b):
    predicted_target =[]
    for i in range(len(data[:,0])):
        predicted_target.append(1.0 / (1 + np.exp(np.matmul(data[i], w) + b)) )
    return predicted_target

def log_reg_tf(input , target):
    X = tf.placeholder(dtype=tf.float64, shape=[None, len(input[1])], name="x")
    Y = tf.placeholder(dtype=tf.float64 ,name="y")
    w = tf.Variable(tf.random_normal(shape=[len(input[1]), 1], stddev=0.01, dtype=tf.float64), name="weights", dtype=tf.float64)
    b = tf.Variable(0.0, dtype=tf.float64)
    Y_predicted = 1.0 / (1 + tf.exp( tf.matmul(X , w) + b))

    if loss_selector == 0:
        loss = -1 * tf.reduce_sum(Y * tf.log(Y_predicted) + (1 - Y) * tf.log(1 - Y_predicted))
    if loss_selector == 1:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_predicted))
    if loss_selector == 2:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_predicted)) + tf.constant(learning_rates[2] / 2, dtype=tf.float64) * tf.pow(tf.linalg.norm(w), 2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rates[loss_selector]).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_epoch):
            train_loss = 0
            for idx in range(int(len(input[:,0])/batch_size)):
                Input_list = {X: input[idx*batch_size:(idx+1)*batch_size], Y: target[idx*batch_size:(idx+1)*batch_size]}
                _,i_Loss,w_value ,b_value = sess.run([optimizer,loss,w,b], feed_dict=Input_list)
                train_loss = train_loss + i_Loss
                # print (w_value)
            # train_loss_list.append(train_loss/len(input))
            # valid_loss_list.append(sess.run(loss, feed_dict={X: x_valid, Y: y_valid}) / len(x_valid))
            # train_accuracy_list.append(accuracy(x_train, y_train, w_value, b_value))
            # valid_accuracy_list.append(accuracy(x_valid, y_valid, w_value, b_value))
        # print(w_value,b_value)
        return w_value , b_value





