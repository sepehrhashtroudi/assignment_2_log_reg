import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data set please wait ... ")
credit_card = pd.read_csv("dataset/creditcard.csv")
credit_card = credit_card.sample(frac=1).reset_index(drop=True)

data, target = credit_card.drop(['Class'], axis=1), np.array(credit_card['Class']).reshape(-1, 1)
data_n = (data - data.mean())/(data.max() - data.min())

l_d = len(data_n)
print("len = ", l_d)
train_indx, valid_indx, test_indx =  int(l_d * 0.75), int(l_d * 0.95), l_d

train_data, train_target = data_n[0: train_indx], target[0: train_indx]
valid_data, valid_target = data_n[train_indx: valid_indx], target[train_indx: valid_indx]
test_data, test_target = data_n[valid_indx: test_indx], target[valid_indx: test_indx]

#parameters 
learning_rate = 0.001
epoch = 500
batch_size = 40000

#tensorflow model
x = tf.placeholder(tf.float32, shape=(None, 30), name="x")#30 feature
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
w = tf.Variable(tf.random_normal(shape=[30, 1], stddev=0.1, dtype=tf.float32), 
                name="weights", dtype=tf.float32)
b = tf.Variable(0.0, name="bias", dtype=tf.float32)

#svm loss
model_output = tf.subtract(tf.matmul(x, w), b)
l2_norm = tf.reduce_sum(tf.square(w))
alpha = tf.constant([0.0], dtype=tf.float32)
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1.,tf.multiply(model_output, y))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

ses = tf.InteractiveSession()
ses.run(tf.global_variables_initializer())
train_loss_list = []
valid_loss_list = []
train_acc_list = []
valid_acc_list = []
st = time.time()
for e_itr in range(epoch):
    train_loss = 0
    start = time.time()
    for indx  in range(len(train_data)//batch_size):
        input_list = {x: train_data[indx * batch_size:(indx + 1) * batch_size],
                      y: train_target[indx * batch_size:(indx + 1) * batch_size]}
        _, tl = ses.run([optimizer, loss], feed_dict=input_list)
        train_loss += tl
    print(train_loss, "time = ", time.time()-start)
print("full time is = ", time.time()-st)

predict_test_un_round = ses.run(model_output, feed_dict={x: test_data})
predict_test =  tf.sign(predict_test_un_round)
print(predict_test)
print(test_target)
print('Confusion matrix with 0.5 threshold:\n', confusion_matrix(test_target, predict_test))
print(classification_report(test_target, predict_test, digits=3))

train_accuracy = accuracy_score(test_target, predict_test)*100
print('Training accuracy: %.4f %%' % train_accuracy)

ses.close()
