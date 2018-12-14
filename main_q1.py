# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

n_epoch = 60
batch_size = 50000
learning_rates = [1e-5, 1e-3, 1e-3]
loss_selector = 1
np.random.seed(400)


print("Loading data")
credit_card = pd.read_csv("input/creditcard.csv")
credit_card = credit_card.sample(frac=1).reset_index(drop=True)
x, y = credit_card.drop(['Class'], axis=1), np.array(credit_card['Class']).reshape(-1, 1)
sc = StandardScaler()
sc.fit(x)
x_normalized = sc.transform(x)
train_x, test_x, train_y, test_y = train_test_split(x_normalized, y, test_size=0.1, random_state=42)


X = tf.placeholder(dtype=tf.float64, shape=[None, 30], name="x")
Y = tf.placeholder(dtype=tf.float64 ,name="y")
w = tf.Variable(tf.random_normal(shape=[30, 1], stddev=0.01, dtype=tf.float64), name="weights", dtype=tf.float64)
b = tf.Variable(0.0, dtype=tf.float64)
logit = tf.matmul(X, w) + b
Y_predicted = 1.0 / (1 + tf.exp(-logit))


if loss_selector == 0:
    loss = -1 * tf.reduce_sum(Y * tf.log(Y_predicted) + (1 - Y) * tf.log(1 - Y_predicted))
if loss_selector == 1:
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logit))
if loss_selector == 2:
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logit)) + tf.constant(learning_rates[2] / 2, dtype=tf.float64) * tf.pow(tf.linalg.norm(w), 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rates[loss_selector]).minimize(loss)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(n_epoch):
    train_loss = 0
    for idx in range(int(len(train_x)/batch_size)):
        Input_list = {X: train_x[idx*batch_size:(idx+1)*batch_size], Y: train_y[idx*batch_size:(idx+1)*batch_size]}
        _,i_Loss,w_value ,b_value = sess.run([optimizer,loss,w,b], feed_dict=Input_list)
        train_loss = train_loss + i_Loss
    print("Epoch {}".format(i))


predict_prob = sess.run(Y_predicted, feed_dict={X: train_x})
predict1 = np.round(predict_prob)
predict2 = (predict_prob > 0.005)
print('Trainig Confusion matrix for 0.5 threshold:\n', confusion_matrix(train_y, predict1))
print('Trainig Confusion matrix for 0.005 threshold:\n', confusion_matrix(train_y, predict2))
print(classification_report(train_y, predict1, digits=3))
print(classification_report(train_y, predict2, digits=3))
train_accuracy = accuracy_score(train_y, predict2) * 100
train_auc_roc = roc_auc_score(train_y, predict_prob) * 100
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)
fpr, tpr, thresholds = roc_curve(train_y, predict_prob, drop_intermediate=True)
f, ax = plt.subplots(figsize=(9, 6))
_ = plt.plot(fpr, tpr, [0, 1], [0, 1])
_ = plt.title('AUC ROC')
_ = plt.xlabel('False positive rate')
_ = plt.ylabel('True positive rate')
plt.show()


predict_prob = sess.run(Y_predicted, feed_dict={X: test_x})
predict1 = np.round(predict_prob)
predict2 = (predict_prob > 0.005)
print('Trainig Confusion matrix for 0.5 threshold:\n', confusion_matrix(test_y, predict1))
print('Trainig Confusion matrix for 0.005 threshold:\n', confusion_matrix(test_y, predict2))
print(classification_report(test_y, predict1, digits=3))
print(classification_report(test_y, predict2, digits=3))
train_accuracy = accuracy_score(test_y, predict2) * 100
train_auc_roc = roc_auc_score(test_y, predict_prob) * 100
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)
fpr, tpr, thresholds = roc_curve(test_y, predict_prob, drop_intermediate=True)
f, ax = plt.subplots(figsize=(9, 6))
_ = plt.plot(fpr, tpr, [0, 1], [0, 1])
_ = plt.title('AUC ROC')
_ = plt.xlabel('False positive rate')
_ = plt.ylabel('True positive rate')
plt.show()

sess.close()






