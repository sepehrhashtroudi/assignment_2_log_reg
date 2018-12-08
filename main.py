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
from log_reg import *


credit_card = pd.read_csv('input/creditcard.csv')
# f, ax = plt.subplots(figsize=(7, 5))
# sns.countplot(x='Class', data=credit_card)
# _ = plt.title('# Fraud vs NonFraud')
# _ = plt.xlabel('Class (1==Fraud)')
# plt.show()

base_line_accuracy = 1-np.sum(credit_card.Class)/credit_card.shape[0]
print(base_line_accuracy)
X = credit_card.drop(columns='Class', axis=1)
y = credit_card.Class.values

# corr = X.corr()
#
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio
# f, ax = plt.subplots(figsize=(11, 9))
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})

np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
w,b = log_reg_tf(X_train,y_train)

y_train_hat = predict(X_train,w,b)
y_train_hat_probs = predict_prob(X_train,w,b)
train_accuracy = accuracy_score(y_train, y_train_hat)*100
train_auc_roc = roc_auc_score(y_train, y_train_hat_probs)*100
print("TF implementation results:")
print('Confusion matrix:\n', confusion_matrix(y_train, y_train_hat))
print('Training accuracy: %.4f %%' % train_accuracy)
print('Training AUC: %.4f %%' % train_auc_roc)


X_test = scaler.transform(X_test)
y_test_hat = predict(X_test,w,b)
y_test_hat_probs = predict_prob(X_test,w,b)
test_accuracy = accuracy_score(y_test, y_test_hat)*100
test_auc_roc = roc_auc_score(y_test, y_test_hat_probs)*100
print("TF implementation results test:")
print('Confusion matrix:\n', confusion_matrix(y_test, y_test_hat))
print('Test accuracy: %.4f %%' % test_accuracy)
print('Test AUC: %.4f %%' % test_auc_roc)
print(classification_report(y_test, np.array(y_test_hat) , digits=6))
fpr, tpr, thresholds = roc_curve(y_test, y_test_hat_probs, drop_intermediate=True)

f, ax = plt.subplots(figsize=(9, 6))
_ = plt.plot(fpr, tpr, [0,1], [0, 1])
_ = plt.title('AUC ROC')
_ = plt.xlabel('False positive rate')
_ = plt.ylabel('True positive rate')
plt.style.use('seaborn')

plt.savefig('auc_roc.png', dpi=600)
y_hat_90 = []
for i in range(len(y_test_hat_probs)):
    y_hat_90.append((y_test_hat_probs[i] > 0.90 )*1)
print('Confusion matrix:\n', confusion_matrix(y_test, y_hat_90))
print(classification_report(y_test, np.array(y_hat_90), digits=6))
y_hat_10 =[]
for i in range(len(y_test_hat_probs)):
    y_hat_10.append((y_test_hat_probs[i] > 0.10)*1)
print('Confusion matrix:\n', confusion_matrix(y_test, y_hat_10))
print(classification_report(y_test, np.array(y_hat_10), digits=4))