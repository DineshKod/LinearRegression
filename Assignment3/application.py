import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.array(y).T
X = np.array(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE
trained_weights = train(Xtrain, Ytrain, alpha=0.01, n_epoch=1000)
print(f'trained_weights (w) : {trained_weights}')

yhat_train = compute_yhat(Xtrain, trained_weights)
train_loss = compute_L(yhat_train, Ytrain)
print(f'Training Loss: {train_loss}')

yhat_test = compute_yhat(Xtest, trained_weights)
test_loss = compute_L(yhat_test, Ytest)
print(f'Training Loss: {test_loss}')

#########################################
