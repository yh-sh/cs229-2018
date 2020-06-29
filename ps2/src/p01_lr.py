# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    m, n = X.shape

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10
    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad
        learning_rate = 1 / (i * i) 
        if i % 10000 == 0:
            print('Finished {} iterations, update scale: {}'.format(i, np.linalg.norm(prev_theta - theta)))
        if np.linalg.norm(prev_theta - theta) < 1e-10:
            print('Converged in %d iterations' % i)
            break
    #print(theta)
    return


def main():
    # print('==== Training model on data set A ====')
    # Xa, Ya = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    # logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    logistic_regression(Xb, Yb)

def test():
    #==== Plot data set A ====')
    x, y = util.load_csv('../data/ds1_a.csv', add_intercept=True)
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=2)
    plt.show()
    #==== Plot data set B ====')
    x, y = util.load_csv('../data/ds1_b.csv', add_intercept=True)
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'go', linewidth=2)
    plt.show()

if __name__ == '__main__':
    #main()
    test()
