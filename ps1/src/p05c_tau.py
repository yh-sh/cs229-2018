import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    sorted_indx = np.argsort(x_eval[:, 1])
    x_eval = x_eval[sorted_indx]
    color = ['y', 'r', 'm', 'g', 'brown', 'purple']
    model = LocallyWeightedLinearRegression(0)
    model.fit(x_train, y_train)
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    for i in range(len(tau_values)):
        model.tau = tau_values[i]
        y_pred = model.predict(x_eval)
        mse = np.mean((y_pred - y_eval)**2)
        print("tau={}, MSE={}".format(tau_values[i], mse))       
        plt.plot(x_eval[:,1], y_pred, c = color[i], linewidth=2, label="tau={}".format(tau_values[i]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    # *** END CODE HERE ***
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    model.tau = 0.121
    y_pred = model.predict(x_test)
    mse = np.mean((y_pred - y_test)**2)
    print("Test time   tau=0.121, MSE={}".format(mse))