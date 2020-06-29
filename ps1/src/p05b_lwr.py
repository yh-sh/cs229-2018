import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    model = LocallyWeightedLinearRegression(0.5)
    model.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    mse = np.mean((y_pred - y_eval)**2)
    print("MSE={}".format(mse))
    sorted_indx = np.argsort(x_eval[:, 1])
    x_eval = x_eval[:, 1][sorted_indx]
    y_pred = y_pred[sorted_indx]
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    #plt.plot(x_eval[:,1], y_eval, 'ro', linewidth=2)
    plt.plot(x_eval, y_pred, 'y-', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        mm = x.shape[0]
        pred_y = np.zeros(mm)
        calw = lambda x: np.exp(-1/(2*(self.tau)**2) * x)
        for i in range(mm): 
            weight = calw(np.linalg.norm(self.x - x[i], ord=2, axis=1))
            w = np.diag(weight)
            theta = np.linalg.pinv(self.x.T.dot(w).dot(self.x)).dot(self.x.T).dot(w).dot(self.y)
            pred_y[i] = theta.T.dot(x[i])
        return pred_y
        # *** END CODE HERE ***
