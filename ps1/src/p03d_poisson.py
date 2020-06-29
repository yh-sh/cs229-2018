import numpy as np
import util
import random

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    model = PoissonRegression()
    model.fit(x_train, y_train, iter=10000)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_pred = model.predict(x_val)
    np.savetxt(pred_path, y_pred)
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y, lr=1e-7, iter=10000, batch_size=500):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        mm = x.shape[0]
        nn = x.shape[1]
        self.theta = np.zeros(nn)
        iter = 10000
        train_n = len(x)
        for i in range(iter):
            permutation = list(np.random.permutation(mm))
            x_rand = x[permutation]
            y_rand = y[permutation]
            mini_batch_x = [ x_rand[k:k+batch_size] for k in range(0, train_n, batch_size)]
            mini_batch_y = [ y_rand[k:k+batch_size] for k in range(0, train_n, batch_size)]
            for j in range(len(mini_batch_x)):
                grad = (mini_batch_y[j] - np.exp(np.dot(self.theta, mini_batch_x[j].T))).dot(mini_batch_x[j]) / batch_size
                self.theta = self.theta + lr * grad
        # *** END CODE HERE ***
    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        z = np.dot(self.theta, x.T)
        print(z)
        return np.rint(np.exp(z))
        # *** END CODE HERE ***
