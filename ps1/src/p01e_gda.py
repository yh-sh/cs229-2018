import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    model = GDA()
    model.fit(x_train, y_train)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    x_val[:, 1] = np.log(x_val[:, 1])
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, 'output')
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        #x[:, -1] = np.log(x[:, -1])
        mm = x.shape[0]
        nn = x.shape[1]
        mu_0 = np.zeros(nn)
        mu_1 = np.zeros(nn)
        sigma = np.zeros((nn, nn))
        phi = np.sum(y) / mm
        x_1 = np.sum(x[y==1, :], axis=0)
        x_0 = np.sum(x[y==0, :], axis=0)
        mu_0 = x_0 / (mm - phi * mm)
        mu_1 = x_1 / (phi * mm)
        for i in range(mm):
            sig = (x[i, :] - mu_0).reshape(-1, 1) * (x[i, :] - mu_1)
            sigma = sigma + sig
        sigma = sigma / mm
        self.theta = np.zeros(nn + 1)
        self.theta[0] = np.log((1-phi)/phi) - 0.5 * (np.dot(np.dot(mu_0, np.linalg.pinv(sigma)), mu_0) - np.dot(np.dot(mu_1, np.linalg.pinv(sigma)), mu_1))
        self.theta[1:] = np.dot(mu_0 - mu_1, np.linalg.pinv(sigma))
        # *** END CODE HERE ***
        self.theta = -self.theta
    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = self.theta[0] + np.dot(x, self.theta[1:])
        res = np.zeros(z.shape)
        res[z>=0] = 1
        res[z<0] = 0
        return res
        # *** END CODE HERE
