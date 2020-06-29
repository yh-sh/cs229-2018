import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    model = LogisticRegression()
    model.fit(x_train, y_train)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    util.plot(x_val, y_val, model.theta, 'output')
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def sigmoid(self, z):
        s = np.zeros(z.shape)
        if z>=0:     
            s = 1.0/(1+np.exp(-z))
        else:
            s =  np.exp(z)/(1+np.exp(z))
        return s
    
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        mm = x.shape[0]
        nn = x.shape[1]
        self.theta = np.zeros(nn)
        tol = 1e9
        iter = 0
        while tol > 1e-6:
            Hessian = np.zeros((nn,nn))
            Jaco = np.zeros(nn)
            for i in range(mm):
                h_theta = self.sigmoid(np.dot(self.theta, x[i, :]))
                Jaco = Jaco + y[i] * x[i, :] - x[i, :] * h_theta
                Hessian = Hessian + h_theta * (1 - h_theta) * x[i,:].reshape(-1, 1) * x[i,:]
            Hessian = Hessian / mm
            #print(Hessian)
            Jaco = Jaco / mm
            old_theta = self.theta.copy()
            delta = np.dot(Jaco, np.linalg.pinv(Hessian))
            self.theta = old_theta - delta
            tol = np.sum(np.abs(delta))
            iter += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = - np.dot(x, self.theta.T)
        res = np.zeros(z.shape)
        res[z >=0] = 1
        res[z < 0] = 0
        return res
        # *** END CODE HERE ***

    def predict1(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = - np.dot(x, self.theta.T)
        h_y = 1/(1+np.exp(-z))
        return h_y

    def predict2(self, x, alpha):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = - np.dot(x, self.theta.T)
        h_y = 1/(1+np.exp(-z))
        s = h_y / alpha
        res = s.copy()
        res[s>=0.5] = 1
        res[s<0.5] = 0
        return res
