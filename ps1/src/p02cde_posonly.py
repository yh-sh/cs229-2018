import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels

    model = LogisticRegression()
    train_x, train_t = util.load_dataset(train_path,label_col="t", add_intercept=True)
    model.fit(train_x, train_t)
    val_x, val_y = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    y_pred = model.predict(val_x)
    np.savetxt(pred_path_c, y_pred)
    util.plot(val_x, val_y, model.theta, 'output')
    # Make sure to save outputs to pred_path_c


    # Part (d): Train on y-labels and test on true labels
    train_x, train_y = util.load_dataset(train_path, label_col='y', add_intercept=True)
    model.fit(train_x, train_y)
    val_x, val_y = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    y_pred = model.predict(val_x)
    np.savetxt(pred_path_d, y_pred)
    util.plot(val_x, val_y, model.theta, 'output')
    # Make sure to save outputs to pred_path_d


    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    test_x, test_t = util.load_dataset(test_path, label_col="t", add_intercept=True)
    y_pred = model.predict1(val_x)
    alpha = np.sum(y_pred[val_y==1]) / np.sum(val_y)
    y_pred = model.predict2(test_x, alpha)
    np.savetxt(pred_path_e, y_pred)
    model.theta[0] -= np.log(alpha/(2-alpha))
    util.plot(test_x, test_t, model.theta, 'output')
    # *** END CODER HERE
