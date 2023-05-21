import numpy as np

from scipy import optimize
from scipy.special import expit  # logistic function

##################### MISSING DATA MECHANISMS #############################

##### Missing Completely At Random ######

def MCAR_mask(X, p):
    """
    Missing completely at random mechanism. Each value in the input matrix has an independent probability p of being missing.
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate.
    Returns
    -------
    mask : np.ndarray
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape
    mask = np.zeros((n, d)).astype(bool)

    ## Each value has a probability p of being missing
    mask = np.random.rand(n, d) < p

    return mask


##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of features with *no* missing values is
    randomly selected. The remaining features have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those features.
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for features which will have missing values.
    p_obs : float
        Proportion of features with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    mask : np.ndarray
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1)  ## number of features that will have no missing values (at least one variable)
    d_na = d - d_obs  ## number of features that will have missing values

    ### Sample features that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other features will have NA proportions that depend on those observed features, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = expit(np.dot(X[:, idxs_obs], coeffs) + intercepts)

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


##### Missing Not At Random ######

def MNAR_mask_logistic_old(X, p, p_params):
    """
    Missing not at random mechanism with a logistic masking model. Missing probabilities are selected with a logistic model,
    taking all features as inputs. Hence, values that are inputs can also be missing. Features are split into a set of 
    inputs for a logistic model, and a set whose missing probabilities are determined by the logistic model.
    Weights are random and the intercept is selected to attain the desired proportion of missing values.
    
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for features which will have missing values.
    p_params : float
        Proportion of features that will be used for the logistic masking model.

    Returns
    -------
    mask : np.ndarray
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    mask = np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1)  ## number of features used as inputs (at least one)
    d_na = d - d_params  ## number of features masked with the logistic model

    ### Sample features that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params])

    ### Other features will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = expit(np.dot(X[:, idxs_params], coeffs) + intercepts)

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## Mask some values used in the logistic model at random.
    ## This makes the missingness of other features potentially dependent on masked values
    mask[:, idxs_params] = np.random.rand(n, d_params) < p

    return mask

import numpy as np

import numpy as np

def MNAR_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic masking model.
    Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    The weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    Returns
    -------
    mask : np.ndarray
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    mask = np.zeros((n, d), dtype=bool)

    ### Sample variables that will be parameters for the logistic regression
    idxs_params = np.arange(d)

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_params)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = 1 / (1 + np.exp(-np.dot(X[:, idxs_params], coeffs) - intercepts))

    ber = np.random.rand(n, d)
    mask[:, idxs_params] = ber < ps

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = np.random.randn(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.randn(d_obs, d_na)
        Wx = np.dot(X[:, idxs_obs], coeffs)
        coeffs /= np.std(Wx, 0, keepdims=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        for j in range(d):
            def f(x):
                return expit(X * coeffs[j] + x).mean() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return expit(np.dot(X, coeffs[:, j]) + x).mean() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts

