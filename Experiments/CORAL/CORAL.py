import numpy as np

def coral(Xs, Xt, regularization=1e-5):
    """
    Perform CORAL: align source features Xs to target features Xt.
    
    Parameters:
        Xs: np.array of shape (n_s, d) - Source features
        Xt: np.array of shape (n_t, d) - Target features
        regularization: float - To stabilize covariance inversion

    Returns:
        Xs_aligned: np.array of shape (n_s, d) - CORAL-transformed source features
    """
    # Center the data
    Xs_centered = Xs - np.mean(Xs, axis=0)
    Xt_centered = Xt - np.mean(Xt, axis=0)

    # Compute covariance matrices
    Cs = np.cov(Xs_centered, rowvar=False) + regularization * np.eye(Xs.shape[1])
    Ct = np.cov(Xt_centered, rowvar=False) + regularization * np.eye(Xt.shape[1])

    # Matrix square root and inverse square root
    from scipy.linalg import fractional_matrix_power

    Cs_inv_sqrt = fractional_matrix_power(Cs, -0.5)
    Ct_sqrt = fractional_matrix_power(Ct, 0.5)

    # CORAL transformation
    A = Cs_inv_sqrt @ Ct_sqrt
    Xs_aligned = Xs_centered @ A

    return Xs_aligned

def predict(X, w, b):
    """
    Predict using the linear SVM model.
    
    Parameters:
        X: np.array of shape (n, d) - Data to predict
        w: np.array of shape (d,) - Weight vector
        b: float - Bias term

    Returns:
        predictions: np.array of shape (n,) - Predicted labels
    """
    return np.sign(X @ w + b)

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

def primal_svm_qp(X, y, C):
    N, d = X.shape

    # Construct P, q
    P = np.zeros((d + 1 + N, d + 1 + N))
    P[:d, :d] = np.eye(d)  # Only regularizing w, not b or xi
    
    q = np.hstack([np.zeros(d+1), C * np.ones(N)])

    # Construct G, h
    G = np.zeros((2*N, d + 1 + N))
    h = np.zeros(2*N)

    # First N rows: y_i (w^T x_i + b) >= 1 - xi_i
    G[:N, :d] = -y[:, None] * X
    G[:N, d] = -y
    G[:N, d+1:] = -np.eye(N)
    h[:N] = -1

    # Next N rows: xi_i >= 0
    G[N:, d+1:] = -np.eye(N)
    h[N:] = 0

    # Convert to CVXOPT format
    P = cvxopt_matrix(P)
    q = cvxopt_matrix(q)
    G = cvxopt_matrix(G)
    h = cvxopt_matrix(h)

    # Solve QP
    cvxopt_solvers.options['show_progress'] = False
    solution = cvxopt_solvers.qp(P, q, G, h)

    # Extract results
    w = np.array(solution['x'][:d]).flatten()
    b = np.array(solution['x'][d]).flatten()
    
    return w, b

sum = 0
for i in range(100):
    #generate random cov matrix
    Cs = np.random.rand(50, 50)
    Cs = Cs @ Cs.T  # Make it symmetric positive definite
    Ct = Cs

    Xs = np.random.multivariate_normal(mean=np.zeros(50), cov=Cs, size=100)
    Xs[:50] -= 5  # Shift class -1 in source domain
    Xs[50:] += 5  # Shift class 1 in source domain
    Xt = np.random.multivariate_normal(mean=np.zeros(50), cov=Ct, size=100)
    Xt[:50] -= 5  # Shift class -1 in target domain
    Xt[50:] += 5  # Shift class 1 in target domain
    y_s = np.ones(100)
    y_s[:50] = -1
    y_t = np.ones(100)
    y_t[:50] = -1


    Xs_coral = coral(Xs, Xt)
    # Train SVM on Xs_coral and y_s, and test on Xt
    w, b = primal_svm_qp(Xs_coral, y_s, C=1.0)
    predictions = predict(Xt, w, b)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_t, predictions)
    sum += accuracy
print(f"Average accuracy over 100 trials(Gaussian Data)(Same Covariances): {sum/100:.4f}")

sum = 0
for i in range(100):
    Xs = np.random.randn(100, 1)  # Random univariate data for source domain
    Xt = np.random.randn(100, 1)

    # Assign labels such that data is linearly separable
    y_s = np.ones(100)
    y_s[:50] = -1
    y_t = np.ones(100)
    y_t[:50] = -1

    # Shift the data to make it linearly separable
    Xs[:50] -= 3.75  # Shift class -1 in source domain
    Xs[50:] += 3.75  # Shift class 1 in source domain
    Xt[:50] -= 3.75  # Shift class -1 in target domain
    Xt[50:] += 3.75  # Shift class 1 in target domain

    Xt = Xt*np.sqrt(np.var(Xs)/np.var(Xt))  # Adjust variance of target domain
    Xs_coral = coral(Xs, Xt)

    # You can now train a classifier on (Xs_coral, y_s) and test it on Xt

    w, b = primal_svm_qp(Xs_coral, y_s, C=1.0)
    # Now you can use w and b to classify the target domain data
    predictions = predict(Xt, w, b)
    # Evaluate predictions
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_t, predictions)
    sum += accuracy
print(f"Average accuracy over 100 iterations(Random Data)(Same Covariances): {sum/100:.4f}")

sum = 0
for i in range(100):
    #generate random cov matrix
    Cs = np.random.rand(50, 50)
    Cs = Cs @ Cs.T  # Make it symmetric positive definite
    Ct = np.random.rand(50, 50)
    Ct = Ct @ Ct.T  # Make it symmetric positive definite

    Xs = np.random.multivariate_normal(mean=np.zeros(50), cov=Cs, size=100)
    Xs[:50] -= 5  # Shift class -1 in source domain
    Xs[50:] += 5  # Shift class 1 in source domain
    Xt = np.random.multivariate_normal(mean=np.zeros(50), cov=Ct, size=100)
    Xt[:50] -= 5  # Shift class -1 in target domain
    Xt[50:] += 5  # Shift class 1 in target domain
    y_s = np.ones(100)
    y_s[:50] = -1
    y_t = np.ones(100)
    y_t[:50] = -1


    Xs_coral = coral(Xs, Xt)
    # Train SVM on Xs_coral and y_s, and test on Xt
    w, b = primal_svm_qp(Xs_coral, y_s, C=1.0)
    predictions = predict(Xt, w, b)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_t, predictions)
    sum += accuracy
print(f"Average accuracy over 100 trials(Gaussian Data)(Different Covariances): {sum/100:.4f}")

sum = 0
for i in range(100):
    Xs = np.random.randn(100, 1)  # Random univariate data for source domain
    Xt = np.random.randn(100, 1)

    # Assign labels such that data is linearly separable
    y_s = np.ones(100)
    y_s[:50] = -1
    y_t = np.ones(100)
    y_t[:50] = -1

    # Shift the data to make it linearly separable
    Xs[:50] -= 3.75  # Shift class -1 in source domain
    Xs[50:] += 3.75  # Shift class 1 in source domain
    Xt[:50] -= 3.75  # Shift class -1 in target domain
    Xt[50:] += 3.75  # Shift class 1 in target domain

    Xs_coral = coral(Xs, Xt)

    # You can now train a classifier on (Xs_coral, y_s) and test it on Xt

    w, b = primal_svm_qp(Xs_coral, y_s, C=1.0)
    # Now you can use w and b to classify the target domain data
    predictions = predict(Xt, w, b)
    # Evaluate predictions
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_t, predictions)
    sum += accuracy

print(f"Average accuracy over 100 trials(Random Data)(Different Covariances): {sum/100:.4f}")