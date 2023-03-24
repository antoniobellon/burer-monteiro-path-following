import numpy as np  

# Calculates the residual of an approximate solution to the non-linear Burer-Monteiro problem
def resid(A: np.ndarray, b: np.ndarray, C: np.ndarray, Y: np.ndarray, lam: np.ndarray):
    
    m = np.shape(A)[0]

    X = Y@Y.T  

    # Compute Lagrangian residual   
    Z = C-np.tensordot(lam, A, 1)
    lagran_grad = np.matmul(2*Z, Y)   
    resA = np.linalg.norm(lagran_grad.ravel(), np.inf)

    # Compute constraints residual 
    constr_err = -b 
    for i in range(m):
        constr_err[i] += np.dot(X.ravel(), A[i,:,:].ravel())  
    resB = np.linalg.norm(constr_err, np.inf) 

    return np.array([resA, resB]) 

# Calculates the residual of an approximate solution to the SDP problem
def SDP_resid(A: np.ndarray, b: np.ndarray, C: np.ndarray, X: np.ndarray, lam: np.ndarray):
    """
    Computes the residual for the SDP problem
    """   
    m=np.shape(A)[0]

    # Compute Lagrangian residual  
    lagran_grad = np.matmul(C-np.tensordot(lam, A, 1), X)  
    resA = np.linalg.norm(lagran_grad.ravel(), np.inf)
     
    # Compute constraints residual 
    constr_err = -b 
    for i in range(m):
        constr_err[i] += np.dot(X.ravel(), A[i,:,:].ravel())  
    resB = np.linalg.norm(constr_err, np.inf) 

    return np.array([resA, resB]) 