import scs
import numpy as np 
from scipy import sparse
import sys
import io
import time
import residual

# The vec function as documented in api/cones
def vec(S): 
    n = S.shape[0]
    S = np.copy(S) 
    S *= np.sqrt(2) 
    S[range(n), range(n)] /= np.sqrt(2) 
    return S[np.triu_indices(n)]

# The mat function as documented in api/cones
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S

class _SCS_tracker:

    def __init__(self,): 
  
        self._SCS_runtime   = 0.0  
        self._SCS_residuals = []  

    def run(self, 
            A: np.ndarray, 
            b: np.ndarray,  
            C: np.ndarray, 
            
            initial_time:  float,
            final_time:    float, 
            stepsize:      float, 
            scs_tolerance: float 
            ):  

        text_trap = io.StringIO()
        sys.stdout = text_trap
        
        m = np.shape(A(initial_time))[0]
        n = np.shape(A(initial_time))[1]     
        
        iteration = 0
         
        curr_time = initial_time
        next_time = initial_time + stepsize
     
        def A_vectorized(time):

            A_vect = np.ndarray((m,int(n*(n+1)/2)))
            for i in range(m):  
                A_vect[i] = vec(A(time)[i])
                
            return sparse.csr_matrix(A_vect.T) 
        
        timer = time.time()

        data = dict(P=None, A=A_vectorized(initial_time), b=vec(C(initial_time)) , c=-b(initial_time))
        cone = dict(s=n)
        solver = scs.SCS(data, cone, eps_abs=scs_tolerance, eps_rel=scs_tolerance)
        sol = solver.solve(warm_start=True, x=None, y=None, s=None)

        SCS_res = residual.SDP_resid(A=A(initial_time), b=b(initial_time), C=C(initial_time), X=mat(sol["y"]), lam=sol["x"])
        self._SCS_residuals.append(max(SCS_res))   

        self._SCS_runtime += time.time() - timer 

        while curr_time < final_time:  

            timer  = time.time()

            data   = dict(P=None, A=A_vectorized(next_time), b=vec(C(next_time)) , c=-b(next_time))
            solver = scs.SCS(data, cone, eps_abs=1e-6, eps_rel=1e-6)
            sol    = solver.solve(warm_start=True, x=sol["x"], y=sol["y"], s=sol["s"])

            curr_X = mat(sol["y"])
            curr_lam = sol["x"]
            
            SCS_res = residual.SDP_resid(A=A(next_time), b=b(next_time), C=C(next_time), X=curr_X, lam=curr_lam)
            self._SCS_residuals.append(max(SCS_res))    

            self._SCS_runtime += time.time() - timer

            curr_time += stepsize
            next_time = curr_time+stepsize 

            # if iteration%10==0: print("\nITERATION", iteration) 
            
            iteration += 1
            
        sys.stdout = sys.__stdout__