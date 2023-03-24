from traceback import print_tb
import numpy as np  
import mosek_ipm_solver as mis
import residual
import time 

class _IPM_tracker:

    def __init__(self,): 

        self._IPM_residuals = []  
        self._IPM_runtime   = 0.0  

    def run(self, 
            A                             : np.ndarray, 
            b                             : np.ndarray, 
            C                             : np.ndarray, 
            initial_time                  : float, 
            final_time                    : float, 
            stepsize                      : float,
            ipm_tolerance                 : float): 

        iteration = 0
         
        curr_time  = initial_time
        next_time = initial_time + stepsize
        
        run_time_SDP_0, X_0, lam_0 = mis._get_SDP_solution(A(curr_time), b(curr_time), C(curr_time), rel_gap_tol=ipm_tolerance)
        self._IPM_runtime += run_time_SDP_0 

        IPM_res = residual.SDP_resid(A=A(curr_time), b=b(curr_time),  C=C(curr_time),  X=X_0, lam=lam_0)
        self._IPM_residuals.append(max(IPM_res))
           
        while curr_time < final_time:    
        
            run_time, X, lam = mis._get_SDP_solution(A(next_time), b(next_time), C(next_time), rel_gap_tol=ipm_tolerance)
            self._IPM_runtime += run_time 

            IPM_res = residual.SDP_resid(A=A(next_time), b=b(next_time),  C=C(next_time),  X=X, lam=lam)
            self._IPM_residuals.append(max(IPM_res))    
            
            curr_time += stepsize
            next_time = curr_time+stepsize 
            
            iteration += 1