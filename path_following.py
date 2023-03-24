import numpy as np 
import scipy 
import time   
import residual
import linearized_kkt as lk  

class _PathFollowing:

    def __init__(self, n: int, m: int, r: int):

        """ Constructor pre-allocates and pre-computes persistent data structures. """ 

        # Storing problem dimensions
        self._n = n
        self._m = m
        self._r = r

        # Create all necessary modules and pre-allocate the workspace 
        self._LinearizedKKTsystem = lk._LinearizedKKTsystem(n=n, m=m, r=r)
        self._linear_KKT_sol      = np.zeros((int(n*r+m+0.5*r*(r-1)),))
        self._candidate_Y         = np.zeros((n, r)) 
        self._candidate_lam       = np.zeros((m,)) 
        self._Y                   = np.zeros((n, r)) 
        self._X                   = np.zeros((n, n))
        self._lam                 = np.zeros((m,))   
 
        # Initializing variables for storing residuals and runtime
        self.times                  = []
        self._primal_solutions_list = []
        self._PC_residuals          = []   
        self._PC_SDP_residuals      = []   
        self._PC_runtime            = 0.0
        
    def run(self, 
            A                  : np.ndarray, 
            b                  : np.ndarray, 
            C                  : np.ndarray, 
            Y_0                : np.ndarray, 
            lam_0              : np.ndarray, 
            initial_time       : float,
            final_time         : float,
            initial_stepsize   : float,
            gamma_1            : float,
            gamma_2            : float,
            residual_tolerance : float,
            STEPSIZE_TUNING    : bool,
            FOLLOW_GRID        : bool):   
            
        iteration       = 0 
        reduction_steps = 0
         
        # Get copies of all problem parameters  
        dt         = initial_stepsize  
        curr_time  = initial_time
        next_time  = initial_time + dt
        n, m, r    = self._n, self._m, self._r

        # STEPSIZE_TUNING indicates whether to tune the stepsize by a factor_gamma2 and check the residual, or not 
        if not STEPSIZE_TUNING: 
            gamma_2=1
            residual_tolerance=np.Inf

        # Store initial solution as the current iterate
        np.copyto(self._Y, Y_0)
        np.copyto(self._X, np.matmul(Y_0,Y_0.T))
        np.copyto(self._lam, lam_0)   

        res = residual.resid(A=A(initial_time), b=b(initial_time),  C=C(initial_time),  Y=Y_0, lam=lam_0)
        self._PC_residuals.append(max(res))

        SDP_res = residual.SDP_resid(A=A(initial_time), b=b(initial_time),  C=C(initial_time),  X=self._X, lam=lam_0)
        self._PC_SDP_residuals.append(max(SDP_res))

        while curr_time < final_time:

            self.times.append(curr_time)

            start_time = time.time() 
            
            # Compute linearized KKT system
            H = self._LinearizedKKTsystem.computeMatrix(A=A(next_time), C=C(next_time), Y=self._Y, lam=self._lam) 
            k = self._LinearizedKKTsystem.computeRhside(A=A(next_time), b=b(next_time), C=C(next_time), Y=self._Y, X=self._X)   
        
            # Solve the system and store the candidate solutions, recording the execution time
            np.copyto(self._linear_KKT_sol, scipy.linalg.solve(H, k, assume_a='sym'))
            np.copyto(self._candidate_Y, self._Y + np.reshape(self._linear_KKT_sol[:n*r],(r,n)).T)
            np.copyto(self._candidate_lam, self._linear_KKT_sol[n*r:n*r+m]) 

            self._PC_runtime += time.time() - start_time 
            
            # Compute the residuals
            res = residual.resid(A=A(next_time), b=b(next_time), C=C(next_time), Y=self._candidate_Y, lam=self._candidate_lam) 

            if reduction_steps >= 1000:  
                print("1000 stepsize reductions threshold exceeded") 
                break
            
            if max(res)>residual_tolerance:  
                dt *= gamma_1
                next_time = curr_time + dt  
                reduction_steps += 1
                
            # ... or keep a constant stepsize
            else:

                reduction_steps = 0
 
                np.copyto(self._Y, self._candidate_Y)
                np.copyto(self._lam, self._candidate_lam) 
                np.copyto(self._X, np.matmul(self._Y,self._Y.T))  
                self._primal_solutions_list.append(self._X)
                
                self._PC_residuals.append(max(res))
                SDP_res = residual.SDP_resid(A=A(next_time), b=b(next_time), C=C(next_time),  X=self._X, lam=self._lam)
                self._PC_SDP_residuals.append(max(SDP_res))
                curr_time += dt 
                iteration += 1  
                
                next_time = curr_time+dt 
                
                if FOLLOW_GRID: 
                    distance_to_grid = initial_stepsize-curr_time%initial_stepsize
                    if distance_to_grid> 1.e-9:
                        dt = min(initial_stepsize-curr_time%initial_stepsize, gamma_2 * dt)
                    else: dt =  gamma_2 * dt
                    
            
               