import scipy as sc
import numpy as np 
from   scipy import stats
 
normal_seed = stats.norm(loc=0, scale=10).rvs 

class _ProblemCreator:

    def __init__(self):
        pass

    def _create_random_problem(self, n, m, sparsity_coefficient): 

        A_init = np.ndarray((m, n, n)) 
        A_pert = np.ndarray((m, n, n)) 
 
        for i in range(m):
            
            rand_A_init = sc.sparse.random(n, n, density=sparsity_coefficient, data_rvs=normal_seed).toarray()
            rand_A_pert = sc.sparse.random(n, n, density=sparsity_coefficient, data_rvs=normal_seed).toarray() 

            A_init[i] = rand_A_init + rand_A_init.T
            A_pert[i] = rand_A_pert + rand_A_pert.T 

        A_init = np.array(A_init)
        A_pert = np.array(A_pert)  

        b_init = sc.sparse.random(m, 1, density=sparsity_coefficient, data_rvs=normal_seed).toarray().T[0]
        b_pert = sc.sparse.random(m, 1, density=sparsity_coefficient, data_rvs=normal_seed).toarray().T[0]
        
        q1, _ = np.linalg.qr(np.random.rand(n, n))
        q2, _ = np.linalg.qr(np.random.rand(n, n))
        C_init = q1.T @ np.diag([x**2 for x in sc.sparse.random(n, 1, density=1, data_rvs=normal_seed).toarray().T[0]]) @ q1
        C_pert = q2.T @ np.diag([x**2 for x in sc.sparse.random(n, 1, density=1, data_rvs=normal_seed).toarray().T[0]]) @ q2

        def A(time: float): 
            return A_init + time*A_pert 
        
        def b(time: float):
            return b_init + time*b_pert 
        
        def C(time: float):
            return C_init + time*C_pert 

        return A, b, C

    def _create_MaxCut(self, n: int):
        
        A_init = np.ndarray((n, n, n))  
        for i in range(n): 
            constraint_i = np.zeros((n,n))
            constraint_i[i,i] = 1  
            A_init[i]=constraint_i
    
        b_init = np.ones(n)

        rand_C_init = sc.sparse.random(n, n, density=0.5, data_rvs=normal_seed).toarray()
        NZ = rand_C_init.nonzero() 
        I = NZ[0]
        J = NZ[1]
        nr_NZ = len(I)
        V = np.random.rand(nr_NZ,) 
        rand_C_pert = sc.sparse.coo_matrix((V,(I,J)),shape=(n,n)).toarray()*10
        C_init = np.abs(rand_C_init - rand_C_init.T) 
        C_pert = np.abs(rand_C_pert - rand_C_pert.T)     

        def A(time: float): return A_init
        def b(time: float): return b_init 
        def C(time: float): return C_init + time*C_pert 
       
        return A, b, C 