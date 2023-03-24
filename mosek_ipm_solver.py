import numpy as np  
import scipy as sc
import mosek 
import time 
import sys 
  
def myStream(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def _get_SDP_solution(A: np.ndarray, b: np.ndarray, C: np.ndarray, rel_gap_tol: float):
    
    n = np.shape(C)[0]
    m = np.shape(A)[0]

    # Make mosek environment
    
    with mosek.Env() as env:

        # Create a task object and attach log stream printer
        with env.Task(0, 0) as task:
 
            start_time = time.time() 

            # Append  empty constraints and matrix variables 
            task.appendcons(m) 
            task.appendbarvars([n]) 

            for i in range(m): 
                # print(repr(A[i]))
                barak   = sc.sparse.find(sc.sparse.tril(A[i]))[0].tolist()
                baraj   = sc.sparse.find(sc.sparse.tril(A[i]))[1].tolist()
                baraval = sc.sparse.find(sc.sparse.tril(A[i]))[2].tolist()

                syma = task.appendsparsesymmat(n, barak, baraj, baraval)
                task.putbaraij(i, 0, [syma], [1.0])   
            
            # Constraints RHS
            blc = b
            buc = blc

            # Set the bounds on constraints. 
            for i in range(m):
                task.putconbound(i, mosek.boundkey.fx, blc[i], buc[i])

            # Trace objective
            barci   = sc.sparse.find(sc.sparse.tril(C))[0].tolist()
            barcj   = sc.sparse.find(sc.sparse.tril(C))[1].tolist()
            barcval = sc.sparse.find(sc.sparse.tril(C))[2].tolist()

            symc = task.appendsparsesymmat(n, barci, barcj, barcval)
            task.putbarcj(0, [symc], [1.0])

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)
            
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, rel_gap_tol) 
            task.putintparam(mosek.iparam.infeas_report_auto, mosek.onoffkey.on)

            # Solve the problem and print summary
            task.optimize()
            task.solutionsummary(mosek.streamtype.log)
            # task.analyzeproblem(mosek.streamtype.log)

            # Get status information about the solution 
            solsta = task.getsolsta(mosek.soltype.itr)

            run_time = time.time() - start_time

            if (solsta == mosek.solsta.optimal):
                lenbarvar = n * (n + 1) / 2
                barx = [0.] * int(lenbarvar)
                task.getbarxj(mosek.soltype.itr, 0, barx)
                y = [0.] * m
                task.gety(mosek.soltype.itr,y)
                
                X_lt = np.ndarray((n, n))
                for k in range(n):
                    X_lt[k] = np.append(np.zeros(k), barx[:n-k])
                    barx = barx[n-k:] 
                X = X_lt+X_lt.T
                for i in range(n):
                    X[i,i] *= .5
                
                return run_time, X, y
                
            elif (solsta == mosek.solsta.prim_infeas_cer):
                print("Primal infeasibility certificate found.\n")
            elif (solsta == mosek.solsta.dual_infeas_cer):
                print("Dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")

def _get_initial_point(A: np.ndarray, b: np.ndarray, C: np.ndarray, rel_gap_tol: float):
     
    run_time, X, lam0 = _get_SDP_solution(A=A, b=b, C=C, rel_gap_tol=rel_gap_tol)
    rank = np.linalg.matrix_rank(X, 1.e-8)
    
    # The eigenvalues in descending order
    eig_dec = np.linalg.eigh(X)
    eig_vals_X = np.flip(eig_dec[0]) 
    eig_vecs_X = np.flip(eig_dec[1],1)
    
    Y= np.ndarray((np.shape(X)[0],rank),dtype=np.float)
    for i in range(rank):
        Y[:,i] = eig_vecs_X[:,i] * np.sqrt(eig_vals_X[i]) 
     
    return Y, rank, lam0, run_time