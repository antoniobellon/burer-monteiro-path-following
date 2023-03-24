import numpy as np  
 
class _LinearizedKKTsystem:

    def __init__(self, n: int, m: int, r: int) -> None:

        self._n                    = n
        self._m                    = m
        self._r                    = r 
        nvars                      = n*r
        rantisymm                  = int((r-1)*r/2) 
        self._gradientGconstraints = np.full((nvars, m), fill_value=0.0)
        self._gradientHconstraints = np.full((nvars, rantisymm), fill_value=0.0)
        self._LinearizedKKTmatrix  = np.full((nvars+m+rantisymm, nvars+m+rantisymm), fill_value=0.0)
        self._LinearizedKKTrhs     = np.full((nvars+m+rantisymm), fill_value=0.0)

    def computeMatrix(self, A: np.ndarray, C: np.ndarray, Y: np.ndarray, lam: np.ndarray) -> np.ndarray:
        
        n     = self._n                    
        m     = self._m                   
        r     = self._r              
        nvars = n * r

        Hess = 2*np.kron(np.eye(r), C-np.tensordot(lam, A, 1))
        np.copyto(self._LinearizedKKTmatrix[:nvars,:nvars], Hess)

        for i in range(m): 
            np.copyto(self._gradientGconstraints.T[i], -2*(np.ravel(np.matmul(A[i],Y).T))) 

        np.copyto(self._LinearizedKKTmatrix[:nvars,nvars:nvars+m], self._gradientGconstraints)
        np.copyto(self._LinearizedKKTmatrix[nvars:nvars+m,:nvars], self._gradientGconstraints.T) 
        k = 0
        for i in range(r):  

            Y_i_col = np.reshape(Y[:,i], (n, 1))

            for j in range(i+1,r): 
                
                Y_j_col = np.reshape(Y[:,j], (n, 1))

                part_1 = Y_i_col @ np.eye(1, r, j) 
                part_2 = Y_j_col @ np.eye(1, r, i)

                np.copyto(self._gradientHconstraints[:,k],((part_2-part_1).T).ravel())
                k += 1  
        
        np.copyto(self._LinearizedKKTmatrix[:nvars,nvars+m:], self._gradientHconstraints )
        np.copyto(self._LinearizedKKTmatrix[nvars+m:,:nvars], self._gradientHconstraints.T )
        
        return self._LinearizedKKTmatrix

    def computeRhside(self,  A: np.ndarray, b: np.ndarray, C: np.ndarray, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        
        nvars = self._n*self._r

        np.copyto(self._LinearizedKKTrhs[:nvars],-2*(np.matmul(C,Y).T).ravel())

        for i in range(self._m):
            self._LinearizedKKTrhs[nvars+i] = np.tensordot(A[i],X) - b[i]
        
        return self._LinearizedKKTrhs    