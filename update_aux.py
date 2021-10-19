import numpy as np
import pdb
import torch
import mars.tensor as mt

def soft(S, thd):
#     pdb.set_trace()
    return np.sign(S) * np.maximum(0, abs(S)-thd)
    
def update_aux(A, rho, eps = 1e-5):
    A = A.cpu().numpy()
    A = A.transpose(1,2,0)
    n_3 = np.size(A, 2)
    
    A_fft = np.fft.fft(A, axis = 2)
    End_Value = np.floor(n_3/2).astype(int)+1
    
    
    # pdb.set_trace()
    for i in range(End_Value+1):
        # S_hat <-- S * F
        U, S, V = mt.linalg.svd(A_fft[:, :, i])
        weight = rho * n_3/(S + eps)
        S_hat = soft(np.diag(S), np.diag(weight))
        
        # A_hat <-- U * S_hat * V
        A_fft[:,:,i] = U.dot(S_hat).dot(V) 
        if i > 0:
            
            A_fft[:, :, n_3-i] = U.conj().dot(S_hat).dot(V.conj())
    A = np.fft.ifft(A_fft, axis = 2)
    A = A.transpose(2,0,1)
    A = torch.from_numpy(np.real(A))
    return A