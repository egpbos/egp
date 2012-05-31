N = 3
gridsize = 64
a_inv_for = np.empty((N, N), dtype=np.complex256)
a_for = np.empty((N, N), dtype=np.complex256)
b_for = np.empty((N, N), dtype=np.complex256)
np.random.seed(0)
constraints = np.sqrt(2)*(np.random.randn(N, gridsize, gridsize, gridsize/2+1) + 1j* np.random.randn(N, gridsize, gridsize, gridsize/2+1)).astype(np.complex256)

#constraints[

for i, Hi in enumerate(constraints):
    for j, Hj in enumerate(constraints):
        a = np.sum( 2*Hi[...,1:-1].conj()*Hj[...,1:-1] )
        b = a + np.sum( Hi[...,(0,-1)].conj()*Hj[...,(0,-1)] )
        a_inv_for[i,j] = 1/b
        a_for[i,j] = a
        b_for[i,j] = b


H = constraints
a_np = 2*np.einsum('iklm,jklm->ij', H[...,1:-1].conj(), H[...,1:-1])
b_np = a_np + np.einsum('iklm,jklm->ij', H[...,(0,-1)].conj(), H[...,(0,-1)])
#a_np = 2*np.tensordot(H[...,1:-1].conj(), H[...,1:-1], axes=((1,2,3),(1,2,3)))
#b_np = a_np + np.tensordot(H[...,(0,-1)].conj(), H[...,(0,-1)], axes=((1,2,3),(1,2,3)))
a_inv_np = 1/b_np

#a = 2*np.einsum('iklm,jklm->ij', H[...,1:-1].conj(), H[...,1:-1])

print np.mean(np.abs(b_np-b_for))