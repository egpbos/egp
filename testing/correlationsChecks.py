rhoC.calculate_constraint_correlations()
fast = rhoC.xi_ij_inverse.copy()
rhoC.calculate_constraint_correlations(fast=False)
slow = rhoC.xi_ij_inverse.copy()
fast - slow

# 23 Feb 2012: correlations checks
ki = k_i_grid(rhoC2.gridsize, rhoC2.boxlen)
k = k_abs_grid(rhoC2.gridsize, rhoC2.boxlen)
Rg = rhoC2.constraints[0].scale.scale

# H_1.conj * H_2
summable = ki[0]*ki[1]*rhoC2.power(k)*np.exp(-k**2 * Rg**2)
1/(np.sum(2*summable[...,1:-1]) + np.sum(summable[...,(0,-1)]))
rhoC2.xi_ij_inverse[1,2]
# DIT GAAT BEST FOUT!

# H_1.conj * H_1
summable = ki[0]*ki[0]*rhoC2.power(k)*np.exp(-k**2 * Rg**2)
1/(np.sum(2*summable[...,1:-1]) + np.sum(summable[...,(0,-1)]))
rhoC2.xi_ij_inverse[1,1]

# H_0.conj * H_0
summable = rhoC3.power(k)*np.exp(-k**2 * Rg**2)
1/(np.sum(2*summable[...,1:-1]) + np.sum(summable[...,(0,-1)]))
rhoC3.xi_ij_inverse[0,0]

