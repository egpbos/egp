# Een nog enigszins begrijpelijk voorbeeld van symmetrie aanbrengen in een 
# matrix die met ifft naar een reele matrix moet worden getransformeerd.

from time import time

gridsize=8
halfgrid=gridsize/2

m0 = np.random.random((gridsize,gridsize,halfgrid+1)) + 2.0j*(np.random.random((gridsize,gridsize,halfgrid+1))-0.5)
m7 = np.random.random(7)
real7x,real7y,real7z = np.mgrid[0:2,0:2,0:2][:, ]*halfgrid
real7x,real7y,real7z = real7x.ravel()[1:],real7y.ravel()[1:],real7z.ravel()[1:]

# HELEMAAL UITGESCHREVEN PER SYMMETRISCH ONDERDEEL VAN HET FOURIER VELD
time0 = time()
m = m0.copy()
m[real7x,real7y,real7z] = m7
m[0,0,0] = 0.
# plane intersections
m[0,-1:halfgrid:-1,0] = m[0,1:halfgrid,0].conjugate()
m[0,-1:halfgrid:-1,halfgrid] = m[0,1:halfgrid,halfgrid].conjugate()
m[-1:halfgrid:-1,0,0] = m[1:halfgrid,0,0].conjugate()
m[-1:halfgrid:-1,0,halfgrid] = m[1:halfgrid,0,halfgrid].conjugate()
m[halfgrid,-1:halfgrid:-1,0] = m[halfgrid,1:halfgrid,0].conjugate()
m[halfgrid,-1:halfgrid:-1,halfgrid] = m[halfgrid,1:halfgrid,halfgrid].conjugate()
m[-1:halfgrid:-1,halfgrid,0] = m[1:halfgrid,halfgrid,0].conjugate()
m[-1:halfgrid:-1,halfgrid,halfgrid] = m[1:halfgrid,halfgrid,halfgrid].conjugate()
# rest of the planes
m[-1:halfgrid:-1,-1:halfgrid:-1,0] = m[1:halfgrid,1:halfgrid,0].conjugate()
m[-1:halfgrid:-1,-1:halfgrid:-1,halfgrid] = m[1:halfgrid,1:halfgrid,halfgrid].conjugate()
m[-1:halfgrid:-1,1:halfgrid,0] = m[1:halfgrid,-1:halfgrid:-1,0].conjugate()
m[-1:halfgrid:-1,1:halfgrid,halfgrid] = m[1:halfgrid,-1:halfgrid:-1,halfgrid].conjugate()
# Klopt het eigenlijk wel zo met die negatieve indices in de rest van de planes?
# Moeten dat geen transposes zijn?
print "Took %10.7f seconds." % (time()-time0)


# ELEGANT/LEESBAAR UITGESCHREVEN (~2 maal langzamer dan helemaal uitgeschreven):
time0 = time()
mE = m0.copy()
mE[real7x,real7y,real7z] = m7
mE[0,0,0] = 0.
# N.B.: xi en yi zijn hier omgedraaid zodat hetzelfde resultaat als met de for-
#       loop en de hierboven uitgeschreven versies wordt behaald. xi en yi
#       kunnen ook prima andersom; geeft ook een goed gesymmetriseerde matrix,
#       maar dan met de (geconjugeerde) random getallen uit het andere vlak.
xi, yi, zi = np.ogrid[0:halfgrid+1,0:gridsize,0:halfgrid+1:halfgrid]
xj, yj, zj = np.ogrid[gridsize:halfgrid-1:-1,gridsize:0:-1,0:halfgrid+1:halfgrid]
xj[0], yj[:,0] = 0,0
# De volgende actie moet in 2 helften, omdat je anders geen symmetrische matrix
# krijgt, maar gewoon de volledige matrix gespiegeld en geconjugeerd.
mE[xj,yj[:,:halfgrid],zj] = mE[xi,yi[:,:halfgrid],zi].conj()
mE[xj,yj[:,halfgrid:],zj] = mE[xi,yi[:,halfgrid:],zi].conj()
print "Took %10.7f seconds." % (time()-time0)


# ZELFDE RESULTAAT ALS DE VOORHEEN GEBRUIKTE FOR-LOOP (ook in Riens code)
# N.B.: Deze for-loop is verkeerd! In de echte for-loop wordt elke stap een nieuw
# random nummer toegekend en daarna pas geconjugeerd. Omdat we hier in de eerste
# run over k1,k2 al conjugeren en zo het getal dat we in de bijbehorende k1,k2
# veranderen in de volgende run over die eerste k1,k2 weer tegenkomen is dat hier
# dus een ander getal dan we in de for-loop in fieldgenOld zouden krijgen; daar
# wordt het random getal uit de tweede run namelijk gebruikt, terwijl die hier dus
# al in de eerste run verdwenen is. Om dit te fixen moet je in de echte code in
# de elegante versie xi en xj's omwisselen vergeleken met wat hierboven staat;
# dus m[xi,yi[:,:halfgrid],zi] = m[xj,yj[:,:halfgrid],zj].conj() in plaats van
# m[xj,yj[:,:halfgrid],zj] = m[xi,yi[:,:halfgrid],zi].conj().
time0 = time()
mloop = m0.copy()
for k1 in range(0,gridsize):
	for k2 in range(0,gridsize):
		for k3 in range(2):
			# Do k3=0 and k3=n12.
			# Actually this part will go over the two planes twice, keeping
			# only the numbers determined in the second run. This is because
			# every run actually determines two numbers, through symmetry.
			# The loss in efficiency however is only O(n^2), whereas the
			# rest of the program goes as O(n^3).
			if k3 == 1: k3 = halfgrid
				# Determine conjugates using symmetry.
			mloop[(gridsize-k1)%gridsize][(gridsize-k2)%gridsize][(gridsize-k3)%gridsize] = (mloop[k1][k2][k3]).conjugate()
mloop[real7x,real7y,real7z] = m7
mloop[0,0,0] = 0.
print "Took %10.7f seconds." % (time()-time0)


# NU HET HELE GEDOE EVEN MET EEN VOLLEDIGE k-space BOX.
# DEZE BOX GEEFT IN EEN VOLLEDIGE FOURIER TRANSFORMATIE HETZELFDE RESULTAAT
# ALS DE REAL FFT OP DE HALF(+1) BOX (binnen numerieke precisie):
time0 = time()
mEf = np.random.random((gridsize,gridsize,gridsize)) + 2.0j*(np.random.random((gridsize,gridsize,gridsize))-0.5)
mEf[:,:,:halfgrid+1] = m0
mEf[real7x,real7y,real7z] = m7
mEf[0,0,0] = 0.
# N.B.: xi en yi zijn hier omgedraaid zodat hetzelfde resultaat als met de for-
#       loop en de hierboven uitgeschreven versies wordt behaald. xi en yi
#       kunnen ook prima andersom; geeft ook een goed gesymmetriseerde matrix,
#       maar dan met de (geconjugeerde) random getallen uit het andere vlak.
xi, yi, zi = np.ogrid[0:gridsize,0:gridsize,0:halfgrid+1]
xj, yj, zj = np.ogrid[gridsize:0:-1,gridsize:0:-1,gridsize:halfgrid-1:-1]
xj[0], yj[:,0], zj[:,:,0] = 0,0,0
# De volgende actie moet zelfs in 4 keer, zelfde reden als de 2 helften hierboven.
mEf[xj[:halfgrid],yj[:,:halfgrid],zj] = mEf[xi[:halfgrid],yi[:,:halfgrid],zi].conj()
mEf[xj[:halfgrid],yj[:,halfgrid:],zj] = mEf[xi[:halfgrid],yi[:,halfgrid:],zi].conj()
mEf[xj[halfgrid:],yj[:,:halfgrid],zj] = mEf[xi[halfgrid:],yi[:,:halfgrid],zi].conj()
mEf[xj[halfgrid:],yj[:,halfgrid:],zj] = mEf[xi[halfgrid:],yi[:,halfgrid:],zi].conj()
print "Took %10.7f seconds." % (time()-time0)


# DIT FIXEN (ook voor N-D):
def symmetrizeMatrix4D(m):
	gridsize = m.shape[0]
	halfgrid = gridsize/2
	x1i, x2i, x3i, x4i = np.ogrid[0:halfgrid+1,0:gridsize,0:gridsize,0:halfgrid+1:halfgrid]
	x1j, x2j, x3j, x4j = np.ogrid[gridsize:halfgrid-1:-1,gridsize:0:-1,gridsize:0:-1,0:halfgrid+1:halfgrid]
	x1j[0], x2j[:,0], x3j[:,:,0] = 0,0,0
	mE[x1j,x2j[:,:halfgrid], x3j, x4j] = mE[x1i,x2i[:,:halfgrid],x3i,x4i].conj()
	mE[x1j,x2j[:,halfgrid:], x3j, x4j] = mE[x1i,x2i[:,halfgrid:],x3i,x4i].conj()
