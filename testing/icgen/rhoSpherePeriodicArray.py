# Calculate the density in a sphere of certain radius around a position. Radius
# and position in grid-units (1 cell is 1 unit wide).

from egp.periodic import PeriodicArray

def rhoSphere(rhoIn, position, radius):
	gridsize = len(rhoIn)
	halfgrid = gridsize/2
	
	# reorder box for position to be at the center:
	i1,i2,i3 = np.mgrid[0:gridsize, 0:gridsize, 0:gridsize]
	i1 -= position[0]
	i2 -= position[1]
	i3 -= position[2]
	rho = PeriodicArray(rhoIn)[i1,i2,i3]
	
	# define sphere at center - 0.5*cell of the box
	x,y,z = np.mgrid[-halfgrid:halfgrid, -halfgrid:halfgrid, -halfgrid:halfgrid]
	sphere = np.sqrt(x**2+y**2+z**2) < radius
	
	return (rho*sphere).sum()/sphere.sum(), sphere.sum()

rho = out[0]
rho8s = []
radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 64
radius = radiusReal/boxsize*gridsize

for i in range(1000):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])


# test ground

# 9 Feb 2012:
gridsize = 8
halfgrid = gridsize/2
radius = 3
rhoIn = np.arange(gridsize**3).reshape((gridsize,gridsize,gridsize))
i1,i2,i3 = np.mgrid[0:gridsize, 0:gridsize, 0:gridsize]
position = np.int64(np.random.random(3)*8)
i1 -= position[0]
i2 -= position[1]
i3 -= position[2]
rho = PeriodicArray(rhoIn)[i1,i2,i3]
x,y,z = np.mgrid[-halfgrid:halfgrid, -halfgrid:halfgrid, -halfgrid:halfgrid]
sphere = np.sqrt(x**2+y**2+z**2) < radius


# Van voor 9 Feb 2012:

gridsize = 16
halfgrid = gridsize/2
position = [7,8,8]
i1,i2,i3 = np.mgrid[0:gridsize, 0:gridsize, 0:gridsize]
j1,j2,j3 = np.mgrid[0:gridsize, 0:gridsize, 0:gridsize]
i1 = (i1-position[0]+halfgrid)%gridsize
i2 = (i2-position[1]+halfgrid)%gridsize
i3 = (i3-position[2]+halfgrid)%gridsize
rhoc2 = rhoc[i1,i2,i3]


# tests 3 Feb 2012
import gaussianField as GF
import powerSpectrum as PS

# POGING 1: trans1 (power law)
cosmology1 = {
'omegaM': 1,
'omegaL': 0,
'h': 0.7,
'trans': PS.trans1,
'primn': -1,
'rth': 8./0.7,
'sigma0': 1
}
out = GF.fieldgen(cosmology1, 100./0.7, 64, seed=2, returnFourier=True)
rho = out[0]

radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 64
radius = radiusReal/boxsize*gridsize

rho8s = []
for i in range(1000):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# Geeft een std van 0.9, iets te laag.. (moet 1 zijn).
# IS MSS EEN KLEIN NUMERIEK EFFECT VAN DE GRIDSIZE!
# sphere.sum() bij gridsize = 64 en rth = 8/0.7 Mpc is 587.
# Dat is equivalent aan een bol met straal (587/np.pi/4*3)**(1./3) (in grid units).
# De radius die we bepaald hadden was (bij boxlen = 100) radius = 5.12.
# 5.12/(587/np.pi)**(1./3) = 0.98571976235269354

# POGING 2: trans3
cosmology3 = {
'omegaM': 1,
'omegaL': 0,
'h': 0.7,
'trans': PS.trans3,
'primn': -1,
'rth': 8./0.7,
'sigma0': 1
}
out2 = GF.fieldgen(cosmology3, 100./0.7, 64, seed=2, returnFourier=True)
rho = out2[0]

radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 64
radius = radiusReal/boxsize*gridsize

rho8s = []
for i in range(100):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# een std van ongeveer 0.8...

# POGING 3: Riens code
rhopsi = np.memmap('../andermans/rienICs/unconstrained/20120126_100Mpc_64.dat', dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))

rho8s = []
for i in range(100):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# std convergeert ook weer naar ongeveer 0.9 (lijkt ietsje hoger, 0.92, maar dat kan door de randomheid komen)

# POGING 4: andere seed
out4 = GF.fieldgen(cosmology3, 100./0.7, 64, seed=25, returnFourier=True)
rho = out4[0]

radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 64
radius = radiusReal/boxsize*gridsize

rho8s = []
for i in range(100):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# std van ongeveer 0.87
# Wrs komt de 0.89 factor ook niet lineair overeen
# met de factor verschil in sigma_8...
# wel misschien met iets als 1-(1-0.89)**(3/2) = 0.96351712730609063


# POGING 5: hogere gridresolutie voor betere sphere estimator
out5 = GF.fieldgen(cosmology3, 100./0.7, 128, seed=25, returnFourier=True)
rho = out5[0]

radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 128
radius = radiusReal/boxsize*gridsize
# in dit geval is de echte radius (10.24) iets groter dan de radius in rhoSphere
# 10.24/(4457/np.pi/4*3)**(1./3) = 1.0030331387477884

rho8s128 = []
for i in range(100):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s128.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# dit geeft een std van ~ 0.93

# POGING 6: nog een andere seed
out6 = GF.fieldgen(cosmology3, 100./0.7, 64, seed=253, returnFourier=True)
rho = out6[0]

radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 64
radius = radiusReal/boxsize*gridsize

rho8s = []
for i in range(100):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# ~ 0.85

out6b = GF.fieldgen(cosmology3, 100./0.7, 64, seed=653, returnFourier=True)
rho = out6b[0]

radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 64
radius = radiusReal/boxsize*gridsize

rho8s = []
for i in range(100):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# ~ 0.83

out6c = GF.fieldgen(cosmology3, 100./0.7, 64, seed=45626, returnFourier=True)
rho = out6c[0]

radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 64
radius = radiusReal/boxsize*gridsize

rho8s = []
for i in range(500):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# ~ 0.95

# POGING 7: ligt het misschien aan de ksphere cut?
out7 = GF.fieldgen(cosmology3, 100./0.7, 64, seed=45626, ksphere=False)
rho = out7[0]

radiusReal = 8/0.7
boxsize = 100/0.7
gridsize = 64
radius = radiusReal/boxsize*gridsize

rho8s = []
for i in range(500):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# maakt geen merkbaar verschil, std ~ 0.95

out7b = GF.fieldgen(cosmology3, 100./0.7, 64, seed=653, ksphere=False)
rho = out7b[0]

rho8s = []
for i in range(500):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# std ~ 0.85, ongeveer zelfde als 6b (zelfde root)

out7c = GF.fieldgen(cosmology3, 100./0.7, 64, seed=654, ksphere=False)
rho = out7c[0]

rho8s = []
for i in range(500):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# std ~ 1.06... misschien dan toch hierboven gewoon verkeerde root keuzes?

out7d = GF.fieldgen(cosmology3, 100./0.7, 64, seed=656, ksphere=False)
rho = out7d[0]

rho8s = []
for i in range(500):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# std ~ 0.81

out7e = GF.fieldgen(cosmology3, 100./0.7, 64, seed=655, ksphere=False)
rho = out7e[0]

rho8s = []
for i in range(500):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# std ~ 0.99!

out7f = GF.fieldgen(cosmology3, 100./0.7, 64, seed=657, ksphere=False)
rho = out7f[0]

rho8s = []
for i in range(1000):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# std ~ 0.942

out7g = GF.fieldgen(cosmology3, 100./0.7, 64, seed=658, ksphere=False)
rho = out7g[0]

rho8s = []
for i in range(1000):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])

np.std(rho8s)
# std ~ 0.893

out7h = GF.fieldgen(cosmology3, 100./0.7, 64, seed=658, ksphere=False)
rho = out7h[0]

rho8s = []
for i in range(1000):
  x,y,z=np.random.randint(0,gridsize),np.random.randint(0,gridsize),np.random.randint(0,gridsize)
  rho8s.append(rhoSphere(rho, np.array((x,y,z)), radius)[0])


# POGING 8: voor gemak, andersom beredeneren: neem cells met volume = 4/3pi*8^3
cellsize = (4./3*np.pi)**(1./3)*(8/0.7) # == boxsize/gridsize
gridsize = 64
boxsize = gridsize*cellsize

out8 = GF.fieldgen(cosmology3, boxsize, gridsize, ksphere=True)
rho = out8[0]

rho.std()
# std ~ 1.12 voor verschillende random seeds met cosmology3, ~ 1.13 met cosmo1 (beide met ksphere=False)
# !!! std ~ 0.915 met ksphere=True (cosmo1)! Das andere koek...
#     std ~ 1.05 met ksphere=True (cosmo3).

gridsize = 128
boxsize = gridsize*cellsize

out8b = GF.fieldgen(cosmology3, boxsize, gridsize, ksphere=True)
rho = out8b[0]

rho.std()
# std ~ 1.05 met ksphere=True (cosmo3) op 128^3, geen resolutie effect dus.

gridsize = 256
boxsize = gridsize*cellsize

out8c = GF.fieldgen(cosmology3, boxsize, gridsize, ksphere=True)
rho = out8c[0]

rho.std()
# std ~ 1.05; GEEN RESOLUTIE EFFECT MET COSMO3.

gridsize = 128
boxsize = gridsize*cellsize

out8b = GF.fieldgen(cosmology1, boxsize, gridsize, ksphere=True)
rho = out8b[0]

rho.std()
