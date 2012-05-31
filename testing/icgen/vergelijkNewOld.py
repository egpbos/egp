# Vergelijk mijn code met fieldgen.f van Rien (20120126_100Mpc_64):
import egpICgen.gaussianField as GF
import egpICgen.powerSpectrum as PS

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
# load Rien data (power law):
rhopsi = np.memmap('20120126_100Mpc_64.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
psi1 = rhopsir[:,1].reshape((64,64,64))
# Met seed = 2:
# mean en std van de absolute density waarden wijken af met een factor 1.277 (Riens waardes zijn hoger)
# er gaat hier ook iets fout bij het integreren in het power spectrum, dus daar kan het misschien aan liggen.
# MET GEFIXTE CODE VOOR INTGTINV: nog maar een factor 1.229


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
out = GF.fieldgen(cosmology3, 100./0.7, 64, seed=2, returnFourier=True)
np.abs(out[0]).mean()
#Out[42]: 1.0979295301629779
out = GF.fieldgen(cosmology3, 100./0.7, 64, seed=3, returnFourier=True)
np.abs(out[0]).mean()
#Out[44]: 1.2434344997109292
out = GF.fieldgen(cosmology3, 100./0.7, 64, seed=4, returnFourier=True)
np.abs(out[0]).mean()
#Out[47]: 1.2940812282127823
out = GF.fieldgen(cosmology3, 100./0.7, 64, seed=5, returnFourier=True)
np.abs(out[0]).mean()
#Out[49]: 1.25594465788745


# load Rien data (trans3):
rhopsi = np.memmap('20120126_100Mpc_64_trans3.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[73]: 1.2216898202896118
# bij deze poging zijn mean en std een factor 1.11 hoger bij Rien (vgl met seed=2!)
# nog een paar realisaties proberen:
rhopsi = np.memmap('20120126_100Mpc_64_root5.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[69]: 1.1675454378128052
rhopsi = np.memmap('20120126_100Mpc_64_root6.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[82]: 1.1746419668197632
rhopsi = np.memmap('20120126_100Mpc_64_root7.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[87]: 1.1315021514892578
rhopsi = np.memmap('20120126_100Mpc_64_root8.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[92]: 1.1794736385345459
rhopsi = np.memmap('20120126_100Mpc_64_root9.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[97]: 1.1825929880142212

# AFGAANDE OP DEZE trans3 TEST LIJKEN DEZE CODES STATISTISCH IDENTIEKE RESULTATEN TE GEVEN

# POGING 3: trans1 weer, nu wat meer pogingen:
rhopsi = np.memmap('20120126_100Mpc_64_root10.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[101]: 7.437227725982666
rhopsi = np.memmap('20120126_100Mpc_64_root11.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[106]: 7.4500002861022949
rhopsi = np.memmap('20120126_100Mpc_64_root12.dat',dtype='float32')
rhopsir = rhopsi[9:].reshape((64**3,6))[:,1:-1]
rho = rhopsir[:,0].reshape((64,64,64))
np.abs(rho).mean()
#Out[111]: 7.4327888488769531

# DEZE RESULTATEN ZIJN DAN WEER TOTAAL TEGENSTRIJDIG, WANT ER ZIT EEN ZEER CONSTANT VERSCHIL TUSSEN:
out = GF.fieldgen(cosmology1, 100./0.7, 64, seed=6, returnFourier=True)
np.abs(out[0]).mean()
#Out[65]: 6.0431926959424072
out = GF.fieldgen(cosmology1, 100./0.7, 64, seed=4, returnFourier=True)
np.abs(out[0]).mean()
#Out[63]: 6.0573555184064469

# MISSCHIEN HEEFT HET TE MAKEN MET DE DIVIDE BY ZERO EN INVALID VALUES? NEE! ZIE HIERONDER
"""
Begin building rho array in Fourier space...
maxiter (200) exceeded. Latest difference = 7.460202e-06
Warning: divide by zero encountered in reciprocal
Warning: invalid value encountered in multiply
Done building rho.
Begin Fourier transformation on rho...
Fourier on rho done
Begin building psi arrays in Fourier space...
Warning: invalid value encountered in divide
Done building psi.
Begin Fourier transformations on psi...
Fourier on psi done.
"""


# IN ELK GEVAL HEEFT HET TE MAKEN MET DE kSPHERE: zet die uit en je krijgt:
out = GF.fieldgen(cosmology1, 100./0.7, 64, seed=6, returnFourier=True, ksphere=False)
np.abs(out[0]).mean()
#Out[67]: 7.4535291635708987
out = GF.fieldgen(cosmology1, 100./0.7, 64, seed=7, returnFourier=True, ksphere=False)
np.abs(out[0]).mean()
#Out[69]: 7.4405276016819855


# EXCELLENTE SEÑOR!!

# MAAR OOK HIER HEB JE NOG DIVIDE BY ZERO ENZO...

# Misschien ligt de "Warning: divide by zero encountered in reciprocal" aan
# de reciproke die je krijgt bij negatieve primn. Proberen:
cosmology1b = {
'omegaM': 1,
'omegaL': 0,
'h': 0.7,
'trans': PS.trans1,
'primn': 0,
'rth': 8./0.7,
'sigma0': 1
}
out = GF.fieldgen(cosmology1b, 100./0.7, 64, seed=2, returnFourier=True, ksphere=False)
# Jep. De "Warning: invalid value encountered in multiply" verdwijnt hier ook mee.
# Dat moet de vermenigvuldiging van inf met nul zijn geweest => NaN

# Dan nog de "Warning: invalid value encountered in divide". Die komt door de
# Z = 1.0j/k**2/boxlen * rho waar je met k[0,0,0] deelt door nul.


# Alle warnings zijn verklaard.


# RANDOM TESTS
import egpICgen.gaussianField as GF
import egpICgen.powerSpectrum as PS
cosmology = {
'omegaM': 0.3,
'omegaB': 0.04,
'omegaL': 0.7,
'h': 0.7,
'trans': PS.trans7,
'primn': 0.9,
'rth': 8./0.7,
'sigma0': 0.8
}
out = GF.fieldgen(cosmology, 100./0.7, 64, seed=4, returnFourier=True)


# VERGELIJKING VAN fieldgen EN fieldgenLessOld:
import egpICgen
cosmology = {
'omegaM': 0.3,
'omegaB': 0.04,
'omegaL': 0.7,
'h': 0.7,
'trans': egpICgen.trans7,
'primn': 0.9,
'rth': 8./0.7,
'sigma0': 0.8
}
out = egpICgen.fieldgen(cosmology, 100./0.7, 64, seed=4, returnFourier=True)
outOld = egpICgen.fieldgenLessOld(0.3,0.04,0.7,0.7,100.,64,7,0.9,8.,0.8, seed=4, returnFourier=True)
# zijn hetzelfde



# HET VOLGENDE IS VERGELIJKING VAN fieldgenOld MET WAT INMIDDELS fieldgenLessOld
# HEET, MAAR DESTIJDS DUS fieldgen WAS.

import egpICgen
out = egpICgen.fieldgen(0.3,0.04,0.7,0.7,100.,64,7,0.9,8.,0.8, returnFourier=True)
outNoKsphere = egpICgen.fieldgen(0.3,0.04,0.7,0.7,100.,64,7,0.9,8.,0.8, ksphere = False, returnFourier=True)
outOld = egpICgen.fieldgenOld(0.3,0.04,0.7,0.7,100.,64,7,0.9,8.,0.8, returnFourier=True)
outdif = out[0]-outOld[0]
outfrac = out[0]/outOld[0]
outcdif = out[4]-outOld[4]
outcfrac = out[4]/outOld[4]

pl.figure(1);pl.imshow(outOld[0][48], interpolation="nearest")
pl.figure(2);pl.imshow(out[0][48], interpolation="nearest")
pl.show()

# plot difference and fraction in real space rho
pl.figure(1);pl.imshow(outdif[48], interpolation="nearest")
pl.figure(2);pl.imshow(outfrac[48], interpolation="nearest")
pl.show()
pl.figure(1);pl.imshow(outdif[:,48], interpolation="nearest")
pl.figure(2);pl.imshow(outfrac[:,48], interpolation="nearest")
pl.show()
pl.figure(1);pl.imshow(outdif[:,:,48], interpolation="nearest")
pl.figure(2);pl.imshow(outfrac[:,:,48], interpolation="nearest")
pl.show()

# plot difference in fourier space rho; real and imaginary parts
pl.figure(1);pl.imshow(outcdif[21].real, interpolation="nearest")
pl.figure(2);pl.imshow(outcdif[21].imag, interpolation="nearest")
pl.show()
pl.figure(1);pl.imshow(outcdif[:,21].real, interpolation="nearest")
pl.figure(2);pl.imshow(outcdif[:,21].imag, interpolation="nearest")
pl.show()
pl.figure(1);pl.imshow(outcdif[:,:,21].real, interpolation="nearest")
pl.figure(2);pl.imshow(outcdif[:,:,21].imag, interpolation="nearest")
pl.show()

# plot difference in fourier space rho, only on k3 = 0,nyquist:
pl.figure(1);pl.imshow(outcdif[:,:,0].real, interpolation="nearest")
pl.figure(2);pl.imshow(outcdif[:,:,0].imag, interpolation="nearest")
pl.show()
pl.figure(1);pl.imshow(outcdif[:,:,32].real, interpolation="nearest")
pl.figure(2);pl.imshow(outcdif[:,:,32].imag, interpolation="nearest")
pl.show()



pl.figure(1);pl.imshow(outOld[4][30].real, interpolation="nearest")
pl.figure(2);pl.imshow(out[4][30].real, interpolation="nearest")
pl.show()

pl.figure(2);pl.imshow(out128[4][30].real, interpolation="nearest")
pl.show()
