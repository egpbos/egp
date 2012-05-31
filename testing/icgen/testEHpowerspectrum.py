import egp.gaussianField as GF
import egp.powerSpectrum as PS

cosmology8 = {
'omegaM': 0.268,
'omegaB': 0.044,
'omegaL': 0.732,
'h': 0.704,
'trans': PS.trans8,
'primn': 0.947,
'rth': 8./0.7,
'sigma0': 0.776,
'TCMB': 2.7
}

logk = np.linspace(-3,3,601)
k = 10**logk

power = PS.powerspectrum(k, 1., cosmology8)
logpower = np.log10(power)

D2Dolag = np.loadtxt("/Users/users/pbos/dataserver/GO/gsims/ICs/input_spectrum.txt")

powerDolag = D2Dolag[:,1] - 3*D2Dolag[:,0] # P = Delta^2/k^3
# \Delta^2(k) = V/(2\pi)^3 4\pi k^3 P(k) is meestal de definitie (van Marius)

pl.plot(logk, powerDolag-logpower, '-')
pl.show()

pl.plot(logk, logpower, 'r-', logk, powerDolag, 'b-')
pl.show()

# Conclusie: het is dus duidelijk geen EH99 power spectrum (N.B.: op arxiv is
# het EH97, maar de ApJ versie is pas in 99 gepubliceerd!). De parameters
# lijken wel prima te zijn though (als je het aanpast wordt het pas echt heel
# erg anders).
# Overigens lijkt dit EH99 spectrum op zich wel redelijk op dat van Dolag, het
# is het gewoon nog net niet. Het lijkt wel beter dan Klypin & Holtzman 97.

# Nieuwe aanwijzingen om op te volgen:
# In een van de mailtjes van Klaus met topic Initial Conditions staat een
# bestand met parameters; een paar daarvan heb ik nog niet gebruikt. Ze worden
# door Klaus op zich ook niet gebruikt omdat hij al een tabulated power
# spectrum gebruikt, maar misschien zijn ze wel gebruikt om dat tabulated
# spectrum te genereren... De belangrijkste:
# ShapeGamma       0.201     % only needed for Efstathiou power spectrum 
# In de N-GenIC code (zie volgende mailtje) wordt alleen iets gedaan met deze
# parameter als de WhichSpectrum parameter niet gedefinieerd is ofzo (iig is
# de efstathiou methode de 'default' case in de code). Misschien produceert
# N-GenIC wel een power spectrum hiermee als je niks opgeeft?

# Misschien kan CMBeasy opheldering geven? Het resultaat hiervan:

# met n_t = 0
cmbePower1 = np.loadtxt('/Users/users/pbos/code/egpTesting/icgen/cmbeasyDolagParams.plt')
powerCMBe1 = np.log10(cmbePower1[:,1])
logkCMBe1 = np.log10(cmbePower1[:,0])

# met n_t = -0.03 weer; exact gelijk aan de eerste (blijkbaar)
cmbePower2 = np.loadtxt('/Users/users/pbos/code/egpTesting/icgen/cmbeasyDolagParamsAlt.plt')
powerCMBe2 = np.log10(cmbePower2[:,1])
logkCMBe2 = np.log10(cmbePower2[:,0])

# met n_t = n_s - 1 (dus 0.05nogiets)
# Deze is wel een paar procent anders dan de andere twee!
cmbePower3 = np.loadtxt('/Users/users/pbos/code/egpTesting/icgen/cmbeasyDolagParamsSynchronous.plt')
powerCMBe3 = np.log10(cmbePower3[:,1])
logkCMBe3 = np.log10(cmbePower3[:,0])

pl.plot(logk, logpower, 'r-', logk, powerDolag, 'b-', logkCMBe1, powerCMBe1, 'g-')
pl.show()

# Jep!
# Dit spectrum (iig op een bepaalde range) komt op het oog exact overeen met
# dat van Klaus. Hij lijkt het alleen iets te hebben aangepast/uitgebreid op
# de grenzen (bij k -> 10^-3 en k -> 10^3).

from scipy.interpolate import interp1d, UnivariateSpline, InterpolatedUnivariateSpline

pCMBe1 = interp1d(logkCMBe1, powerCMBe1, bounds_error=False, fill_value=powerCMBe1[-1], kind=1)

pl.plot(logk[:350], powerDolag[:350] - pCMBe1(logk[:350]), '-')
pl.show()

# Het scheelt dus nog wel enkele procenten (in log! 0.01 in log is factor 1.023)
# en bovendien zijn de fluctuaties rond k = 0.1 niet helemaal hetzelfde. Het
# maakt niet uit of je de n_t op -0.03 laat staan of niet.

pCMBe2 = interp1d(logkCMBe2, powerCMBe2, bounds_error=False, fill_value=powerCMBe2[-1], kind='linear')
pCMBe3 = interp1d(logkCMBe3, powerCMBe3, bounds_error=False, fill_value=powerCMBe3[-1], kind='linear')

pl.plot(logk[:350],pCMBe1(logk[:350])-pCMBe2(logk[:350]), '-')
pl.show()

pl.plot(logk[:350],powerDolag[:350]-pCMBe2(logk[:350]), '-')
pl.plot(logk[:350],powerDolag[:350]-pCMBe3(logk[:350]), '-')
pl.show()

np.std(powerDolag[:350]-pCMBe1(logk[:350])) #0.040004846131253914
                                            # pCMBe2 is exact zelfde
np.std(powerDolag[:350]-pCMBe3(logk[:350])) #0.040265076171644397
# weinig verschil dus; allebei even goed/slecht

pCMBe1a = UnivariateSpline(logkCMBe1, powerCMBe1, k=1)

pl.plot(logk[:350],powerDolag[:350]-pCMBe1a(logk[:350]), '-')
pl.show()

# Dit geeft echt een super waardelozeinterpolatie (ook met k>2)... de hele bumps zijn weg.
# Misschien ligt het daar gewoon allemaal wel aan, ook bij de interp1d...
# Anyway, op zich geeft cmbfast een goeie benadering, moeilijk te zeggen of het
# perfect is, maar goed, what is.

# Hiervoor moeten duplicates eruit (die zitten er blijkbaar in!):
mask = np.r_[True, (np.diff(logkCMBe1) > 0)]
logkCMBe1 = logkCMBe1[mask]
powerCMBe1 = powerCMBe1[mask]
pCMBe1b = InterpolatedUnivariateSpline(logkCMBe1, powerCMBe1, k=1)

pl.plot(logk[:350],powerDolag[:350]-pCMBe1b(logk[:350]), '-')
pl.show()

# oke, dit doet het dus wel prima, zelfde als interp1d, maar dan object oriented


# OKE OPNIEUW
# 13 februari, zelfde tests als hierboven, maar nu met icgen klasses:

from egp.icgen import Cosmology, CosmoPowerSpectrum as CPS, InterpolatedPowerSpectrum as IPS

logk = np.linspace(-3,3,601)
k = 10**logk

cosmo = Cosmology('wmap3')
cosmo.trans = 8
cps = CPS(cosmo)
cps.normalize(1)

D2Dolag = np.loadtxt("/Users/users/pbos/dataserver/GO/gsims/ICs/input_spectrum.txt")
powerDolag = 10**(D2Dolag[:,1] - 3*D2Dolag[:,0])
kDolag = 10**D2Dolag[:,0]

ips = IPS(kDolag, powerDolag)
ips.normalize(1, 8/0.7, 0.776)

pl.loglog(k,cps(k), k,ips(k), '-') # dolag vs Eisenstein&Hu
pl.show()

cmbePower1 = np.loadtxt('/Users/users/pbos/code/egpTesting/icgen/cmbeasyDolagParams.plt')
powerCMBe1 = cmbePower1[:,1]
kCMBe1 = cmbePower1[:,0]

ipsCMBe = IPS(kCMBe1, powerCMBe1)
ipsCMBe.normalize(1, 8/0.7, 0.776)

pl.loglog(k,cps(k), k,ips(k), k,ipsCMBe(k), '-') # dolag vs Eisenstein&Hu vs CMBeasy
pl.show()

# Als je het zo plot lijken de Dolag en CMBeasy dingen wel weer enorm veel op
# elkaar... laten we maar gewoon zeggen dat dat dezelfde zijn.
# Overigens goed om te zien dat extrapolatie blijkbaar niet cruciaal is.