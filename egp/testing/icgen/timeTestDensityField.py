import egp.icgen
from time import time

N = 128

tClass = []
tNaked = []

R = np.random.random((N,N,N))

for i in range(10):
    den = egp.icgen.Field()
    t=time();den.t = R.copy();t2 = time()#;print t2-t
    t=time();f = den.f;t2 = time();tClass.append(t2-t)#;print t2-t
    
    t=time();a = R.copy();t2 = time()#;print t2-t
    t=time();b = np.fft.rfftn(a)/np.size(a);t2 = time();tNaked.append(t2-t)#;print t2-t

for i in range(10):
    t=time();a = R.copy();t2 = time()#;print t2-t
    t=time();b = np.fft.rfftn(a)/np.size(a);t2 = time();tNaked.append(t2-t)#;print t2-t
    
    den = egp.icgen.Field()
    t=time();den.t = R.copy();t2 = time()#;print t2-t
    t=time();f = den.f;t2 = time();tClass.append(t2-t)#;print t2-t


np.mean(tClass)/np.mean(tNaked)
np.std(tClass)/np.std(tNaked)
np.min(tClass)/np.min(tNaked)
np.max(tClass)/np.max(tNaked)

# memory usage check (hou top in de gaten per loop) = ZELFDE
for i in range(50):
    t=time();a = R.copy();t2 = time()#;print t2-t
    t=time();b = np.fft.rfftn(a)/np.size(a);t2 = time();tNaked.append(t2-t)#;print t2-t

for i in range(50):
    den = egp.icgen.Field()
    t=time();den.t = R.copy();t2 = time()#;print t2-t
    t=time();f = den.f;t2 = time();tClass.append(t2-t)#;print t2-t
