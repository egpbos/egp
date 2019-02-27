def skew(a):
	m1 = a.mean()
	m2 = a.std()
	m3 = 0.0
	for i in range(len(a)):
		m3 += (a[i]-m1)**3.0
	return m3/m2**3/len(a)

def kurtosis(a):
	m1 = a.mean()
	m2 = a.std()
	m4 = 0.0
	for i in range(len(a)):
		m4 += (a[i]-m1)**4.0
	return m4/m2**4/len(a) - 3.0
