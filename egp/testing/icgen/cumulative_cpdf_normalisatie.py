#!/usr/bin/env python
# 3 mei 2012

import numpy as np
from egp.constrainedField import *

x = np.linspace(0,10,5001)

cpdf = cpdf_curvature(x,height.value, gamma)
cumcpdf = np.cumsum(cpdf)/cpdf.sum()
cumcpdfBBKS = cumulative_cpdf_curvature(x,height.value, gamma)

pl.plot(x,cpdf,'-',x,cumcpdf,'-',x,cumcpdfBBKS,'-')
pl.show()
# plaatje opslaan en klaar