import egp.io as io

data = io.GadgetData("random_verschillende_resoluties_N-GenIC/ICs/ic64")
data.loadPos()
data.loadVel()

io.write_gadget_ic_dm("rewrite_test_ic64.dat", data.pos, data.vel, data.header[0]['Massarr'][1], data.header[0]['Redshift'], data.header[0]['BoxSize'], 0.3, 0.7, 0.7)

data_dup = io.GadgetData("rewrite_test_ic64.dat")
data_dup.loadPos()
data_dup.loadVel()

np.prod(data.pos == data_dup.pos)
np.prod(data.vel == data_dup.vel)
data.header
data_dup.header
# klopt!