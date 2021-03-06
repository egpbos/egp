#!/usr/bin/env python
# -*- coding: utf-8 -*-

from iconstrain_test_analyser import *
from glob import glob

boxlen = 100. # Mpc h^-1
gridsize = 256
run_dir_base = "/Users/users/pbos/dataserver/sims"
pos0 = np.array([20., 40., 70.])
radius = 4. # Mpc h^-1 (N.B.: Gadget positions are in kpc h^-1!)

log_file_names = glob(run_dir_base+'/'+'run10[089]*.log')
run_names = [x[len(run_dir_base)+1:-4] for x in log_file_names]

analyses = {'run100_2522572538': (3.3849016418140998,
                       3.5215450239977386,
                       45.987098543198776,
                       1635.0378000000001),
 'run100_2522572539': (2.9848764946389643,
                       2.6505789070406935,
                       62.122197968751479,
                       14.397691999999999),
 'run100_2522572540': (4.6077648196931271,
                       4.7931895016547017,
                       58.033835177339647,
                       2101.1631000000002),
 'run100_2522572541': (5.1407281355140242,
                       4.8284014524910761,
                       54.561573767911518,
                       134.52842999999999),
 'run100_2522572542': (6.5831478524550171,
                       6.3976305931800805,
                       57.487791672721698,
                       18.896971000000001),
 'run100_2522572543': (3.5739701623706224,
                       3.5928413292079742,
                       65.975440802335683,
                       1178.3611000000001),
 'run100_2522572544': (4.1645847708277008,
                       4.2070872832470831,
                       36.441858256723478,
                       3293.0219999999999),
 'run100_2522572545': (5.0224637726422285,
                       5.0100586338752349,
                       35.144181956486413,
                       23.846176),
 'run100_2522572546': (7.3766342581566366,
                       7.0522034685202097,
                       69.954863028442318,
                       44.992786000000002),
 'run100_2522572547': (2.8611699045389645,
                       2.5297497505247124,
                       67.113922284821982,
                       17.097259999999999),
 'run108_2522572538': (1.0366133315284938,
                       1.1626623409920767,
                       81.043005558777253,
                       1152.2653),
 'run108_2522572539': (0.56933999904875676,
                       0.65729932155119553,
                       76.023934861824927,
                       19.796824999999998),
 'run108_2522572540': (0.78003102185389916,
                       1.565713534650242,
                       54.778997119248402,
                       31.944878),
 'run108_2522572541': (0.48531856473939528,
                       0.91430268099989664,
                       32.257773711831852,
                       27.445601),
 'run108_2522572542': (1.0543698276298157,
                       1.7878518531831777,
                       90.328549341677544,
                       67.93911),
 'run108_2522572543': (3.2587609960582169,
                       3.4716181034604729,
                       58.760491915540072,
                       2196.9978),
 'run108_2522572545': (0.38388612416326151,
                       0.67482039664530025,
                       26.962294635866026,
                       139.02771000000001),
 'run108_2522572547': (0.96338646804983219,
                       2.0442666278519535,
                       51.506894733405595,
                       14.847619999999999)}


for run_name in run_names:
    print run_name
    if run_name not in analyses:
        try:
            analyses[run_name] = analyse_run(run_dir_base, run_name, pos0, boxlen, gridsize, radius)
        except IOError:
            continue
        
