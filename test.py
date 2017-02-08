# Imports
import numpy
import numpy as np
import pandas as pd

import tigramite_preprocessing as pp
import tigramite_estimation_weighted_PE as tig
import ordinal_TSA
#from tigramite_src import tigramite_plotting
from scipy.misc import factorial


data = pd.io.parsers.read_csv('/Users/admin/Documents/Projects/Climate-Ice Core/data/NGRIP_depth_d18O.csv')
orig_fulldata = numpy.asarray(data['d18O']).squeeze()

orig_fulldata_mask = numpy.ones(orig_fulldata.shape, dtype='int32')
T = len(orig_fulldata)

fulldata_mask = numpy.ones(orig_fulldata.shape, dtype='int32')
dim=5
step=2
symb_fulldata, symb_fulldata_mask, Tpatt, full_weights = pp.ordinal_patt_array(orig_fulldata, fulldata_mask,
                                        weights=True,
                                       dim=dim, step=step, verbosity=0)
# print orig_fulldata.squeeze()
# print symb_fulldata.squeeze()
max_window_size = 1000
window_endpoints = numpy.arange(max_window_size, T-(dim-1)*step+1, 1)

window_length = 1000


for w in [0,1]:
    print "\nweighted ", w
    print 'windows'
    print "josh  ", np.asarray(ordinal_TSA.windowed_permutation_entropy(numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),
                                                              10,dim,max_window_size=max_window_size,step=step,w=w))


lower, upper = ordinal_TSA.sig_testing(numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),10000,window_length*5,window_length,dim,step=1,w=1)
for x in np.asarray(lower):
    if x ==0:
        print x

