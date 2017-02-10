# Imports
import numpy
import numpy as np
import pandas as pd

import tigramite_preprocessing as pp
# import tigramite_estimation_weighted_PE as tig
import ordinal_TSA
#from tigramite_src import tigramite_plotting
from scipy.misc import factorial


# data = pd.io.parsers.read_csv('/Users/admin/Documents/Projects/Climate-Ice Core/data/NGRIP_depth_d18O.csv')
# orig_fulldata = numpy.asarray(data['d18O']).squeeze()
orig_fulldata = numpy.random.randn(55)

orig_fulldata_mask = numpy.ones(orig_fulldata.shape, dtype='int32')
T = len(orig_fulldata)

fulldata_mask = numpy.ones(orig_fulldata.shape, dtype='int32')
dim=3
step=1
symb_fulldata, symb_fulldata_mask, Tpatt, full_weights = pp.ordinal_patt_array(orig_fulldata, fulldata_mask,
                                        weights=True,
                                       dim=dim, step=step, verbosity=0)
# print orig_fulldata.squeeze()
# print symb_fulldata.squeeze()
window_length = 11
window_steps = 6
max_window_size = window_length

window_endpoints = numpy.arange(max_window_size, T-(dim-1)*step+1, 1)

for w in [0,1]:
    print "\nweighted ", w
    print 'windows'
    one = np.asarray(ordinal_TSA.windowed_permutation_entropy(
                    numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),
                    window_length,dim,max_window_size=max_window_size,step=step,w=w))
    print "one  ", one[::window_steps], len(one)
    two = np.asarray(ordinal_TSA.windowed_bootstrap_permutation_entropy(
                    numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),
                    window_size=window_length,dim=dim,max_window_size=max_window_size,
                    step=step,w=w,window_steps=window_steps))
    print "two  ", two, len(two)
    # assert numpy.all(one==two)



# lower, upper = ordinal_TSA.sig_testing(numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),10000,window_length*5,window_length,dim,step=1,w=1)
# for x in np.asarray(lower):
#     if x ==0:
#         print x

