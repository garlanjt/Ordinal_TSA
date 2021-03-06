# Imports
import numpy
import numpy as np
import pandas as pd

#import tigramite_preprocessing as pp
## import tigramite_estimation_weighted_PE as tig
import ordinal_TSA
##from tigramite_src import tigramite_plotting
from scipy.misc import factorial


# data = pd.io.parsers.read_csv('/Users/admin/Documents/Projects/Climate-Ice Core/data/NGRIP_depth_d18O.csv')
# orig_fulldata = numpy.asarray(data['d18O']).squeeze()

#orig_fulldata_mask = numpy.ones(orig_fulldata.shape, dtype='int32')
#T = len(orig_fulldata)
#
#fulldata_mask = numpy.ones(orig_fulldata.shape, dtype='int32')

#symb_fulldata, symb_fulldata_mask, Tpatt, full_weights = pp.ordinal_patt_array(orig_fulldata, fulldata_mask,
#                                        weights=True,
#                                       dim=dim, step=step, verbosity=0)
# print orig_fulldata.squeeze()
# print symb_fulldata.squeeze()
#max_window_size = 1000
#window_endpoints = numpy.arange(max_window_size, T-(dim-1)*step+1, 1)
#
#window_length = 1000
#
#window_endpoints = numpy.arange(max_window_size, T-(dim-1)*step+1, 1)

#for w in [0,1]:
#    print "\nweighted ", w
#    print 'windows'
#    one = np.asarray(ordinal_TSA.windowed_permutation_entropy(
#                    numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),
#                    window_length,dim,max_window_size=max_window_size,step=step,w=w))
#    print "one  ", one[::window_steps], len(one)
#    two = np.asarray(ordinal_TSA.windowed_bootstrap_permutation_entropy(
#                    numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),
#                    window_size=window_length,dim=dim,max_window_size=max_window_size,
#                    step=step,w=w,window_steps=window_steps))
#    print "two  ", two, len(two)
    # assert numpy.all(one==two)

orig_fulldata = numpy.random.randn(20000)
#Num Random shuffles
B = 1000
dim=3
step=1

windowsize = 2400
max_window_size = windowsize
surrogate_windowsize = int((len(orig_fulldata)-step*(dim-1)))
shift = 500

conf_level = 0.1
weight = 0


data = pd.io.parsers.read_csv('cpu_experiment/iozone_cpu_load_p.txt',names=['bst','cpu'])
#lower, upper = ordinal_TSA.sig_testing_w_overlaps(data=numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),B=B,B_w=surrogate_windowsize,windowsize=windowsize,dim=dim,shift=shift,step=step,w=weight, conf_level=conf_level)
TS = np.asarray(data['cpu']).reshape(len(data['cpu']),1)

wpe = ordinal_TSA.per


#wwpe =ordinal_TSA.windowed_permutation_entropy(data=
                    numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),
                    window_size=windowsize,dim=dim,max_window_size=max_window_size,step=step,w=weight)

lower = np.asarray(lower)
upper = np.asarray(upper)
wwpe = np.asarray(wwpe)

print(lower[max_window_size-1:])
print(upper[max_window_size-1:])
#print(wwpe)
#print(len(wwpe))

#outside = 0
#for x in range(len(lower)):
#    if lower[x]>wwpe[x] or upper[x]<wwpe[x]:
#        outside +=1

#num_indep_samples = (len(orig_fulldata)-step*(dim-1))/windowsize

#print conf_level, np.sqrt(conf_level*(1.-conf_level)/windowsize),np.sqrt(conf_level*(1.-conf_level)/num_indep_samples)
#print ("%upper",(wwpe >= upper[max_window_size-1:]).mean())
#print ("%lower",(wwpe <= lower[max_window_size-1:]).mean())


# lower, upper = ordinal_TSA.sig_testing(numpy.asarray(orig_fulldata).reshape(len(orig_fulldata), 1),10000,window_length*5,window_length,dim,step=1,w=1)
# for x in np.asarray(lower):
#     if x ==0:
#         print x

rand = np.random.randn()