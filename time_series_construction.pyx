
# cython: profile=False
import numpy as np
cimport numpy as np
import cython


cdef extern from "math.h":
    double log(double)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def logistic_map_lyap(double[:] TS, int N,double r):
    #<todo> need to add a check if r-2*r*TS[ii] ==0
    cdef double lyap =0
    cdef int ii
    for ii in range(0,N):
        lyap+=log(abs(r-2*r*TS[ii]))
    return lyap/N



@cython.boundscheck(False)
@cython.wraparound(False)
def logistic_map(double r, int N,double x_0,lyap=False):
    cdef double[:] TS = np.zeros(N, dtype='float64')
    cdef double x
    cdef int ii
    TS[0] =x_0
    for ii in range(1,N):
        x=TS[ii-1]
        TS[ii] =r*x*(1.0-x)
    if not lyap:
        return TS
    else:
        lyap = logistic_map_lyap(TS=TS,N=N,r=r)
        return TS,lyap


def logistic_map_w_process_noise(double r, int N,double x_0,mean=0,var=10e-16,lyap=False):
    cdef double[:] noise = np.random.normal(mean,var,N)
    cdef double[:] TS = np.zeros(N, dtype='float64')
    cdef double x
    cdef double xnew
    cdef double check
    cdef int ii
    TS[0] =x_0
    for ii in range(1,N):
        x=TS[ii-1]
        check=r*x*(1.0-x)+noise[ii-1]
        if 0<=check and check <=1:
            TS[ii] =check
        else:
            TS[ii] =check-noise[ii-1]
    if not lyap:
        return TS
    else:
        lyap = logistic_map_lyap(TS=TS,N=N,r=r)
        return TS,lyap


    return final

@cython.boundscheck(False)
@cython.wraparound(False)
def transient_logistic_map(double r_step,double x_0,int per_R=500,double r_start=0.5,double r_end=4.0):
    #cdef int per_R =500
    #cdef double start = 0.5
    #cdef double end = 4.0
    cdef int N = int((r_end-r_start)/r_step)
    cdef double[:] r_range =np.linspace(start=r_start,stop=r_end,num=N)
    cdef double[:] TS = np.zeros(N*per_R, dtype='float64')
    cdef double x
    cdef int ii,jj
    TS[0] =x_0

    for ii in range(1,N*per_R,per_R):
        for jj in range(0,per_R):
            x=TS[(ii+jj)-1]
            TS[ii+jj] =r_range[ii/per_R]*x*(1.0-x)
    return TS



def sinewnoise(f=5, fs=8000,mean=0,var=10e-14, N=10000):
    t = np.arange(N)
    sine = np.sin(2*np.pi*t*f/fs)
    noise = np.random.normal(mean,var,N)
    final = sine+noise
    return final


def ARMA(phi,theta,sigma,n,transient=0,verbose=0):
     """
     Random generation of Gaussian ARMA(p,q) time series.
     X_t = \sum_{i=1}^p \phi_i*X_{t-i} + \sum_{i=1}^q \theta_i*eps_{t-i}+eps_t
     INPUTS

     phi:      An array of length p with the AR coefficients (the autoregressive part of
           the ARMA model).

     theta:    An array of length q with the MA coefficients (the Moving Average part of
           the ARMA model).

     sigma:    Standard deviation of the Gaussian noise.

     n:        Length of the returned time-series.

     transient:   Number of datapoints that are going to be discarded (the higher
           the better) to avoid dependence of the ARMA time-series on the
           initial values.
     """
     p = len(phi)
     q = len(theta)
     l=max(p,q)

     if(transient==0):
       transient=10*l
     #Generate the gaussian noise necessary
     eps=np.random.normal(0,sigma,n+transient)
     ARMA=np.array([])
     if p>1:
        if abs(phi[0])>=1:
            print("This ARMA process is not stationary")

     for i in range(n+transient):
         if(i<l):
             ARMA=np.append(ARMA,eps[i])
         else:
           s=0.0
           for j in range(p):
               s=s+phi[j]*ARMA[i-j-1]
           for j in range(q):
               s=s+theta[j]*eps[i-j-1]
           ARMA=np.append(ARMA,s+eps[i])
     if(verbose!=0):
       print 'Measured standard deviation: '+str(np.sqrt(np.var(eps[transient:])))
     return ARMA[transient:]