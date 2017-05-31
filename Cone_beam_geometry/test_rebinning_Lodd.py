'''
Testing the core of the function that does mapping data from detector 
coordinates to kline coordinates. 
'''
import numpy as np
import matplotlib.pyplot as plt

# number of Nrows even and number of k-lines odd
w = np.arange(-6.0, 9.0, 2.0)-1
Pw = w**2

plt.figure()
plt.plot(w,Pw, 'r')

wk = np.arange(-7, 6.5, 0.15)
Pws = np.ones((len(wk),))

plt.hold(True)
#plt.plot(wk, Pws, '*b')


delta_w = -w[0]+w[1]
Nr = len(w)

Pwk = np.zeros((len(wk),))
for wki in range(len(wk)):
    w1i = np.floor((wk[wki]-w[0])/(delta_w))
    w2i = w1i+1
    t = (wk[wki]-w[w1i])/delta_w
    if w2i <= Nr-1:
        Pwk[wki] = t*Pw[w2i] + (1-t)*Pw[w1i]
    else:
        Pwk[wki] = (1-t)*Pw[w1i]

plt.plot(wk, Pwk, '.g')

Pwi = np.zeros((len(w),))
mask = np.zeros((len(w),))
for wki in range(len(wk)-1):
    w1i = np.floor((wk[wki]-w[0])/(delta_w))
    w2i = w1i+1
    if w2i <= Nr-1:  
        if mask[w1i] == 0 and mask[w2i] == 0:
            t = (w[w1i]-wk[wki])/(wk[wki+1]-wk[wki])          
            Pwi[w1i] = t*Pwk[wki+1] + (1-t)*Pwk[wki]
            t = (w[w2i]-wk[wki])/(wk[wki+1]-wk[wki])
            Pwi[w2i] = t*Pwk[wki+1] + (1-t)*Pwk[wki]
            mask[w1i] = 1
            mask[w2i] = 1
    else:
        if mask[w1i] == 0:
            t = (w[w1i]-wk[wki])/(wk[wki+1]-wk[wki])
            Pwi[w1i] = t*Pwk[wki+1] + (1-t)*Pwk[wki]
            mask[w1i] = 1

plt.plot(w, Pwi, '.b')
plt.show()  
