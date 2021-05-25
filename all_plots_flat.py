#Ava Nederlander Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import sys
import corner
import csv
import pylab
import seaborn as sns
nwalkers=30
ndim=13

#df = pd.read_csv("combined_triplepp.csv")

#df = pd.read_csv("combined_triplepp.csv", skiprows=np.append(range(1, 24000, 30), range(24002, 60000, 30)))
#df = pd.read_csv("combined_triplepp.csv", skiprows=np.append(range(13, 24000, 30), range(24015, 60000, 30)))
#df = pd.read_csv("chain_400steps_april20_nogap.csv")
#df = pd.read_csv("chain_2000steps_flat_dec29.csv")
#df = pd.read_csv("chain_251steps_flat_jan4.csv")
df = pd.read_csv("flat_combined_2000.csv")
nsamples, ndim = df.shape
print(nsamples)
##################
#BEST FIT
##################

max_lnprob = df['lnprob'].max()
max_r_in = df.r_in[df.lnprob.idxmax()]
max_delta_r = df.delta_r[df.lnprob.idxmax()]
max_m_disk = df.m_disk[df.lnprob.idxmax()]
max_f_star = df.f_star[df.lnprob.idxmax()]
max_position_angle = df.position_angle[df.lnprob.idxmax()]
max_inclination = df.inclination[df.lnprob.idxmax()]
max_xoffs = df.xoffs[df.lnprob.idxmax()] #nogap
max_yoffs = df.yoffs[df.lnprob.idxmax()] #nogap

print("Best-Fit Parameters:")
print("lnprob =", max_lnprob)
print("r_in =", max_r_in)
print("delta_r =", max_delta_r)
print("m_disk =", max_m_disk)
print("f_star =", max_f_star)
print("position_angle =", max_position_angle)
print("inclination =", max_inclination)
print("xoffs =", max_xoffs) #nogap
print("yoffs =", max_yoffs) #no gap



############

lnprob = df.lnprob
r_in = df.r_in
delta_r = df.delta_r
m_disk = df.m_disk
f_star = df.f_star
position_angle = df.position_angle
inclination = df.inclination
xoffs_stellar = df.xoffs
yoffs_stellar = df.yoffs
#lnprob = df.lnprob

nsamples, ndim = df.shape
#nwalkers = 29 #30
nsteps = 2000 #1749 #251
nwalkers = 16
#nsteps = nsamples
r_in_ar = np.empty((nsteps, nwalkers))
delta_r_ar = np.empty((nsteps, nwalkers))
m_disk_ar = np.empty((nsteps, nwalkers))
f_star_ar = np.empty((nsteps, nwalkers))
position_angle_ar = np.empty((nsteps, nwalkers))
inclination_ar = np.empty((nsteps, nwalkers))
xoffs_stellar_ar = np.empty((nsteps, nwalkers))
yoffs_stellar_ar = np.empty((nsteps, nwalkers))
lnprob_ar = np.empty((nsteps, nwalkers))
print(r_in_ar.shape)
steps = np.arange(0, nsteps)
#another for loop with array for shapes n steps x n walkers
for i in range(0, nwalkers):
    r_in_ar[:, i] = r_in[i::nwalkers]
    delta_r_ar[:, i] = delta_r[i::nwalkers]
    m_disk_ar[:, i] = m_disk[i::nwalkers]
    f_star_ar[:, i] = f_star[i::nwalkers]
    position_angle_ar[:, i] = position_angle[i::nwalkers]
    inclination_ar[:, i] = inclination[i::nwalkers]
    xoffs_stellar_ar[:, i] = xoffs_stellar[i::nwalkers]
    yoffs_stellar_ar[:, i] = yoffs_stellar[i::nwalkers]
    lnprob_ar[:, i] = lnprob[i::nwalkers]



##############






w1m = df['r_in'][0::nwalkers]
x = np.arange(0,len(w1m))


pylab.rcParams['xtick.major.pad']='8'
#pylab.rcParams['ytick.major.pad']='8'
x = np.arange(0,len(w1m))


#################
#WALKER EVOLUTION
#################

plt.figure(figsize=(17.0,17.0))

plt.subplot(5,2,1)
for i in range(0, nwalkers):
    loc = np.where(np.ravel(r_in_ar[:,i]<200))
    plt.plot(steps[loc[0]], r_in_ar[loc[0],i])
plt.ylabel('$R_i$$_n$ (au)', fontsize=12,rotation=90, labelpad=10)
#plt.title("Walker Evolution Plot")
#plt.set_ylim([0, 20])
plt.yticks(fontsize=7)
plt.xticks([])
#plt.ylim(0, 75)

plt.subplot(5,2,2)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]<200)
    plt.plot(steps[loc[0]], delta_r_ar[loc[0],i])
plt.ylabel('$\Delta$ R (au)', fontsize=12,rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(5,2,3)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]<200)
    plt.plot(steps[loc[0]], m_disk_ar[loc[0],i])
plt.ylabel('Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])
plt.ylim(-6.8, -7.4)

plt.subplot(5,2,4)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]<200)
    plt.plot(steps[loc[0]], f_star_ar[loc[0],i])
plt.ylabel('F$_s$$_t$$_a$$_r$ ($\mu$ Jy)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(5,2,5)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]<200)
    plt.plot(steps[loc[0]], position_angle_ar[loc[0],i])
plt.ylabel('PA ($\degree$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([]
        )
plt.subplot(5,2,6)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]<200)
    plt.plot(steps[loc[0]], inclination_ar[loc[0],i])
plt.ylabel('i ($\degree$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(5,2,7)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]<200)
    plt.plot(steps[loc[0]], xoffs_stellar_ar[loc[0],i])
plt.ylabel('$\Delta$ x (")', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(5,2,8)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]<200)
    plt.plot(steps[loc[0]], yoffs_stellar_ar[loc[0],i])
plt.ylabel('$\Delta$ y (")', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
#plt.xticks([])
plt.xticks(fontsize=7)
"""

plt.subplot(7,2,9)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]>0)
    plt.plot(steps[loc[0]], pp1_ar[loc[0],i])
plt.ylabel('$pp1$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,10)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]>0)
    plt.plot(steps[loc[0]], pp2_ar[loc[0],i])
plt.ylabel('$pp2$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,11)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]>0)
    plt.plot(steps[loc[0]], pp3_ar[loc[0],i])
plt.ylabel('$pp3$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,12)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]>0)
    plt.plot(steps[loc[0]], rt1_ar[loc[0],i])
plt.ylabel('$rt_1$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,13)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]>0)
    plt.plot(steps[loc[0]], rt2_ar[loc[0],i])
plt.ylabel('$rt_2$', fontsize=12, rotation=90, labelpad=10)
plt.xlabel("Step", fontsize=15)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

"""
plt.subplot(5,2,9)
for i in range(0, nwalkers):
    loc = np.where(r_in_ar[:,i]<200)
    plt.plot(steps[loc[0]], (lnprob_ar[loc[0],i])/1e7)
    #print(np.min((lnprob_ar[loc[0],i])/1e7))
    #print(np.max(lnprob_ar[loc[0],i])/1e7)
    #print(np.median((lnprob_ar[loc[0],i])/1e7))
    #plt.plot(x, np.array(df['lnprob'][i::nwalkers])/1e7)
plt.ylabel('lnprob', fontsize=12, rotation=90, labelpad=10)
plt.xlabel('Step', fontsize=15)
plt.ylim(-1.073310, -1.0732990)
#plt.tight_layout(pad=3.0)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)

#plt.set_size_inches(18.5, 10.5, forward=True)
print(np.median(df['lnprob']))
plt.savefig("walker_flatdisk.pdf", bbox_inches="tight")




############
#CORNER PLOT
############

#df = old_df[~old_df.lnprob.isin(x)]
#ndim = len(df.columns) - 1
#nsamples =
#nsamples, ndim = df.shape
#nsamples = 58000 #60000
#ndim = 15
#Here: see if you can get ndim and nsamples from .csv file directly
ndim = ndim - 2
print(ndim)
print(nsamples)
max_lnprob = df['lnprob'].max()
max_r_in = df.r_in[df.lnprob.idxmax()]
max_delta_r = df.delta_r[df.lnprob.idxmax()]
max_m_disk = df.m_disk[df.lnprob.idxmax()]
max_f_star = df.f_star[df.lnprob.idxmax()]*1e6
max_position_angle = df.position_angle[df.lnprob.idxmax()]
max_inclination = df.inclination[df.lnprob.idxmax()]
max_xoffs_stellar = df.xoffs[df.lnprob.idxmax()]
max_yoffs_stellar = df.yoffs[df.lnprob.idxmax()]

#sampler = emcee.EnsembleSampler(16, ndim, max_lnprob)
#samples = sampler.chain[:,:,:].reshape((-1, ndim))
samples = np.zeros([nsamples,ndim])

samples[:,0] = df['r_in']
samples[:,1] = df['delta_r']
samples[:,2] = df['m_disk']
samples[:,3] = df['f_star']
samples[:,4] = df['position_angle']
samples[:,5] = df['inclination']
samples[:,6] = df['xoffs']
samples[:,7] = df['yoffs']



#fig = corner.corner(samples, labels=["$r_in$", "$delta_r$"],truths=[max_r_in, max_delta_r])
#fig = corner.corner(samples, labels=["$r_{in}$", "$\Delta_r$", "$m_{disk}$", "$f_{star}$", "$position angle$", "$inclination$", "$\Delta_x$", "$\Delta_y$", "$pp1$", "$pp2$", "$pp3$", "$rt1$","$rt2$"],truths=[max_r_in, max_delta_r, max_m_disk, max_f_star, max_position_angle, max_inclination, max_xoffs_stellar, max_yoffs_stellar, max_pp1, max_pp2, max_pp3, max_rt1, max_rt2])
loc = np.where(samples[:,0]<200)
#loc2 = loc[0,np.where(loc[0]>24650)]
loc2 = np.where(loc[0]<24650)
ind = loc2[0]
locflat = loc[0]
fig = corner.corner(samples[locflat[ind],:], labels=["$R_{in}$ (au)", "$\Delta$ R (au)", "Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)", "F$_s$$_t$$_a$$_r$ ($\mu$ Jy)", "$PA (^\circ)$", "$i (^\circ)$", '$\Delta$ x (")', '$\Delta$ y (")'], max_n_ticks=2, label_kwargs={"fontsize": 22},truths=[max_r_in, max_delta_r, max_m_disk, max_f_star, max_position_angle, max_inclination, max_xoffs_stellar, max_yoffs_stellar])
fig.show()
fig.savefig("cornerplot_flatdisk.pdf")








#end of file:)
