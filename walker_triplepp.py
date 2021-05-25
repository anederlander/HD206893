#Ava Nederlander Code
#for triplepp without gap ONLY

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
"""
with open('chain_1000steps_oct12_triplepp.csv', 'r') as fin, open('chain_1000steps_oct12_triplepp2.csv', 'w', newline='') as fout:

    # define reader and writer objects
    reader = csv.reader(fin, skipinitialspace=True)
    writer = csv.writer(fout, delimiter=',')

    # write headers 
    writer.writerow(next(reader))
    k=0   
    # iterate and write rows based on condition
    for i in reader:
        #if ((float(i[4])*1e6) < (25)):
            #print(i[4])
        if (float(i[13]) < 75):
            #print("HERE")
            #if ((float(i[4])*1e6) > (25)):
            writer.writerow(i)
            k+=1 
print("included:", k/16)
#df = pd.read_csv("chain_400steps_april20_nogap1.csv")
"""
df = pd.read_csv("chain_1000steps_oct12_triplepp.csv")
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
max_position_angle = df.cos_position_angle[df.lnprob.idxmax()]
max_inclination = df.cos_inclination[df.lnprob.idxmax()]
max_xoffs_stellar = df.xoffs_stellar[df.lnprob.idxmax()] #gap
max_yoffs_stellar = df.yoffs_stellar[df.lnprob.idxmax()] #gap
max_pp1 = df.pp1[df.lnprob.idxmax()]
max_pp2 = df.pp2[df.lnprob.idxmax()]
max_pp3 = df.pp3[df.lnprob.idxmax()]
max_rt1 = df.rt1[df.lnprob.idxmax()]
max_rt2 = df.rt2[df.lnprob.idxmax()]

print("Best-Fit Parameters:")
print("lnprob =", max_lnprob)
print("r_in =", max_r_in)
print("delta_r =", max_delta_r)
print("m_disk =", max_m_disk)
print("f_star =", max_f_star)
print("position_angle =", max_position_angle)
print("inclination =", max_inclination)
#print("xoffs =", max_xoffs) #nogap
#print("yoffs =", max_yoffs) #no gap
print("xoffs_stellar =", max_xoffs_stellar) #gap
print("yoffs_stellar =", max_yoffs_stellar)
#print("r_in_gap =", max_r_in_gap)
#print("delta_r_gap =", max_delta_r_gap)
print("pp1 =", max_pp1)
print("pp2 =", max_pp2)
print("pp3 =", max_pp3)
print("rt1 =", max_rt1)
print("rt2 =", max_rt2)
#print("rt =", max_rt)





############

lnprob = df.lnprob
r_in = df.r_in
delta_r = df.delta_r
m_disk = df.m_disk
f_star = df.f_star
position_angle = df.cos_position_angle
inclination = df.cos_inclination
xoffs_stellar = df.xoffs_stellar
yoffs_stellar = df.yoffs_stellar
pp1 = df.pp1
rt1 = df.rt1
pp2 = df.pp2
pp3 = df.pp3
rt2 = df.rt2
#lnprob = df.lnprob

nsamples, ndim = df.shape
#nwalkers = ndim
nsteps = 2000
#nsteps = nsamples
nwalkers = 30
#nsteps = nsamples
r_in_ar = np.empty((nsteps, nwalkers))
delta_r_ar = np.empty((nsteps, nwalkers))
m_disk_ar = np.empty((nsteps, nwalkers))
f_star_ar = np.empty((nsteps, nwalkers))
position_angle_ar = np.empty((nsteps, nwalkers))
inclination_ar = np.empty((nsteps, nwalkers))
xoffs_stellar_ar = np.empty((nsteps, nwalkers))
yoffs_stellar_ar = np.empty((nsteps, nwalkers))
pp1_ar = np.empty((nsteps, nwalkers))
pp2_ar = np.empty((nsteps, nwalkers))
pp3_ar = np.empty((nsteps, nwalkers))
rt1_ar = np.empty((nsteps, nwalkers))
rt2_ar = np.empty((nsteps, nwalkers))
lnprob_ar = np.empty((nsteps, nwalkers))

#print(r_in_ar.shape)
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
    pp1_ar[:, i] = pp1[i::nwalkers]
    pp2_ar[:, i] = pp2[i::nwalkers]
    pp3_ar[:, i] = pp3[i::nwalkers]
    rt1_ar[:, i] = rt1[i::nwalkers]
    rt2_ar[:, i] = rt2[i::nwalkers]
    lnprob_ar[:, i] = lnprob[i::nwalkers]
    #loc = np.where(r_in_ar[:,i]>0)
    #ax0.plot(steps[loc],r_in_ar[loc, i])
    #ax0.plot(steps[loc],delta_r_ar[loc, i])
    #ax0.plot(steps[loc],delta_r_ar[loc, i])
    #ax0.plot(steps[loc],r_in_ar[loc, i])
    #ax0.plot(steps[loc],r_in_ar[loc, i])
    #ax0.plot(steps[loc],r_in_ar[loc, i])
    #ax0.plot(steps[loc],r_in_ar[loc, i])
    #ax0.plot(steps[loc],r_in_ar[loc, i])




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

plt.subplot(7,2,1)
for i in range(0, nwalkers):
    if (lnprob_ar[-1,i] > -10733012.0):
        print(i)
        loc = np.where(np.ravel(r_in_ar[:,i]>0))
        plt.plot(steps[loc[0]], r_in_ar[loc[0],i])
plt.ylabel('$R_i$$_n$ (au)', fontsize=12,rotation=90, labelpad=10)
#plt.title("Walker Evolution Plot")
#plt.set_ylim([0, 20])
plt.yticks(fontsize=7)
plt.xticks([])
#plt.ylim(0, 75)

plt.subplot(7,2,2)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], delta_r_ar[loc[0],i])
plt.ylabel('$\Delta$ R (au)', fontsize=12,rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,3)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], m_disk_ar[loc[0],i])
plt.ylabel('Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])
#plt.ylim(-6.8, -7.4)

plt.subplot(7,2,4)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], f_star_ar[loc[0],i])
plt.ylabel('F$_s$$_t$$_a$$_r$ ($\mu$ Jy)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,5)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], position_angle_ar[loc[0],i])
plt.ylabel('PA ($\degree$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([]
        )
plt.subplot(7,2,6)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], inclination_ar[loc[0],i])
plt.ylabel('i ($\degree$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,7)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], xoffs_stellar_ar[loc[0],i])
plt.ylabel('$\Delta$ x (")', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,8)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], yoffs_stellar_ar[loc[0],i])
plt.ylabel('$\Delta$ y (")', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])


plt.subplot(7,2,9)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], pp1_ar[loc[0],i])
plt.ylabel('$pp1$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,10)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], pp2_ar[loc[0],i])
plt.ylabel('$pp2$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,11)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):        
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], pp3_ar[loc[0],i])
plt.ylabel('$pp3$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,12)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], rt1_ar[loc[0],i])
plt.ylabel('$rt_1$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,13)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], rt2_ar[loc[0],i])
plt.ylabel('$rt_2$', fontsize=12, rotation=90, labelpad=10)
plt.xlabel("Step", fontsize=15)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)


plt.subplot(7,2,14)
for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        #loc = np.where(r_in_ar[:,i]>0)
        loc = np.where(r_in_ar[:,i]>0)
        plt.plot(steps[loc[0]], (lnprob_ar[loc[0],i])/1e7)
        print(np.min((lnprob_ar[-1,i])))
    #print(np.max(lnprob_ar[loc[0],i])/1e7)
    #print(np.median((lnprob_ar[loc[0],i])/1e7))
    #plt.plot(x, np.array(df['lnprob'][i::nwalkers])/1e7)
plt.ylabel('lnprob', fontsize=12, rotation=90, labelpad=10)
plt.xlabel('Step', fontsize=15)
plt.ylim(-1.073303, -1.0732990)
#plt.tight_layout(pad=3.0)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)

#plt.set_size_inches(18.5, 10.5, forward=True)
#print(np.median(df['lnprob']))
plt.savefig("walker_nov30_triplepp.pdf", bbox_inches="tight")




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
max_position_angle = df.cos_position_angle[df.lnprob.idxmax()]
max_inclination = df.cos_inclination[df.lnprob.idxmax()]
max_xoffs_stellar = df.xoffs_stellar[df.lnprob.idxmax()]
max_yoffs_stellar = df.yoffs_stellar[df.lnprob.idxmax()]
max_pp1 = df.pp1[df.lnprob.idxmax()]
max_rt1 = df.rt1[df.lnprob.idxmax()]
max_pp2 = df.pp2[df.lnprob.idxmax()]
max_pp3 = df.pp3[df.lnprob.idxmax()]
max_rt2 = df.rt2[df.lnprob.idxmax()]



r_in_arr = np.empty([])
delta_r_arr = np.empty([])
m_disk_arr = np.empty([])
f_star_arr = np.empty([])
position_angle_arr = np.empty([])
inclination_arr = np.empty([])
xoffs_stellar_arr = np.empty([])
yoffs_stellar_arr = np.empty([])
pp1_arr = np.empty([])
pp2_arr = np.empty([])
pp3_arr = np.empty([])
rt1_arr = np.empty([])
rt2_arr = np.empty([])
#r_in_arr = np.empty([])



for i in range(0, nwalkers):
    #if (rt2_ar[-1,i] > 75):
    if (lnprob_ar[-1,i] > -10733012.0):
        #loc = np.where(r_in_ar[:,i]>0)
        loc = np.where(r_in_ar[:,i]>0)
        #plt.plot(steps[loc[0]], (lnprob_ar[loc[0],i])/1e7)
        r_in_arr = np.append(r_in_arr, r_in_ar[:, i])
        delta_r_arr = np.append(delta_r_arr, delta_r_ar[:, i])
        m_disk_arr = np.append(m_disk_arr, m_disk_ar[:, i])
        f_star_arr = np.append(f_star_arr, f_star_ar[:, i])
        position_angle_arr = np.append(position_angle_arr, position_angle_ar[:, i])
        inclination_arr = np.append(inclination_arr, inclination_ar[:, i])
        xoffs_stellar_arr = np.append(xoffs_stellar_arr, xoffs_stellar_ar[:, i])
        yoffs_stellar_arr = np.append(yoffs_stellar_arr, yoffs_stellar_ar[:, i])
        pp1_arr = np.append(pp1_arr, pp1_ar[:, i])
        pp2_arr = np.append(pp2_arr, pp2_ar[:, i])
        pp3_arr = np.append(pp3_arr, pp3_ar[:, i])
        rt1_arr = np.append(rt1_arr, rt1_ar[:, i])
        rt2_arr = np.append(rt2_arr, rt2_ar[:, i])



nsamples = np.shape(r_in_arr)[0]

#sampler = emcee.EnsembleSampler(16, ndim, max_lnprob)
#samples = sampler.chain[:,:,:].reshape((-1, ndim))
samples = np.zeros([nsamples,ndim])

#print(nsamples,np.shape(lnprob_ar))

samples[:,0] = r_in_arr
samples[:,1] = delta_r_arr
samples[:,2] = m_disk_arr
samples[:,3] = f_star_arr
samples[:,4] = position_angle_arr
samples[:,5] = inclination_arr
samples[:,6] = xoffs_stellar_arr
samples[:,7] = yoffs_stellar_arr
samples[:,8] = pp1_arr
samples[:,9] = pp2_arr
samples[:,10] = pp3_arr
samples[:,11] = rt1_arr
samples[:,12] = rt2_arr

lnprob_s = df['lnprob']




#fig = corner.corner(samples, labels=["$r_in$", "$delta_r$"],truths=[max_r_in, max_delta_r])
#fig = corner.corner(samples, labels=["$r_{in}$", "$\Delta_r$", "$m_{disk}$", "$f_{star}$", "$position angle$", "$inclination$", "$\Delta_x$", "$\Delta_y$", "$pp1$", "$pp2$", "$pp3$", "$rt1$","$rt2$"],truths=[max_r_in, max_delta_r, max_m_disk, max_f_star, max_position_angle, max_inclination, max_xoffs_stellar, max_yoffs_stellar, max_pp1, max_pp2, max_pp3, max_rt1, max_rt2])
#loc = np.where(samples[:,0]>0)
loc = np.where(lnprob_s > -10733012.0)
loc2 = np.where(loc[0] > 300)
ind = loc2[0]
locflat = loc[0]
fig = corner.corner(samples,labels=["$R_{in}$ (au)", "$\Delta$ R (au)", "Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)", "F$_s$$_t$$_a$$_r$ ($\mu$ Jy)", "$PA (^\circ)$", "$i (^\circ)$", '$\Delta$ x (")', '$\Delta_y$ (")', "$pp1$", "$pp2$", "$pp3$", "$R_t1$", "$R_t2$"], max_n_ticks=2, label_kwargs={"fontsize": 22},truths=[max_r_in, max_delta_r, max_m_disk, max_f_star, max_position_angle, max_inclination, max_xoffs_stellar, max_yoffs_stellar, max_pp1, max_pp2, max_pp3, max_rt1, max_rt2])
fig.show()
fig.savefig("cornerplot_nov30_triplepp.pdf")







#end of file:)
