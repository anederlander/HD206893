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



df = csv.reader(open('triplepp_nogap_2000.csv', 'rt'), delimiter=",", quotechar="|")
freq, r_in, delta_r, m_disk, f_star, position_angle, inclination, xoffs, yoffs, pp1, pp2, pp3, rt1, rt2, lnprob = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

next(df)
for row in df:
    freq.append((row[0]))
    r_in.append(float(row[1]))
    delta_r.append(float(row[2]))
    m_disk.append(float(row[3]))
    f_star.append(float(row[4])*1e6)
    position_angle.append(float(row[5]))
    inclination.append(float(row[6]))
    xoffs.append(float(row[7]))
    yoffs.append(float(row[8]))
    pp1.append(float(row[9]))
    pp2.append(float(row[10]))
    pp3.append(float(row[11]))
    rt1.append(float(row[12]))
    rt2.append(float(row[13]))
    lnprob.append(float(row[14]))


#for i in f_star:

#ndim=6
#old_df = pd.read_csv("chain_20steps_new8params.csv", skiprows=range(1,2401))
df = pd.read_csv("triplepp_nogap_2000.csv")
#df = pd.read_csv("chain_20steps_new8params.csv")
#x = [-np.inf]
w1m = df['r_in'][0::nwalkers]
x = np.arange(0,len(w1m))


pylab.rcParams['xtick.major.pad']='8'
#pylab.rcParams['ytick.major.pad']='8'
x = np.arange(0,len(w1m))

plt.subplot(7,2,1)
for i in range(0, nwalkers):
    plt.plot(x, df['r_in'][i::nwalkers])
plt.ylabel('$R_i$$_n$ (au)', fontsize=7,rotation=0, labelpad=30)
#plt.title("Walker Evolution Plot")
#plt.set_ylim([0, 20])
#plt.yticks(fontsize=5, rotation=90)

plt.subplot(7,2,2)
for i in range(0, nwalkers):
    plt.plot(x, df['delta_r'][i::nwalkers])
plt.ylabel('$\Delta$ R (au)', fontsize=7,rotation=0, labelpad=20)
#plt.yticks(fontsize=5)

plt.subplot(7,2,3)
for i in range(0, nwalkers):
    plt.plot(x, df['m_disk'][i::nwalkers])
plt.ylabel('Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,4)
for i in range(0, nwalkers):
    plt.plot(x, df['f_star'][i::nwalkers]*1e6)
plt.ylabel('f$_s$$_t$$_a$$_r$ ($\mu$ Jy)', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,5)
for i in range(0, nwalkers):
    plt.plot(x, df['cos_position_angle'][i::nwalkers])
plt.ylabel('PA ($\degree$)', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,6)
for i in range(0, nwalkers):
    plt.plot(x, df['cos_inclination'][i::nwalkers])
plt.ylabel('i ($\degree$)', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,7)
for i in range(0, nwalkers):
    plt.plot(x, df['xoffs_stellar'][i::nwalkers])
plt.ylabel('$\Delta$ x (")', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,8)
for i in range(0, nwalkers):
    plt.plot(x, df['yoffs_stellar'][i::nwalkers])
plt.ylabel('$\Delta$ y (")', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,9)
for i in range(0, nwalkers):
    plt.plot(x, df['pp1'][i::nwalkers])
plt.ylabel('$pp1$', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,10)
for i in range(0, nwalkers):
    plt.plot(x, df['pp2'][i::nwalkers])
plt.ylabel('$pp2$', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,11)
for i in range(0, nwalkers):
    plt.plot(x, df['pp3'][i::nwalkers])
plt.ylabel('$pp3$', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,12)
for i in range(0, nwalkers):
    plt.plot(x, df['rt1'][i::nwalkers])
plt.ylabel('$rt_1$', fontsize=7, rotation=0, labelpad=40)
#plt.yticks(fontsize=5)

plt.subplot(7,2,13)
for i in range(0, nwalkers):
    plt.plot(x, df['rt2'][i::nwalkers])
plt.ylabel('$rt_2$', fontsize=7, rotation=0, labelpad=40)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)


plt.subplot(7,2,14)
for i in range(0, nwalkers):
    plt.plot(x, df['lnprob'][i::nwalkers])
plt.ylabel('lnprob', fontsize=7, rotation=0, labelpad=40)
plt.tight_layout(pad=3.0)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.005)

"""
cmap_light = sns.diverging_palette(220, 20, center='dark', n=nwalkers)
#colors = ['red', 'blue', 'green', 'purple', 'yellow', 'black']


fig, ax = plt.subplots()
#fig.tight_layout(pad=3.0)
w1m = df['r_in'][0::nwalkers]
w2m = df['delta_r'][1::nwalkers]
fig, (ax0,ax1,ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13) = plt.subplots(ndim+1)
x = np.arange(0,len(w1m))
print(np.shape(x),np.shape(w1m))
print(np.shape(x),np.shape(w2m))
for i in range(0, nwalkers):
        ax0.plot(x, df['r_in'][i::nwalkers])
        ax1.plot(x, df['delta_r'][i::nwalkers])
        ax2.plot(x, df['m_disk'][i::nwalkers])
        ax3.plot(x, df['f_star'][i::nwalkers])
        ax4.plot(x, df['cos_position_angle'][i::nwalkers])
        ax5.plot(x, df['cos_inclination'][i::nwalkers])
        ax6.plot(x, df['xoffs_stellar'][i::nwalkers])
        ax7.plot(x, df['yoffs_stellar'][i::nwalkers])
        ax8.plot(x, df['pp1'][i::nwalkers])
        ax9.plot(x, df['pp2'][i::nwalkers])
        ax10.plot(x, df['pp3'][i::nwalkers])
        ax11.plot(x, df['rt1'][i::nwalkers])
        ax12.plot(x, df['rt2'][i::nwalkers])
        ax13.plot(x, df['lnprob'][i::nwalkers])
        
        #fig.suptitle('r_in, delta_r, m_disk, f_star, position_angle, inclination, xoffs_stellar, yoffs_stellar, pp1, rt, pp2, lnprob')
        fig.suptitle("Walker Evolution Plot")
        ax0.set_ylabel("$R_i$$_n$ (au)", fontsize=10)
        ax0.tick_params(axis='both', which='major', labelsize=5)
        ax1.set_ylabel("$\Delta$ R (au)")
        ax1.tick_params(axis='both', which='major', labelsize=5)
        ax2.set_ylabel("Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)")
        ax2.tick_params(axis='both', which='major', labelsize=5)
        ax3.set_ylabel("f$_s$$_t$$_a$$_r$ ($\mu$ Jy)")
        ax3.tick_params(axis='both', which='major', labelsize=5)
        ax4.set_ylabel("PA ($\degree$)")
        ax4.tick_params(axis='both', which='major', labelsize=5)
        ax5.set_ylabel("i ($\degree$)")
        ax5.tick_params(axis='both', which='major', labelsize=5)
        ax6.set_ylabel('$\Delta$ x (")')
        ax6.tick_params(axis='both', which='major', labelsize=5)
        ax7.set_ylabel('$\Delta$ y (")')
        ax7.tick_params(axis='both', which='major', labelsize=5)
        ax8.set_ylabel("$pp_1$")
        ax8.tick_params(axis='both', which='major', labelsize=5)
        ax9.set_ylabel("$pp_2$")
        ax9.tick_params(axis='both', which='major', labelsize=5)
        ax10.set_ylabel("$pp_3$")
        ax10.tick_params(axis='both', which='major', labelsize=5)
        ax11.set_ylabel("$rt_1$")
        ax11.tick_params(axis='both', which='major', labelsize=5)
        ax12.set_ylabel("$rt_2$")
        ax12.tick_params(axis='both', which='major', labelsize=5)
        ax13.set_ylabel("lnprob")
        ax13.tick_params(axis='both', which='major', labelsize=5)
        
        fig.tight_layout(pad=3.0)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.001)
        plt.show(block=False)
        #plt.savefig("walker_july29.pdf")
        print(np.shape(x),np.shape(w1m))

"""


"""
#df = old_df[~old_df.lnprob.isin(x)]
#ndim = len(df.columns) - 1
#nsamples =
#nsamples, ndim = df.shape
nsamples = 60000
ndim = 15
#Here: see if you can get ndim and nsamples from .csv file directly
ndim = ndim - 2
print(ndim)
print(nsamples)
max_lnprob = df['lnprob'].max()
max_r_in = df.r_in[df.lnprob.idxmax()]
max_delta_r = df.delta_r[df.lnprob.idxmax()]
max_m_disk = df.m_disk[df.lnprob.idxmax()]
max_f_star = df.f_star[df.lnprob.idxmax()]
max_position_angle = df.cos_position_angle[df.lnprob.idxmax()]
max_inclination = df.cos_inclination[df.lnprob.idxmax()]
max_xoffs_stellar = df.xoffs_stellar[df.lnprob.idxmax()]
max_yoffs_stellar = df.yoffs_stellar[df.lnprob.idxmax()]
max_pp1 = df.pp1[df.lnprob.idxmax()]
max_rt1 = df.rt1[df.lnprob.idxmax()]
max_pp2 = df.pp2[df.lnprob.idxmax()]
max_pp3 = df.pp3[df.lnprob.idxmax()]
max_rt2 = df.rt2[df.lnprob.idxmax()]

#sampler = emcee.EnsembleSampler(16, ndim, max_lnprob)
#samples = sampler.chain[:,:,:].reshape((-1, ndim))
samples = np.zeros([nsamples,ndim])

samples[:,0] = r_in
samples[:,1] = delta_r
samples[:,2] = m_disk
samples[:,3] = f_star
samples[:,4] = position_angle
samples[:,5] = inclination
samples[:,6] = xoffs
samples[:,7] = yoffs
samples[:,8] = pp1
samples[:,9] = pp2
samples[:,10] = pp3
samples[:,11] = rt1
samples[:,12] = rt2

#fig = corner.corner(samples, labels=["$r_in$", "$delta_r$"],truths=[max_r_in, max_delta_r])
fig = corner.corner(samples, labels=["$r_{in}$", "$\Delta_r$", "$m_{disk}$", "$f_{star}$", "$position angle$", "$inclination$", "$\Delta_x$", "$\Delta_y$", "$pp1$", "$pp2$", "$pp3$", "$rt1$","$rt2$"],truths=[max_r_in, max_delta_r, max_m_disk, max_f_star, max_position_angle, max_inclination, max_xoffs_stellar, max_yoffs_stellar, max_pp1, max_pp2, max_pp3, max_rt1, max_rt2])
fig.show()
fig.savefig("cornerplot_july20.png")


"""





#end of file:)
