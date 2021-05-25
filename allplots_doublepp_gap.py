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
nwalkers= 29 #30
ndim=13



#ndim=6
#old_df = pd.read_csv("chain_20steps_new8params.csv", skiprows=range(1,2401))
#df = pd.read_csv("chain_1200steps_july27_doublepp_gap.csv")
df = pd.read_csv("doublepp_combined_2850.csv", skiprows=np.append(range(2, 60000, 30), range(60003, 85500, 30))) #60000
#df = pd.read_csv("doublepp_combined_2850.csv", skiprows=np.append(range(1, 24000, 30), range(24002, 60000, 30), range(60003, 85500, 30)))
#x = [-np.inf]
#df = pd.read_csv("doublepp_gap_850steps_jan6.csv")
nsamples, ndim = df.shape
print(nsamples)
##################
#BEST FIT
##################

df.to_csv("file1.csv")
df = pd.read_csv("file1.csv", skiprows=range(1, 24680))

nsamples, ndim = df.shape
print(nsamples)

"""

#finding the percentiles for each of the parameters
r_in_16 = np.percentile(r_in, 16)
r_in_50 = np.percentile(r_in, 50)
r_in_84 = np.percentile(r_in, 84)
r_in_99 = np.percentile(r_in, 99.7)
print('r\N{SUBSCRIPT TWO}')
print(r_in_16)
print(r_in_50)
print(r_in_84)
print(r_in_99)
delta_r_16 = np.percentile(delta_r, 16)
delta_r_50 = np.percentile(delta_r, 50)
delta_r_84 = np.percentile(delta_r, 84)
#delta_r_84 = np.percentile(delta_r, 84)
print("delta_r:")
print(delta_r_16)
print(delta_r_50)
print(delta_r_84)
r_out_16 = np.percentile(r_out, 16)
r_out_50 = np.percentile(r_out, 50)
r_out_84 = np.percentile(r_out, 84)
print("r_out")
print(r_out_16)
print(r_out_50)
print(r_out_84)
m_disk_16 = np.percentile(m_disk, 16)
m_disk_50 = np.percentile(m_disk, 50)
m_disk_84 = np.percentile(m_disk, 84)
print("m_disk:")
print(m_disk_16)
print(m_disk_50)
print(m_disk_84)
f_star_16 = np.percentile(f_star, 16)
f_star_50 = np.percentile(f_star, 50)
f_star_84 = np.percentile(f_star, 84)
print("f_star:")
print(f_star_16)
print(f_star_50)
print(f_star_84)
position_angle_16 = np.percentile(position_angle, 16)
position_angle_50 = np.percentile(position_angle, 50)
position_angle_84 = np.percentile(position_angle, 84)
print("position_angle:")


"""



max_lnprob = df['lnprob'].max()
max_r_in = df.r_in[df.lnprob.idxmax()]
max_delta_r = df.delta_r[df.lnprob.idxmax()]
max_m_disk = df.m_disk[df.lnprob.idxmax()]
max_f_star = df.f_star[df.lnprob.idxmax()]
max_position_angle = df.cos_position_angle[df.lnprob.idxmax()]
max_inclination = df.cos_inclination[df.lnprob.idxmax()]
#max_xoffs = df.xoffs[df.lnprob.idxmax()] #nogap
#max_yoffs = df.yoffs[df.lnprob.idxmax()] #nogap
max_xoffs_stellar = df.xoffs_stellar[df.lnprob.idxmax()] #gap
max_yoffs_stellar = df.yoffs_stellar[df.lnprob.idxmax()] #gap
max_r_in_gap = df.r_in_gap[df.lnprob.idxmax()] #gap
max_delta_r_gap = df.delta_r_gap[df.lnprob.idxmax()] #gap
max_pp1 = df.pp1[df.lnprob.idxmax()]
max_pp2 = df.pp2[df.lnprob.idxmax()]
#max_pp3 = df.pp3[df.lnprob.idxmax()]
#max_rt1 = df.rt1[df.lnprob.idxmax()]
#max_rt2 = df.rt2[df.lnprob.idxmax()]
max_rt = df.rt[df.lnprob.idxmax()]

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
print("r_in_gap =", max_r_in_gap)
print("delta_r_gap =", max_delta_r_gap)
print("pp1 =", max_pp1)
print("pp2 =", max_pp2)
#print("pp3 =", max_pp3)
#print("rt1 =", max_rt1)
#print("rt2 =", max_rt2)
print("rt =", max_rt)








#####################
#WALKER EVOLUTION PLOT
#####################


cmap_light = sns.diverging_palette(220, 20, center='dark', n=nwalkers)
#colors = ['red', 'blue', 'green', 'purple', 'yellow', 'black']
#fig, ax = plt.subplots()
'''for i in range(nwalkers):
    #c = colors[i]
    ax.plot(df['r_in'][i::nwalkers], df['delta_r'][i::nwalkers], linestyle='-', marker='.', alpha=0.5)
plt.show(block=False)'''

w1m = df['r_in'][0::nwalkers]
w2m = df['delta_r'][1::nwalkers]

x = np.arange(0,len(w1m))

plt.figure(figsize=(17.0,17.0))

plt.subplot(7,2,1)
for i in range(0, nwalkers):
    plt.plot(x, df['r_in'][i::nwalkers])
plt.ylabel('$R_i$$_n$ (au)', fontsize=12,rotation=90, labelpad=10)
#plt.title("Walker Evolution Plot")
#plt.set_ylim([0, 20])
plt.yticks(fontsize=7)
plt.xticks([])
plt.ylim(0, 75)

plt.subplot(7,2,2)
for i in range(0, nwalkers):
    plt.plot(x, df['delta_r'][i::nwalkers])
plt.ylabel('$\Delta$ R (au)', fontsize=12,rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,3)
for i in range(0, nwalkers):
    plt.plot(x, df['m_disk'][i::nwalkers])
plt.ylabel('Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])
plt.ylim(-6.8, -7.4)

plt.subplot(7,2,4)
for i in range(0, nwalkers):
    plt.plot(x, df['f_star'][i::nwalkers]*1e6)
plt.ylabel('F$_s$$_t$$_a$$_r$ ($\mu$ Jy)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,5)
for i in range(0, nwalkers):
    plt.plot(x, df['cos_position_angle'][i::nwalkers])
plt.ylabel('PA ($\degree$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,6)
for i in range(0, nwalkers):
    plt.plot(x, df['cos_inclination'][i::nwalkers])
plt.ylabel('i ($\degree$)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,7)
for i in range(0, nwalkers):
    plt.plot(x, df['xoffs_stellar'][i::nwalkers])
plt.ylabel('$\Delta$ x (")', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,8)
for i in range(0, nwalkers):
    plt.plot(x, df['yoffs_stellar'][i::nwalkers])
plt.ylabel('$\Delta$ y (")', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,9)
for i in range(0, nwalkers):
    plt.plot(x, df['pp1'][i::nwalkers])
plt.ylabel('$pp1$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,10)
for i in range(0, nwalkers):
    plt.plot(x, df['pp2'][i::nwalkers])
plt.ylabel('$pp2$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,11)
for i in range(0, nwalkers):
    plt.plot(x, df['rt'][i::nwalkers])
plt.ylabel('$R_t$', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,12)
for i in range(0, nwalkers):
    plt.plot(x, df['r_in_gap'][i::nwalkers])
plt.ylabel('R$_i$$_n$$_G$$_a$$_p$ (au)', fontsize=12, rotation=90, labelpad=10)
plt.yticks(fontsize=7)
plt.xticks([])

plt.subplot(7,2,13)
for i in range(0, nwalkers):
    plt.plot(x, df['delta_r_gap'][i::nwalkers])
plt.ylabel('$\Delta$ $R_{gap}$ (au)', fontsize=12, rotation=90, labelpad=10)
plt.xlabel("Step", fontsize=15)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)


plt.subplot(7,2,14)
for i in range(0, nwalkers):
    plt.plot(x, np.array(df['lnprob'][i::nwalkers])/1e7)
plt.ylabel('lnprob', fontsize=12, rotation=90, labelpad=10)
plt.xlabel('Step', fontsize=15)
plt.ylim(-1.073303, -1.0732990)
#plt.tight_layout(pad=3.0)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)

#plt.set_size_inches(18.5, 10.5, forward=True)
#print(np.median(df['lnprob']))
plt.savefig("walker_aug12_doublepp_gap.pdf", bbox_inches="tight")
#plt.savefig("walker_aug10_doublepp_gap.png", bbox_inches="tight")







############
#CORNER PLOT
############
df = pd.read_csv("doublepp_combined_2850.csv", skiprows=np.append(range(2, 60000, 30), range(60003, 85500, 30)))
#df = pd.read_csv("doublepp_gap_2000.csv", skiprows=range(1, 24750, 30))

#df = old_df[~old_df.lnprob.isin(x)]
#ndim = len(df.columns) - 1
#nsamples =
nsamples, ndim = df.shape
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
max_rt = df.rt[df.lnprob.idxmax()]
max_pp2 = df.pp1[df.lnprob.idxmax()]
max_r_in_gap = df.r_in_gap[df.lnprob.idxmax()]
max_delta_r_gap = df.delta_r_gap[df.lnprob.idxmax()]


#sampler = emcee.EnsembleSampler(16, ndim, max_lnprob)
#samples = sampler.chain[:,:,:].reshape((-1, ndim))
samples = np.zeros([nsamples,ndim])
samples[:,0] = df['r_in']
samples[:,1] = df['delta_r']
samples[:,2] = df['m_disk']
samples[:,3] = df['f_star']*1e6
samples[:,4] = df['cos_position_angle']
samples[:,5] = df['cos_inclination']
samples[:,6] = df['xoffs_stellar']
samples[:,7] = df['yoffs_stellar']
samples[:,8] = df['pp1']
samples[:,9] = df['rt']
samples[:,10] = df['pp2']
samples[:,11] = df['r_in_gap']
samples[:,12] = df['delta_r_gap']
#samples[24650:,:]
#fig = corner.corner(samples, labels=["$r_in$", "$delta_r$"],truths=[max_r_in, max_delta_r])
fig = corner.corner(samples, labels=["$R_{in}$ (au)", "$\Delta$ R (au)", "Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)", "F$_s$$_t$$_a$$_r$ ($\mu$ Jy)", "$PA (^\circ)$", "$i (^\circ)$", '$\Delta$ x (")', '$\Delta_y$ (")', "$pp1$", "$R_t$", "$pp2$", "$R_{inGap}$ (au)","$\Delta R_{Gap}$ (au)"], max_n_ticks=2, label_kwargs={"fontsize": 22},truths=[max_r_in, max_delta_r, max_m_disk, max_f_star, max_position_angle, max_inclination, max_xoffs_stellar, max_yoffs_stellar, max_pp1, max_rt, max_pp2, max_r_in_gap, max_delta_r_gap])
#fig.tick_params(axis='both', labelsize=15)
fig.savefig("cornerplot_aug13_doublepp_gap.pdf")
