from emcee.autocorr import integrated_time
import csv
import numpy as np
import pandas as pd
#import matplotlib as plti
import matplotlib.pyplot as plt

df = pd.read_csv("file1.csv", skiprows=range(1, 24651))


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
rt = df.rt
pp2 = df.pp2
r_in_gap = df.r_in_gap
delta_r_gap = df.delta_r_gap

nwalkers = 29
nsteps = 2000#2850
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
rt_ar = np.empty((nsteps, nwalkers))
r_in_gap_ar = np.empty((nsteps, nwalkers))
delta_r_gap_ar = np.empty((nsteps, nwalkers))

print(r_in_ar.shape)
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
    #pp3_ar[:, i] = pp3[i::nwalkers]
    rt_ar[:, i] = rt[i::nwalkers]
    #rt2_ar[:, i] = rt2[i::nwalkers]
    r_in_gap_ar[:, i] = r_in_gap[i::nwalkers]
    delta_r_gap_ar[:, i] = delta_r_gap[i::nwalkers]

tau_r_in = np.empty((nsteps-10, nwalkers))
tau_delta_r = np.empty((nsteps-10, nwalkers))
tau_m_disk = np.empty((nsteps-10, nwalkers))
tau_f_star = np.empty((nsteps-10, nwalkers))
tau_position_angle = np.empty((nsteps-10, nwalkers))
tau_inclination = np.empty((nsteps-10, nwalkers))
tau_xoffs_stellar = np.empty((nsteps-10, nwalkers))
tau_yoffs_stellar = np.empty((nsteps-10, nwalkers))
tau_pp1 = np.empty((nsteps-10, nwalkers))
tau_pp2 = np.empty((nsteps-10, nwalkers))
#tau_pp3 = np.empty((nsteps-10, nwalkers))
tau_rt = np.empty((nsteps-10, nwalkers))
#tau_rt2 = np.empty((nsteps-10, nwalkers))
tau_r_in_gap = np.empty((nsteps-10, nwalkers))
tau_delta_r_gap = np.empty((nsteps-10, nwalkers))

plt.figure(figsize=(17.0,17.0))

plt.subplot(471)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_r_in[i-10, j] = integrated_time(r_in_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_r_in[:,j][np.isnan(tau_r_in[:,j])] = 1
    plt.plot(tau_r_in[:,j], alpha=.5)
plt.plot(tau_r_in.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_r_in[-1,:]),color='k',ls=':',lw=2)
plt.ylabel('Integrated Autocorrelation Time')
plt.title("$R_i$$_n$ (au)")
#plt.set_ylim([0, 20])


plt.subplot(4,7,8)
var = np.std(r_in_ar, axis=1)**2.
mean = np.mean(r_in_ar,axis=1)
frac_error = np.sqrt(var[10:]*tau_r_in.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:])
plt.plot(frac_error,color='k',label='frac. error in mean')
std_dev=np.sqrt(var[10:])/np.abs(mean[10:])
plt.plot(std_dev,color='r',label='(std dev)/mean')
plt.ylim(0, 1.0)
plt.legend(prop=dict(size=7)) #LEGEND HERE!!! 


plt.subplot(472)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_delta_r[i-10, j] = integrated_time(delta_r_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_delta_r[:,j][np.isnan(tau_delta_r[:,j])] = 1
    plt.plot(tau_delta_r[:,j], alpha=.5)
plt.plot(tau_delta_r.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_delta_r[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("$\Delta$ R (au)")

plt.subplot(479)
var = np.std(delta_r_ar, axis=1)**2.
mean = np.mean(delta_r_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_delta_r.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
plt.ylim(0, 0.3)
#plt.xlabel('Step Number')
#plt.legend(frameon=False)


plt.subplot(473)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_m_disk[i-10, j] = integrated_time(m_disk_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_m_disk[:,j][np.isnan(tau_m_disk[:,j])] = 1
    plt.plot(tau_m_disk[:,j], alpha=.5)
plt.plot(tau_m_disk.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_m_disk[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)")


plt.subplot(4,7,10)
var = np.std(m_disk_ar, axis=1)**2.
mean = np.mean(m_disk_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_m_disk.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
plt.ylim(0, 0.05)
#plt.xlabel('Step Number')
#plt.legend(frameon=False)


plt.subplot(474)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_f_star[i-10, j] = integrated_time(f_star_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_f_star[:,j][np.isnan(tau_f_star[:,j])] = 1
    plt.plot(tau_f_star[:,j], alpha=.5)
plt.plot(tau_f_star.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_f_star[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("F$_s$$_t$$_a$$_r$ ($\mu$ Jy)")


plt.subplot(4,7,11)
var = np.std(f_star_ar, axis=1)**2.
mean = np.mean(f_star_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_f_star.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
#plt.xlabel('Step Number')
#plt.legend(frameon=False)


plt.subplot(475)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_position_angle[i-10, j] = integrated_time(position_angle_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_position_angle[:,j][np.isnan(tau_position_angle[:,j])] = 1
    plt.plot(tau_position_angle[:,j], alpha=.5)
plt.plot(tau_position_angle.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_position_angle[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("PA ($\degree$)")

plt.subplot(4,7,12)
var = np.std(position_angle_ar, axis=1)**2.
mean = np.mean(position_angle_ar,axis=1)
frac_error = np.sqrt(var[10:]*tau_position_angle.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:])
plt.plot(frac_error,color='k',label='frac. error in mean')
std_dev=np.sqrt(var[10:])/np.abs(mean[10:])
plt.ylim(0, 0.25)
plt.plot(std_dev,color='r',label='(std dev)/mean')
#plt.legend(frameon=False) 


plt.subplot(476)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_inclination[i-10, j] = integrated_time(inclination_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_inclination[:,j][np.isnan(tau_inclination[:,j])] = 1
    plt.plot(tau_inclination[:,j], alpha=.5)
plt.plot(tau_inclination.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_inclination[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("i ($\degree$)")


plt.subplot(4,7,13)
var = np.std(inclination_ar, axis=1)**2.
mean = np.mean(inclination_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_inclination.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
#plt.xlabel('Step Number')
#plt.legend(frameon=False)



plt.subplot(477)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_xoffs_stellar[i-10, j] = integrated_time(xoffs_stellar_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_xoffs_stellar[:,j][np.isnan(tau_xoffs_stellar[:,j])] = 1
    plt.plot(tau_xoffs_stellar[:,j], alpha=.5)
plt.plot(tau_xoffs_stellar.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_xoffs_stellar[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title('$\Delta$ x (")')


plt.subplot(4,7,14)
var = np.std(xoffs_stellar_ar, axis=1)**2.
mean = np.mean(xoffs_stellar_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_xoffs_stellar.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
plt.ylim(0, 1)
#plt.xlabel('Step Number')
#plt.legend(frameon=False)



plt.subplot(4,7,15)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_yoffs_stellar[i-10, j] = integrated_time(yoffs_stellar_ar[:i,j], quiet=True)
    tau_yoffs_stellar[:,j][np.isnan(tau_yoffs_stellar[:,j])] = 1
    plt.plot(tau_yoffs_stellar[:,j], alpha=.5)
plt.plot(tau_yoffs_stellar.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_yoffs_stellar[-1,:]),color='k',ls=':',lw=2)
plt.ylabel('Integrated Autocorrelation Time')
plt.title('$\Delta$ y (")')


plt.subplot(4,7,22)
var = np.std(yoffs_stellar_ar, axis=1)**2.
mean = np.mean(yoffs_stellar_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_yoffs_stellar.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
plt.ylim(0, 10.0)
plt.xlabel('Step Number')
#plt.legend(frameon=False)


plt.subplot(4,7,16)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_pp1[i-10, j] = integrated_time(pp1_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_pp1[:,j][np.isnan(tau_pp1[:,j])] = 1
    plt.plot(tau_pp1[:,j], alpha=.5)
plt.plot(tau_pp1.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_pp1[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("pp1")



plt.subplot(4,7,23)
var = np.std(pp1_ar, axis=1)**2.
mean = np.mean(pp1_ar,axis=1)
frac_error = np.sqrt(var[10:]*tau_pp1.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:])
plt.plot(frac_error,color='k',label='frac. error in mean')
std_dev=np.sqrt(var[10:])/np.abs(mean[10:])
plt.ylim(0, 2)
plt.plot(std_dev,color='r',label='(std dev)/mean')
plt.xlabel('Step Number')


plt.subplot(4,7,17)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_pp2[i-10, j] = integrated_time(pp2_ar[:i,j], quiet=True)
    tau_pp2[:,j][np.isnan(tau_pp2[:,j])] = 1
    plt.plot(tau_pp2[:,j], alpha=.5)
plt.plot(tau_pp2.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_pp2[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("pp2")


plt.subplot(4,7,24)
var = np.std(pp2_ar, axis=1)**2.
mean = np.mean(pp2_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_pp2.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
plt.ylim(0, 0.5)
plt.xlabel('Step Number')



plt.subplot(4,7,18)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_rt[i-10, j] = integrated_time(rt_ar[:i,j], quiet=True)
    tau_rt[:,j][np.isnan(tau_rt[:,j])] = 1
    plt.plot(tau_rt[:,j], alpha=.5)
plt.plot(tau_rt.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_rt[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("rt")

plt.subplot(4,7,25)
var = np.std(rt_ar, axis=1)**2.
mean = np.mean(rt_ar,axis=1)
frac_error = np.sqrt(var[10:]*tau_rt.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:])
plt.plot(frac_error,color='k',label='frac. error in mean')
std_dev=np.sqrt(var[10:])/np.abs(mean[10:])
plt.plot(std_dev,color='r',label='(std dev)/mean')
plt.xlabel('Step Number')



plt.subplot(4,7,19)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_r_in_gap[i-10, j] = integrated_time(r_in_gap_ar[:i,j], quiet=True)
    tau_r_in_gap[:,j][np.isnan(tau_r_in_gap[:,j])] = 1
    plt.plot(tau_r_in_gap[:,j], alpha=.5)
plt.plot(tau_r_in_gap.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_r_in_gap[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("R$_i$$_n$$_G$$_a$$_p$ (au)")



plt.subplot(4,7,26)
var = np.std(r_in_gap_ar, axis=1)**2.
mean = np.mean(r_in_gap_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_r_in_gap.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
plt.xlabel('Step Number')




plt.subplot(4,7,20)
for j in range(nwalkers):
    for i in range(10,nsteps):
        tau_delta_r_gap[i-10, j] = integrated_time(delta_r_gap_ar[:i,j], quiet=True)
        #tau2[i-10, j] = integrated_time(tau[:i,j], quiet=True, tol=1)
    tau_delta_r_gap[:,j][np.isnan(tau_delta_r_gap[:,j])] = 1
    plt.plot(tau_delta_r_gap[:,j], alpha=.5)
plt.plot(tau_delta_r_gap.mean(axis=1), color='k', ls='--', lw=2)
plt.axhline(np.mean(tau_delta_r_gap[-1,:]),color='k',ls=':',lw=2)
#plt.ylabel('Integrated Autocorrelation Time')
plt.title("$\Delta$ $R_{gap}$ (au)")


plt.subplot(4,7,27)
var = np.std(delta_r_gap_ar, axis=1)**2.
mean = np.mean(delta_r_gap_ar,axis=1)
plt.plot(np.sqrt(var[10:]*tau_delta_r_gap.mean(axis=1)/(nwalkers*np.arange(10,nsteps)))/np.abs(mean[10:]),color='k',label='frac. error in mean')
plt.plot(np.sqrt(var[10:])/np.abs(mean[10:]),color='r',label='(std dev)/mean')
plt.xlabel('Step Number')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=None)

plt.savefig("autocor_doublepp_gap_finalv2.pdf", bbox_inches='tight')

