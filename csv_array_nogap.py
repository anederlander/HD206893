import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import sys
import corner
import csv
import pylab
import seaborn as sns
"""
with open('combined_triplepp.csv', 'r') as fin, open('combined_triplepp1.csv', 'w', newline='') as fout:

    # define reader and writer objects
    reader = csv.reader(fin, skipinitialspace=True)
    writer = csv.writer(fout, delimiter=',')

    # write headers
    writer.writerow(next(reader))
    k=0
    # iterate and write rows based on condition
    for i in reader:
        if (float(i[1]) > 0):
            #if ((float(i[8])) < (0.5)):
                #if ((float(i[7])) > (0.5)):
                    #if ((float(i[14])) < (75)):
            writer.writerow(i)
            k+=1

#df = csv.reader(open("combined_triplepp1.csv", 'rt'), delimiter=",", quotechar="|")
"""



with open('chain_1000steps_oct12_triplepp.csv', 'r') as fin, open('chain_1000steps_oct12_triplepp1.csv', 'w', newline='') as fout:

    # define reader and writer objects
    reader = csv.reader(fin, skipinitialspace=True)
    writer = csv.writer(fout, delimiter=',')

    # write headers
    writer.writerow(next(reader))
    k=0
    # iterate and write rows based on condition
    for i in reader:
        if (float(i[10]) > 0):
            #if ((float(i[8])) < (0.5)):
                #if ((float(i[7])) > (0.5)):
                    #if ((float(i[14])) < (75)):
            writer.writerow(i)
            k+=1

df = csv.reader(open("chain_1000steps_oct12_triplepp1.csv", 'rt'), delimiter=",", quotechar="|")


#df = csv.reader(open("chain_2000steps_oct4_triplepp.csv", 'rt'), delimiter=",", quotechar="|")
i#df = csv.reader(open("chain_1000steps_oct12_triplepp.csv", 'rt'), delimiter=",", quotechar="|")

freq, r_in, delta_r, m_disk, f_star, position_angle, inclination, xoffs_stellar, yoffs_stellar, pp1, pp2, pp3, rt1, rt2 = [], [], [], [], [], [], [], [], [], [], [], [], [], []

#i = 0
#while i < 1600:
    #next(df)
    #i += 1
#next(df)
next(df)
for row in df:
    freq.append((row[0]))
    r_in.append(float(row[1]))
    delta_r.append(float(row[2]))
    m_disk.append(float(row[3]))
    f_star.append(float(row[4]))
    position_angle.append(float(row[5]))
    inclination.append(float(row[6]))
    xoffs_stellar.append(float(row[7]))
    yoffs_stellar.append(float(row[8]))
    pp3.append(float(row[11]))
    rt1.append(float(row[12]))
    pp1.append(float(row[9]))
    rt2.append(float(row[13]))
    pp2.append(float(row[10]))

print("Here")
"""
k=0
while k<np.size(f_star):
    f_star[k]=f_star[k]*1e6
    k+=1
    #print(k)
k=0
print("finished f_star")
while k<np.size(m_disk):
    m_disk[k]=m_disk[k]+(np.log10((1.99e33 / 5.976e27)))
    k+=1
print("finished m disk")
print("included", (np.size(r_in))/24)

r_out=np.zeros(np.size(r_in))
k=0
while k<np.size(r_in):
    r_out[k] = r_in[k] + delta_r[k]
    k+=1
print("finished r out")
#print("HERE")
"""



k=0
while k<len(f_star):
    f_star[k]=f_star[k]*1e6 
    k+=1
"""
r_out=np.zeros(8600)
k=0
while k<8600:
    r_out[k] = r_in[k] + delta_r[k]
    k+=1
"""
#f_star = f_star*1e6
r_out = np.zeros(np.size(r_in))
#r_out = r_in + delta_r
r_out = np.asarray(r_in) + np.asarray(delta_r)
m_disk = m_disk + (np.log10((1.99e33 / 5.976e27)))


#Sorting each parameter array in ascending order
r_in.sort()
delta_r.sort()
m_disk.sort()
f_star.sort()
position_angle.sort()
inclination.sort()
#max_xoffs_stellar = df1.xoffs_stellar[df1.lnprob.idxmax()] #gap
#max_yoffs_stellar = df1.yoffs_stellar[df1.lnprob.idxmax()] #gap
#r_in_gap.sort()
#delta_r_gap.sort()
#r_out.sort()
pp1.sort()
pp2.sort()
pp3.sort()
rt1.sort()
rt2.sort()
#print(f_star)

print("HERE")

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
print(position_angle_16)
print(position_angle_50)
print(position_angle_84)
inclination_16 = np.percentile(inclination, 16)
inclination_50 = np.percentile(inclination, 50)
inclination_84 = np.percentile(inclination, 84)
print("inclination:")
print(inclination_16)
print(inclination_50)
print(inclination_84)
xoffs_stellar_16 = np.percentile(xoffs_stellar, 16)
xoffs_stellar_50 = np.percentile(xoffs_stellar, 50)
xoffs_stellar_84 = np.percentile(xoffs_stellar, 84)
print("xoffs_stellar:")
print(xoffs_stellar_16)
print(xoffs_stellar_50)
print(xoffs_stellar_84)
yoffs_stellar_16 = np.percentile(yoffs_stellar, 16)
yoffs_stellar_50 = np.percentile(yoffs_stellar, 50)
yoffs_stellar_84 = np.percentile(yoffs_stellar, 84)
print("yoffs_stellar:")
print(yoffs_stellar_16)
print(yoffs_stellar_50)
print(yoffs_stellar_84)
#r_in_gap_16 = np.percentile(r_in_gap, 16)
#r_in_gap_50 = np.percentile(r_in_gap, 50)
#r_in_gap_84 = np.percentile(r_in_gap, 84)
#print("r_in_gap:")
#print(r_in_gap_16)
#print(r_in_gap_50)
#print(r_in_gap_84)
#delta_r_gap_16 = np.percentile(delta_r_gap, 16)
#delta_r_gap_50 = np.percentile(delta_r_gap, 50)
#delta_r_gap_84 = np.percentile(delta_r_gap, 84)
#print("delta_r_gap:")
#print(delta_r_gap_16)
#print(delta_r_gap_50)
#print(delta_r_gap_84)
pp1_16 = np.percentile(pp1, 16)
pp1_50 = np.percentile(pp1, 50)
pp1_84 = np.percentile(pp1, 84)
print("pp1:")
print(pp1_16)
print(pp1_50)
print(pp1_84)
pp2_16 = np.percentile(pp2, 16)
pp2_50 = np.percentile(pp2, 50)
pp2_84 = np.percentile(pp2, 84)
pp3_point3 = np.percentile(pp2, 1)
print("pp2:")
print(pp2_16)
print(pp2_50)
print(pp2_84)
print(pp3_point3)
pp3_16 = np.percentile(pp3, 16)
pp3_50 = np.percentile(pp3, 50)
pp3_84 = np.percentile(pp3, 84)
#pp3_point3 = np.percentile(pp3, .3)
print("pp3:")
print(pp3_16)
print(pp3_50)
print(pp3_84)
#print(pp3_point3)
rt1_16 = np.percentile(rt1, 16)
rt1_50 = np.percentile(rt1, 50)
rt1_84 = np.percentile(rt1, 84)
print("rt1:")
print(rt1_16)
print(rt1_50)
print(rt1_84)
rt2_16 = np.percentile(rt2, 16)
rt2_50 = np.percentile(rt2, 50)
rt2_84 = np.percentile(rt2, 84)
print("rt2:")
print(rt2_16)
print(rt2_50)
print(rt2_84)
#including subplots



fig, axs = plt.subplots(2, 7)

axs[0, 0].hist(r_in, bins=23, range=[0, 50])
axs[0, 0].axvline(r_in_16, color='k', linestyle='solid', linewidth=1)
axs[0, 0].axvline(r_in_50, color='k', linestyle='dashed', linewidth=1)
axs[0, 0].axvline(r_in_84, color='k', linestyle='solid', linewidth=1)
axs[0, 0].set_title('$R_i$$_n$ (au)', fontsize=8)
axs[0, 0].tick_params(labelsize = 6)

axs[0, 1].hist(delta_r, bins=23)
axs[0, 1].axvline(delta_r_16, color='k', linestyle='solid', linewidth=1)
axs[0, 1].axvline(delta_r_50, color='k', linestyle='dashed', linewidth=1)
axs[0, 1].axvline(delta_r_84, color='k', linestyle='solid', linewidth=1)
axs[0, 1].set_title('$\Delta$ R (au)', fontsize=8)
axs[0, 1].tick_params(labelsize = 6)

axs[0, 2].hist(m_disk, bins=23, range=[-1.9, -1.7])
axs[0, 2].axvline(m_disk_16, color='k', linestyle='solid', linewidth=1)
axs[0, 2].axvline(m_disk_50, color='k', linestyle='dashed', linewidth=1)
axs[0, 2].axvline(m_disk_84, color='k', linestyle='solid', linewidth=1)
axs[0, 2].set_title('Log(M$_d$$_i$$_s$$_k$) (M$_\oplus$)', fontsize=8)
axs[0, 2].tick_params(labelsize = 6)

axs[0, 3].hist(f_star, bins=23)
axs[0, 3].axvline(f_star_16, color='k', linestyle='solid', linewidth=1)
axs[0, 3].axvline(f_star_50, color='k', linestyle='dashed', linewidth=1)
axs[0, 3].axvline(f_star_84, color='k', linestyle='solid', linewidth=1)
axs[0, 3].set_title('f$_s$$_t$$_a$$_r$ ($\mu$ Jy)', fontsize=8)
axs[0, 3].tick_params(labelsize = 6)

axs[0, 4].hist(position_angle, bins=23, range=[.3, .7])
axs[0, 4].axvline(position_angle_16, color='k', linestyle='solid', linewidth=1)
axs[0, 4].axvline(position_angle_50, color='k', linestyle='dashed', linewidth=1)
axs[0, 4].axvline(position_angle_84, color='k', linestyle='solid', linewidth=1)
axs[0, 4].set_title('PA ($\degree$)', fontsize=8)
axs[0, 4].tick_params(labelsize = 6)

axs[0, 5].hist(inclination, bins=23)
axs[0, 5].axvline(inclination_16, color='k', linestyle='solid', linewidth=1)
axs[0, 5].axvline(inclination_50, color='k', linestyle='dashed', linewidth=1)
axs[0, 5].axvline(inclination_84, color='k', linestyle='solid', linewidth=1)
axs[0, 5].set_title('i ($\degree$)', fontsize=8)
axs[0, 5].tick_params(labelsize = 6)

axs[0, 6].hist(xoffs_stellar, bins=23, range=[-.2, .5])
axs[0, 6].axvline(xoffs_stellar_16, color='k', linestyle='solid', linewidth=1)
axs[0, 6].axvline(xoffs_stellar_50, color='k', linestyle='dashed', linewidth=1)
axs[0, 6].axvline(xoffs_stellar_84, color='k', linestyle='solid', linewidth=1)
axs[0, 6].set_title('$\Delta$ x (")', fontsize=8)
axs[0, 6].tick_params(labelsize = 6)

axs[1, 0].hist(yoffs_stellar, bins=23, range=[-.2, .2])
axs[1, 0].axvline(yoffs_stellar_16, color='k', linestyle='solid', linewidth=1)
axs[1, 0].axvline(yoffs_stellar_50, color='k', linestyle='dashed', linewidth=1)
axs[1, 0].axvline(yoffs_stellar_84, color='k', linestyle='solid', linewidth=1)
axs[1, 0].set_title('$\Delta$ y (")', fontsize=8)
axs[1, 0].tick_params(labelsize = 6)

axs[1, 1].hist(pp1, bins=23)
axs[1, 1].axvline(pp1_16, color='k', linestyle='solid', linewidth=1)
axs[1, 1].axvline(pp1_50, color='k', linestyle='dashed', linewidth=1)
axs[1, 1].axvline(pp1_84, color='k', linestyle='solid', linewidth=1)
axs[1, 1].set_title('pp1', fontsize=8)
axs[1, 1].tick_params(labelsize = 6)

axs[1, 2].hist(pp2, bins=23, range=[0, 5])
axs[1, 2].axvline(pp2_16, color='k', linestyle='solid', linewidth=1)
axs[1, 2].axvline(pp2_50, color='k', linestyle='dashed', linewidth=1)
axs[1, 2].axvline(pp2_84, color='k', linestyle='solid', linewidth=1)
axs[1, 2].set_title('pp2', fontsize=8)
axs[1, 2].tick_params(labelsize = 6)

axs[1, 3].hist(pp3, bins=23)
axs[1, 3].axvline(pp3_16, color='k', linestyle='solid', linewidth=1)
axs[1, 3].axvline(pp3_50, color='k', linestyle='dashed', linewidth=1)
axs[1, 3].axvline(pp3_84, color='k', linestyle='solid', linewidth=1)
axs[1, 3].set_title('pp3', fontsize=8)
axs[1, 3].tick_params(labelsize = 6)

axs[1, 4].hist(rt1, bins=23, range=[0, 100])
axs[1, 4].axvline(rt1_16, color='k', linestyle='solid', linewidth=1)
axs[1, 4].axvline(rt1_50, color='k', linestyle='dashed', linewidth=1)
axs[1, 4].axvline(rt1_84, color='k', linestyle='solid', linewidth=1)
axs[1, 4].set_title('rt1', fontsize=8)
axs[1, 4].tick_params(labelsize = 6)

axs[1, 5].hist(rt2, bins=23, range=[90, 150])
axs[1, 5].axvline(rt2_16, color='k', linestyle='solid', linewidth=1)
axs[1, 5].axvline(rt2_50, color='k', linestyle='dashed', linewidth=1)
axs[1, 5].axvline(rt2_84, color='k', linestyle='solid', linewidth=1)
axs[1, 5].set_title('rt2', fontsize=8)
axs[1, 5].tick_params(labelsize = 6)

#fig.delaxes(axs[2,3])
fig.delaxes(axs[1,6])


#Hide y labels for both rows
for ax in axs.flat:
    ax.set_yticklabels([])

fig.subplots_adjust(wspace=.5, hspace=.5)
fig.savefig("histograms_nogap1.pdf")

fig.set_figheight(3)
fig.set_figwidth(8)

fig.show()










#end of code:)
