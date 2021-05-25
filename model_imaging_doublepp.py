import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import sys
import corner
import csv
import pylab
import seaborn as sns
from astropy.io import fits
#from debris_disk import *
from debris_disk_doublepp import *
#from raytrace import *
from raytrace_gaussian import *
from single_model import *





# Doublepp No Gap
#lnprob = -10733005.6875
#r_in = 5.249450428292632
#delta_r = 192.15400247575505
#m_disk = -7.137948708508958
#f_star = 1.7241448387240548e-05
#position_angle = math.acos(0.4649525269769602) * (180/math.pi)
#inclination = math.acos(0.6924394509781522) * (180/math.pi)
#xoffs_stellar = 0.15855046472148154
#yoffs_stellar = 0.016805707216058202
#pp1 = -1.3320852383570128
#pp2 = 3.7499459103406
#rt = 129.37850659349738


#Doublepp with gap
"""
lnprob = -10732992.21875
r_in = 29.09438317749159
delta_r = 167.17900266462595
m_disk = -7.135314339911119
f_star = 1.572096033453631e-05
position_angle = math.acos(0.4988123262304179) * (180/math.pi)
inclination = math.acos(0.7176355599524268) * (180/math.pi)
xoffs_stellar = 0.14703509127231876
yoffs_stellar = 0.032275688403238495
r_in_gap = 65.58950830749322
delta_r_gap = 29.54099360510634
pp1 = -1.4274170607253005
pp2 = 2.9574139512650346
rt = 103.20802144976976
"""
lnprob = -10732991.8125
r_in = 20.668490469777275
delta_r = 176.14594548800298
m_disk = -7.130378795268787
f_star = 1.7497716433050714e-05
position_angle = math.acos(0.5021426711375954) * (180/math.pi)
inclination = math.acos(0.7057841514481599) * (180/math.pi)
xoffs_stellar = 0.1576850010357272
yoffs_stellar = 0.04095954828631752
r_in_gap = 67.13458163931847
delta_r_gap = 31.558572620411663
pp1 = -2.0400224352556924
pp2 = 2.8248306407905357
rt = 96.80570260643572


#total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test0',isgas=False, x_offset=-0.78832077557400004,y_offset=2.2526090775499998,stellar_flux=f_star)

#f_star=0 #TAKE THIS OUT LATER

#x = Disk(params=[-0.5,m_disk,1.,r_in,r_out,150.,inclination,2.3,1e-4,0.01,33.9,[.79,1000],[10.,1000], -1, 500, 500, 2.83, 0.1])
#x = Disk(params=[-0.5,m_disk,1.,r_in,r_out,150.,inclination,2.3,1e-4,0.01,33.9,[.79,1000],[10.,1000], -1, 500, 500, 2.83, 0.1])
#x = Disk(params=[-0.5, 10**(-7.2498684319000004, 1., 66.856559091600005, 93.779505323099997, 150., 44.615509906699998, 2.3, 1e-4, 0.01, 33.9, [.79,1000], [10.,1000], -1, 500, 500, 2.83, 0.1])
#x = Disk(params=[-0.5, 10**(-7.2314459346300008), 1., 12.116377465899999, 158.63347423100001, 150., 51.559876126300004, 2.3, 1e-4, 0.01, 33.9, [.79,1000], [10.,1000], -1, 500, 500, 2.83, 0.1]) #100 steps

#No Gap and Gap:
#x = Disk(params=[-0.5, 10**(m_disk), 1., r_in, r_in+delta_r, 150., inclination, 2.3, 1e-4, 0.01, 33.9, [.79,1000], [10.,1000], -1, 500, 500, 2.83, 0.1, 0.01])

#Double PP:
x = Disk(params=[-0.5, 10**(m_disk), pp1, r_in, r_in+delta_r, rt, inclination, 2.3, 1e-4, pp2, 33.9, [.79,1000], [10.,1000], -1, 500, 500, 2.83, 0.1, 0.01])

r_out_gap = r_in_gap + delta_r_gap
x.add_dust_gap(r_in_gap,r_out_gap)

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test0',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_0',modfile='doublepp_visabilities/raytrace_test0',isgas=True,freq0=345.79599)
print("*********0 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test1',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_1',modfile='doublepp_visabilities/raytrace_test1',isgas=True,freq0=345.79599)
print("*********1 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test2',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_0',modfile='doublepp_visabilities/raytrace_test2',isgas=True,freq0=345.79599)
print("*********2 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test3',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_1',modfile='doublepp_visabilities/raytrace_test3',isgas=True,freq0=345.79599)
print("*********3 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test4',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_2',modfile='doublepp_visabilities/raytrace_test4',isgas=True,freq0=345.79599)
print("*********4 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test5',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_3',modfile='doublepp_visabilities/raytrace_test5',isgas=True,freq0=345.79599)
print("*********5 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test6',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw0',modfile='doublepp_visabilities/raytrace_test6',isgas=True,freq0=345.79599)
print("*********6 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test7',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw1',modfile='doublepp_visabilities/raytrace_test7',isgas=True,freq0=345.79599)
print("*********7 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test8',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw2',modfile='doublepp_visabilities/raytrace_test8',isgas=True,freq0=345.79599)
print("*********8 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test9',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw3',modfile='doublepp_visabilities/raytrace_test9',isgas=True,freq0=345.79599)
print("*********9 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test10',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw4',modfile='doublepp_visabilities/raytrace_test10',isgas=True,freq0=345.79599)
print("*********10 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test11',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw5',modfile='doublepp_visabilities/raytrace_test11',isgas=True,freq0=345.79599)
print("*********11 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test12',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw0',modfile='doublepp_visabilities/raytrace_test12',isgas=True,freq0=345.79599)
print("*********12 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test13',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw1',modfile='doublepp_visabilities/raytrace_test13',isgas=True,freq0=345.79599)
print("*********13 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test14',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw2',modfile='doublepp_visabilities/raytrace_test14',isgas=True,freq0=345.79599)
print("*********14 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test15',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw3',modfile='doublepp_visabilities/raytrace_test15',isgas=True,freq0=345.79599)
print("*********15 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test16',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw4',modfile='doublepp_visabilities/raytrace_test16',isgas=True,freq0=345.79599)
print("*********16 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test17',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw5',modfile='doublepp_visabilities/raytrace_test17',isgas=True,freq0=345.79599)
print("*********17 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test18',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw6',modfile='doublepp_visabilities/raytrace_test18',isgas=True,freq0=345.79599)
print("*********18 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test19',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw7',modfile='doublepp_visabilities/raytrace_test19',isgas=True,freq0=345.79599)
print("*********19 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test20',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw8',modfile='doublepp_visabilities/raytrace_test20',isgas=True,freq0=345.79599)
print("*********20 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test21',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw9',modfile='doublepp_visabilities/raytrace_test21',isgas=True,freq0=345.79599)
print("*********21 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test22',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw10',modfile='doublepp_visabilities/raytrace_test22',isgas=True,freq0=345.79599)
print("*********22 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='doublepp_visabilities/raytrace_test23',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw11',modfile='doublepp_visabilities/raytrace_test23',isgas=True,freq0=345.79599)
print("*********23 through***********")
print("HERE -- 100% through")
