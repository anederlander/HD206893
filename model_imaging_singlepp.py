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
from debris_disk import *
#from debris_disk_doublepp import *
#from raytrace import *
from raytrace_gaussian import *
from single_model import *

#Best-Fit Parameters:
#singlepp - no gap
lnprob = -10733011.5
r_in = 6.854834156713531
delta_r = 157.15804923020409
m_disk = -7.171904432773622
f_star = 1.686096748719349e-05
position_angle = 62.61526066592616
inclination = 47.01056975694721
xoffs_stellar = 0.1359439741853807
yoffs_stellar = -0.00987123322685522
pp = 1.0828929455303098


#singlepp with gap
#lnprob = -10733006.34375
#r_in = 19.21350285764236
#delta_r = 143.18416790116072
#m_disk = -7.187222014748311
#f_star = 1.0087670265720473e-05
#position_angle = 60.066169694008416
#inclination = 47.534369754309594
#xoffs_stellar = 0.07790674770959989
#yoffs_stellar = 0.015551739977984959
#r_in_gap = 68.56311833195011
#delta_r_gap = 19.427044048467266
#pp = 0.8008993774359241


r_out = r_in+delta_r

#total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test0',isgas=False, x_offset=-0.78832077557400004,y_offset=2.2526090775499998,stellar_flux=f_star)

#Uncomment when you want to look at raytrace image
#f_star=0 #TAKE THIS OUT LATER


#Doublepp:
#x = Disk(params=[-0.5, 10**(m_disk), pp1, r_in, r_in+delta_r, rt, inclination, 2.3, 1e-4, pp2, 33.9, [.79,1000], [10.,1000], -1, 500, 500, 2.83, 0.1, 0.01])

#Singlepp:
x = Disk(params=[-0.5,10**(m_disk),pp,r_in,r_out,150.,inclination,2.3,1e-4,0.01,33.9,[.79,1000],[10.,1000], -1, 500, 500, 2.83, 0.1])


r_out_gap = r_in_gap + delta_r_gap
x.add_dust_gap(r_in_gap,r_out_gap)

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test0',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_0',modfile='raytrace_test0',isgas=True,freq0=345.79599)
print("*********0 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test1',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_1',modfile='raytrace_test1',isgas=True,freq0=345.79599)
print("*********1 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test2',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_0',modfile='raytrace_test2',isgas=True,freq0=345.79599)
print("*********2 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test3',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_1',modfile='raytrace_test3',isgas=True,freq0=345.79599)
print("*********3 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test4',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_2',modfile='raytrace_test4',isgas=True,freq0=345.79599)
print("*********4 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test5',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_3',modfile='raytrace_test5',isgas=True,freq0=345.79599)
print("*********5 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test6',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw0',modfile='raytrace_test6',isgas=True,freq0=345.79599)
print("*********6 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test7',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw1',modfile='raytrace_test7',isgas=True,freq0=345.79599)
print("*********7 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test8',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw2',modfile='raytrace_test8',isgas=True,freq0=345.79599)
print("*********8 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test9',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw3',modfile='raytrace_test9',isgas=True,freq0=345.79599)
print("*********9 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test10',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw4',modfile='raytrace_test10',isgas=True,freq0=345.79599)
print("*********10 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test11',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw5',modfile='raytrace_test11',isgas=True,freq0=345.79599)
print("*********11 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test12',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw0',modfile='raytrace_test12',isgas=True,freq0=345.79599)
print("*********12 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test13',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw1',modfile='raytrace_test13',isgas=True,freq0=345.79599)
print("*********13 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test14',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw2',modfile='raytrace_test14',isgas=True,freq0=345.79599)
print("*********14 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test15',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw3',modfile='raytrace_test15',isgas=True,freq0=345.79599)
print("*********15 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test16',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw4',modfile='raytrace_test16',isgas=True,freq0=345.79599)
print("*********16 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test17',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw5',modfile='raytrace_test17',isgas=True,freq0=345.79599)
print("*********17 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test18',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw6',modfile='raytrace_test18',isgas=True,freq0=345.79599)
print("*********18 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test19',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw7',modfile='raytrace_test19',isgas=True,freq0=345.79599)
print("*********19 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test20',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw8',modfile='raytrace_test20',isgas=True,freq0=345.79599)
print("*********20 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test21',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw9',modfile='raytrace_test21',isgas=True,freq0=345.79599)
print("*********21 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test22',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw10',modfile='raytrace_test22',isgas=True,freq0=345.79599)
print("*********22 through***********")

total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test23',isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw11',modfile='raytrace_test23',isgas=True,freq0=345.79599)
print("*********23 through***********")
print("HERE -- 100% through")
