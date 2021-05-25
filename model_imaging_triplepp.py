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
from raytrace_gaussian import *
from single_model import *

#Best Fit Parameters: July 17, 2020
lnprob= -10733005.0
r_in= 27.22621973417132
delta_r= 175.02825107186234
m_disk= -7.285710984613719
f_star= 6.011120093454203e-06
position_angle= 0.4824387042970495
inclination= 0.6468896503009111
xoffs_stellar= 0.17793368373323304
yoffs_stellar= 0.06789809918272766
pp1= -2.8605355788033897
pp2= 2.9735030991951863
pp3= -2.6684509847007254
rt1= 57.12753468246092
rt2= 119.2653934996326


#total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile='raytrace_test0',isgas=False, x_offset=-0.78832077557400004,y_offset=2.2526090775499998,stellar_flux=f_star)

f_star=0 #TAKE THIS OUT LATER

#x = Disk(params=[-0.5,m_disk,1.,r_in,r_out,150.,inclination,2.3,1e-4,0.01,33.9,[.79,1000],[10.,1000], -1, 500, 500, 2.83, 0.1])
#x = Disk(params=[-0.5,m_disk,1.,r_in,r_out,150.,inclination,2.3,1e-4,0.01,33.9,[.79,1000],[10.,1000], -1, 500, 500, 2.83, 0.1])
#x = Disk(params=[-0.5, 10**(-7.2498684319000004, 1., 66.856559091600005, 93.779505323099997, 150., 44.615509906699998, 2.3, 1e-4, 0.01, 33.9, [.79,1000], [10.,1000], -1, 500, 500, 2.83, 0.1])
#x = Disk(params=[-0.5, 10**(-7.2314459346300008), 1., 12.116377465899999, 158.63347423100001, 150., 51.559876126300004, 2.3, 1e-4, 0.01, 33.9, [.79,1000], [10.,1000], -1, 500, 500, 2.83, 0.1]) #100 steps
x = Disk(params=[-0.5, 10**(m_disk), 1., r_in, r_in+delta_r, 150., inclination, 2.3, 1e-4, 0.01, 33.9, [.79,1000], [10.,1000], -1, 500, 500, 2.83, 0.1])
#r_out_gap = r_in_gap + delta_r_gap
#x.add_dust_gap(r_in_gap,r_out_gap)

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
