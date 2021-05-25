#Ava Nederlander MCMC code
#Summer 2019
#mpirun -np=5 nice python rev_mcmcmodel2.py
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
from debris_disk_doublepp import *
from raytrace_gaussian import *
from single_model import *
from multiprocessing import Pool
import os
import shutil

#from ava_disk import *
#rin=1.
#rout=100.
#x_data_1=(np.genfromtxt('hoggs_data.txt',usecols=(1,))), y_data_1=(np.genfromtxt('hoggs_data.txt',usecols=(2,))), sigma_data_1=(np.genfromtxt('hoggs_data.txt',usecols=(4,)))
import time
start=time.time()
'''from emcee.utils import MPIPool
pool = MPIPool()

# Tell the difference between master and worker processes
if not pool.is_master():
    pool.wait()
    sys.exit(0)'''

from emcee.utils import MPIPool
pool = MPIPool()




def chiSq(modelfiles_path, datafiles_path):
    """Calculate the goodness of fit between data and model."""
    data_uvf = fits.open(datafiles_path + '.uvfits')
    data_vis = data_uvf[0].data['data'].squeeze()

    model = fits.open(modelfiles_path + '.vis.fits')
    model_vis = model[0].data['data'].squeeze()

    data_real = (data_vis[:, :, 0, 0] + data_vis[:, :, 1, 0])/2.
    data_imag = (data_vis[:, :, 0, 1] + data_vis[:, :, 1, 1])/2.

    model_real = model_vis[::2, :, 0]
    model_imag = model_vis[::2, :, 1]

    wt = data_vis[:, :, 0, 2]

    raw_chi = np.sum(wt * (data_real - model_real)**2 +
                     wt * (data_imag - model_imag)**2)

    raw_chi = raw_chi
    print("Raw Chi2: ", raw_chi)
    return raw_chi * -0.5

def lnprob(p1):
    r_in, delta_r, log_m_disk, f_star, cos_position_angle, cos_inclination, xoffs_stellar, yoffs_stellar, pp1, rt, pp2, r_in_gap, delta_r_gap = p1
    priors_r_in = [0, 10000]
    priors_delta_r = [0, 10000]
    priors_m_disk = [-10, -2]
    priors_f_star = [0, 100000000]
    priors_cos_position_angle = [-1, 1]
    priors_cos_inclination = [-1, 1]
    priors_xoffs_stellar = [-5, 5] #star
    priors_yoffs_stellar = [-5, 5] #star
    r_out = r_in + delta_r
    priors_pp1 = [-5, 5] 
    priors_rt = [r_in, r_out]
    priors_pp2 = [-5, 5]
    priors_r_in_gap = [r_in, r_out] #inner radius for gap
    priors_delta_r_gap = [0, delta_r] #delta radius for gap

    if r_in < priors_r_in[0] or r_in > priors_r_in[1]:
        return -np.inf
        print("rin out of bounds")

    if delta_r < priors_delta_r[0] or delta_r > priors_delta_r[1]:
        return -np.inf
        print("delta_r out of bounds")

    if log_m_disk < priors_m_disk[0] or log_m_disk > priors_m_disk[1]:
        return -np.inf
        print("mdisk out of bounds")

    if f_star < priors_f_star[0] or f_star > priors_f_star[1]:
        return -np.inf
        print("fstar out of bounds")

    if cos_position_angle < priors_cos_position_angle[0] or cos_position_angle > priors_cos_position_angle[1]:
        return -np.inf
        print("position_angle out of bounds")

    if cos_inclination < priors_cos_inclination[0] or cos_inclination > priors_cos_inclination[1]:
        return -np.inf
        print("inclination out of bounds")

    if xoffs_stellar < priors_xoffs_stellar[0] or xoffs_stellar > priors_xoffs_stellar[1]:
        return -np.inf
        print("stellar x offset out of bounds")

    if yoffs_stellar < priors_yoffs_stellar[0] or yoffs_stellar > priors_yoffs_stellar[1]:
        return -np.inf
        print("stellar y offset out of bounds")

    if pp1 < priors_pp1[0] or pp1 > priors_pp1[1]:
        return -np.inf
        print("power index  out of bounds")
    '''if xoffs_disk < priors_xoffs_disk[0] or xoffs_disk > priors_xoffs_disk[1]:
        return -np.inf
        print("disk x offset out of bounds")

    if yoffs_disk < priors_yoffs_disk[0] or yoffs_disk > priors_yoffs_disk[1]:
        return -np.inf
        print("disk y offset out of bounds")'''

    if rt < priors_rt[0] or rt > priors_rt[1]:
        return -np.inf
        print("disk x offset out of bounds")

    if pp2 < priors_pp2[0] or pp2 > priors_pp2[1]:
        return -np.inf
        print("disk y offset out of bounds")


    if r_in_gap < priors_r_in_gap[0] or r_in_gap > priors_r_in_gap[1]:
        return -np.inf
        print("disk x offset out of bounds")

    if delta_r_gap < priors_delta_r_gap[0] or delta_r_gap > priors_delta_r_gap[1]:
        return -np.inf
        print("disk y offset out of bounds")

    m_disk = 10**log_m_disk

    inclination = math.acos(cos_inclination) * (180/math.pi)
    position_angle = math.acos(cos_position_angle) * (180/math.pi)

    x = Disk(params=[-0.5,m_disk,pp1,r_in,r_out,rt,inclination,2.3,1e-4,pp2,33.9,[.79,1000],[10.,1000], -1, 500, 500, 2.83, 0.1,0.01])
    r_out_gap = r_in_gap + delta_r_gap
    x.add_dust_gap(r_in_gap,r_out_gap)

    unique_id = str(np.random.randint(1e10))
    model_name = 'model_' + unique_id
    model_file_name = model_name + '.model'

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_0',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq0 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_0')
    print("*********0 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_1',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq1 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_1')
    print("*********1 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_0',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq2 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_0')
    print("*********2 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_1',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq3 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_1')
    print("*********3 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_2',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq4 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_2')
    print("*********4 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_3',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq5 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_3')
    print("*********5 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw0',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq6 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw0')
    print("*********6 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw1',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq7 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw1')
    print("*********7 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw2',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq8 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw2')
    print("*********8 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw3',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq9 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw3')
    print("*********9 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw4',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq10 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw4')
    print("*********10 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw5',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq11 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw5')
    print("*********11 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw0',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq12 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw0')
    print("*********12 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw1',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq13 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw1')
    print("*********13 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw2',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq14 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw2')
    print("*********14 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw3',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq15 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw3')
    print("*********15 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw4',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq16 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw4')
    print("*********16 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw5',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq17 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw5')
    print("*********17 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw6',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq18 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw6')
    print("*********18 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw7',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq19 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw7')
    print("*********19 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw8',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq20 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw8')
    print("*********20 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw9',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq21 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw9')
    print("*********21 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw10',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq22 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw10')
    print("*********22 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs_stellar, yoffs_stellar],modfile=model_name,isgas=False, x_offset=xoffs_stellar, y_offset=yoffs_stellar, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw11',modfile=model_name,isgas=True,freq0=345.79599)
    chiSq23 = chiSq(model_file_name, '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw11')
    #make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_0',modfile='raytrace_test2',isgas=True,freq0=345.79599)
    print("*********23 through***********")
    print("HERE -- 100% through")


   
    #deleting the randomly generated files
    shutil.rmtree(model_file_name+'.vis')
    os.remove(model_file_name+'.vis.fits')
    os.remove(model_name+'.fits')




    return chiSq0+chiSq1+chiSq2+chiSq3+chiSq4+chiSq5+chiSq6+chiSq7+chiSq8+chiSq9+chiSq10+chiSq11+chiSq12+chiSq13+chiSq14+chiSq15+chiSq16+chiSq17+chiSq18+chiSq19+chiSq20+chiSq21+chiSq22+chiSq23


def emcee(nsteps=800, ndim=13, nwalkers=30, walker_1=10.0, walker_2=150.0, walker_3=-6, walker_4=1.65e-5, walker_5=0.47, walker_6=0.68, walker_7=.2, walker_8=.8, walker_9=0, walker_10=70, walker_11=0, walker_12=40, walker_13=40, sigma_1=30.0, sigma_2=30.0, sigma_3=2, sigma_4=8e-6, sigma_5=0.1, sigma_6=0.1, sigma_7=0.5, sigma_8=0.5, sigma_9=2, sigma_10=20, sigma_11=2, sigma_12=10, sigma_13=10, restart=False):
    '''Perform MCMC affine invariants
    :param nsteps:              The number of iterations
    :param ndim:                number of dimensions
    :param nwalkers:            number of walkers
    :param walker_1:            the first parameter for the 1st dimension - r_in
    :param walker_2:            the first parameter for the 2nd dimension - delta_r
    :param walker_3:            the first parameter for the 3rd dimension - log_m_disk
    :param walker_4:            the first parameter for the 4th dimension - f_star
    :param walker_5:            the first parameter for the 5th dimension - position_angle
    :param walker_6:            the first parameter for the 6th dimension - inclination
    :param walker_7:            the first parameter for the 7th dimension - xoffs for stellar
    :param walker_8:            the first parameter for the 8th dimension - yoffs for stellar
    :param walker_9:            the first parameter for the 9th dimension - pp1
    :param walker_10:           the first parameter for the 10th dimension - rt
    :param walker_11:            the first parameter for the 11th dimension - pp2
    :param walker_12:            the first parameter for the 12th dimension - r_in for gap
    :param walker_13:           the first parameter for the 13th dimension - delta_r for gap
    :param sigma_1:             sigma for walker_1
    :param sigma_2:             sigma for walker_2
    :param sigma_3:             sigma for walker_3
    :param sigma_4:             sigma for walker_4
    :param sigma_5:             sigma for walker_5
    :param sigma_6:             sigma for walker_6
    :param sigma_7:             sigma for walker_7
    :param sigma_8:             sigma for walker_8
    :param sigma_9:             sigma for walker_9
    :param sigma_10:            sigma for walker_10
    '''
    #r_out = r_in + delta_r
    '''walker_1_array = [walker_1]
    walker_2_array = [walker_2]
    walker_3_array = [walker_3]
    walker_4_array = [walker_4]
    walker_5_array = [walker_5]
    walker_6_array = [walker_6]
    p0 = [walker_1, walker_2, walker_3, walker_4, walker_5, walker_6]'''
    #chi_array = [np.sum(((y_data_1) - (walker_1_array*x_data_1+walker_2_array))**2/sigma_data_1**2)]
    if restart == False:
        p0 = np.random.normal(loc=(walker_1, walker_2, walker_3, walker_4, walker_5, walker_6, walker_7, walker_8, walker_9, walker_10, walker_11, walker_12, walker_13), size=(nwalkers, ndim), scale=(sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7, sigma_8, sigma_9, sigma_10, sigma_11, sigma_12, sigma_13))
    else:
        #read from csv file
        dg = pd.read_csv("chain_800steps_july17_doublepp_gap.csv")
        p0 = np.zeros([nwalkers,ndim])
        for i in range(nwalkers):
            p0[i,0] = dg['r_in'].iloc[-(nwalkers-i+1)]
            p0[i,1] = dg['delta_r'].iloc[-(nwalkers-i+1)]-p0[i,0] #future versions should be delta_r
            p0[i,2] = dg['m_disk'].iloc[-(nwalkers-i+1)]
            p0[i,3] = dg['f_star'].iloc[-(nwalkers-i+1)]
            p0[i,4] = dg['cos_position_angle'].iloc[-(nwalkers-i+1)]
            p0[i,5] = dg['cos_inclination'].iloc[-(nwalkers-i+1)]
            p0[i,6] = dg['xoffs_stellar'].iloc[-(nwalkers-i+1)]
            p0[i,7] = dg['yoffs_stellar'].iloc[-(nwalkers-i+1)]
            p0[i,8] = dg['pp1'].iloc[-(nwalkers-i+1)]
            p0[i,9] = dg['rt'].iloc[-(nwalkers-i+1)]
            p0[i,10] = dg['pp2'].iloc[-(nwalkers-i+1)]
            p0[i,11] = dg['r_in_gap'].iloc[-(nwalkers-i+1)]
            p0[i,12] = dg['delta_r_gap'].iloc[-(nwalkers-i+1)]
        #p0 = (loc=(walker_1, walker_2, walker_3, walker_4, walker_5, walker_6), size=(nwalkers, ndim))
    import emcee
    # Tell the difference between master and worker processes
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool) #threads=10, a=4.0
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    run = sampler.sample(p0, iterations=nsteps, storechain=True)
    steps = []
    for i, result in enumerate(run):
        pos, lnprobs, blob = result

        new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
        steps += new_step
        #print(pos)
        print(lnprobs)
        df = pd.DataFrame(steps)
        df.columns = ['r_in', 'delta_r', 'm_disk', 'f_star', 'cos_position_angle', 'cos_inclination', 'xoffs_stellar', 'yoffs_stellar', 'pp1', 'rt', 'pp2', 'r_in_gap', 'delta_r_gap', 'lnprob']
        df.to_csv('chain_800steps_july17_doublepp_gap.csv', mode='w')
        sys.stdout.write("Completed step {} of {}  \r".format(i, nsteps) )
        sys.stdout.flush()

    print(np.shape(sampler.chain))
    print("Finished MCMC.")
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))


    cmap_light = sns.diverging_palette(220, 20, center='dark', n=nwalkers)
    fig, ax = plt.subplots()
    for i in range(nwalkers):
        ax.plot(df['r_in'][i::nwalkers], df['delta_r'][i::nwalkers], linestyle='-', marker='.', alpha=0.5)
        plt.show(block=False)

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
        ax9.plot(x, df['rt'][i::nwalkers])
        ax10.plot(x, df['pp2'][i::nwalkers])
        ax11.plot(x, df['r_in_gap'][i::nwalkers])
        ax12.plot(x, df['delta_r_gap'][i::nwalkers])
        ax13.plot(x, df['lnprob'][i::nwalkers])
        fig.suptitle('r_in, delta_r, m_disk, f_star, cos_position_angle, cos_inclination, xoffs_stellar, yoffs_stellar, pp1, rt, pp2, r_in_gap, delta_r_gap, lnprob')
        plt.show(block=False)

        print(np.shape(x),np.shape(w1m))

emcee()

print ('Elapsed time (hrs): ',(time.time()-start)/3600.)
'''
max_lnprob = df['lnprob'].max()
max_m = df.m[df.lnprob.idxmax()]
max_b = df.b[df.lnprob.idxmax()]
'''

#finish
