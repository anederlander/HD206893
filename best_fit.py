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
from raytrace import *
from single_model import *

#Parameters = ['r_in', 'delta_r', 'm_disk', 'f_star', 'position_angle', 'inclination', 'lnprob']

#df = pd.read_csv("chain_400steps_april20_nogap1.csv")
df = pd.read_csv("chain_400steps_june11_gap.csv")
#df = pd.read_csv("chain_400steps_10params_april20.csv")
max_lnprob = df['lnprob'].max()
max_r_in = df.r_in[df.lnprob.idxmax()]
max_delta_r = df.delta_r[df.lnprob.idxmax()]
max_m_disk = df.m_disk[df.lnprob.idxmax()]
max_f_star = df.f_star[df.lnprob.idxmax()]
max_position_angle = df.position_angle[df.lnprob.idxmax()]
max_inclination = df.inclination[df.lnprob.idxmax()]
#max_xoffs = df.xoffs[df.lnprob.idxmax()] #nogap
#max_yoffs = df.yoffs[df.lnprob.idxmax()] #nogap
max_xoffs_stellar = df.xoffs_stellar[df.lnprob.idxmax()] #gap
max_yoffs_stellar = df.yoffs_stellar[df.lnprob.idxmax()] #gap
max_r_in_gap = df.r_in_gap[df.lnprob.idxmax()] #gap
max_delta_r_gap = df.delta_r_gap[df.lnprob.idxmax()] #gap
max_pp = df.pp[df.lnprob.idxmax()] 

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
print("pp =", max_pp)
