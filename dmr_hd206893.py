import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import math
from matplotlib import *




plt.clf()
plt.rcParams.update({'font.size': 25})

##opening the .fits file to see what it contains
##With gap
#hdulist = fits.open('/Volumes/disks/ava/hd206893/data_image2.fits') #data image
#hdulist2 = fits.open('/Volumes/disks/ava/hd206893/data_image2.fits') #datafile
#hdulist3 = fits.open('/Volumes/disks/ava/hd206893/gap_model.fits') #model image
#hdulist4 = fits.open('/Volumes/disks/ava/hd206893/gap_residual.fits') #residual image
#hdulist5 = fits.open('/Volumes/disks/ava/hd206893/raytrace_gap3.fits')
##Without gap
#hdulist = fits.open('/Volumes/disks/ava/hd206893/data_image2.fits')
#hdulist2 = fits.open('/Volumes/disks/ava/hd206893/data_image2.fits') #datafile
#hdulist3 = fits.open('/Volumes/disks/ava/hd206893/gapno_model.fits')
#hdulist4 = fits.open('/Volumes/disks/ava/hd206893/gapno_residual.fits')
#hdulist5 = fits.open('/Volumes/disks/ava/hd206893/raytrace_gapno3.fits')


#flat - no gap
#hdulist = fits.open('data_image2.fits') #data image
#hdulist2 = fits.open('data_image2.fits') #data image
#hdulist3 = fits.open('model_nogap.fits') #model image
#hdulist4 = fits.open('residual_nogap.fits') #residual image
#hdulist5 = fits.open('raytrace_nogap.fits') #raytrace image


#flat - gap
#hdulist = fits.open('data_image2.fits') #data image
#hdulist2 = fits.open('data_image2.fits') #data image
#hdulist3 = fits.open('model_gap.fits') #model image
#hdulist4 = fits.open('residual_gap.fits') #residual image
#hdulist5 = fits.open('raytrace_gap.fits') #raytrace image


#single pp - no gap
#hdulist = fits.open('data_image2.fits') #data image
#hdulist2 = fits.open('data_image2.fits') #data image
#hdulist3 = fits.open('model_singlepp_nogap.fits') #model image
#hdulist4 = fits.open('residual_singlepp_nogap.fits') #residual image
#hdulist5 = fits.open('raytrace_singlepp_nogap.fits') #raytrace image

#single pp - gap
#hdulist = fits.open('data_image2.fits') #data image
#hdulist2 = fits.open('data_image2.fits') #data image
#hdulist3 = fits.open('model_singlepp_gap.fits') #model image
#hdulist4 = fits.open('residual_singlepp_gap.fits') #residual image
#hdulist5 = fits.open('raytrace_singlepp_gap.fits') #raytrace image

#double pp - no gap
#hdulist = fits.open('data_image2.fits') #data image
#hdulist2 = fits.open('data_image2.fits') #data image
#hdulist3 = fits.open('model_doublepp_nogap.fits') #model image
#hdulist4 = fits.open('residual_doublepp_nogap.fits') #residual image
#hdulist5 = fits.open('raytrace_doublepp_nogap.fits') #raytrace image

#double pp - gap
hdulist = fits.open('data_image2.fits') #data image
hdulist2 = fits.open('data_image2.fits') #data image
hdulist3 = fits.open('model_doublepp_gap.fits') #model image
hdulist4 = fits.open('residual_doublepp_gap.fits') #residual image
hdulist5 = fits.open('raytrace_doublepp_gap.fits') #raytrace image


#triple pp - no gap
#hdulist = fits.open('data_image2.fits') #data image
#hdulist2 = fits.open('data_image2.fits') #data image
#hdulist3 = fits.open('model_triplepp_nogap1.fits') #model image
#hdulist4 = fits.open('residual_triplepp_nogap1.fits') #residual image
#hdulist5 = fits.open('raytrace_triplepp_nogap1.fits') #raytrace image
hdulist.info()
hdulist2.info()
hdulist3.info()
hdulist4.info()
hdulist5.info()

rms=6.003273e-06


##defining so I can look at the header
hdu = hdulist[0]
print(hdu.header)
##taking off the un-needed dimensions in the nparray
image_data = np.squeeze(hdulist[0].data)*1e6
image_data2 = np.squeeze(hdulist2[0].data)*1e6
image_data3 = np.squeeze(hdulist3[0].data)*1e6
image_data4 = np.squeeze(hdulist4[0].data)*1e6
image_data5 = np.squeeze(hdulist5[0].data)*1e6
##closing because we got everything we need
hdulist.close()
image=image_data2
#checking max and min
print("min",np.amax(image_data))
print("max", np.min(image_data))

print(np.amax(image_data5))
print(np.min(image_data5))

##defining stuff
crval1=hdu.header['CRVAL1']*3600
crval2=hdu.header['CRVAL2']*3600
crdelt1=hdu.header['CDELT1']*3600
crdelt2=hdu.header['CDELT2']*3600
crpix1=hdu.header['CRPIX1']
crpix2=hdu.header['CRPIX2']
pix1=hdu.header['NAXIS1']
pix2=hdu.header['NAXIS2']
pixs1=pix1*crdelt1
pixs2=pix2*crdelt2
ra_values=crdelt1*(np.arange(pix1)-crpix1+1)
dec_values=crdelt2*(np.arange(pix2)-crpix2+1)
font_size=15
xoff=4.5
yoff=-4.5
t=np.arange(0.0,2.0 *math.pi,0.05)
x=hdu.header['bmaj']*1800.0*np.cos(t)*np.cos((hdu.header['bpa']+90.0)*math.pi/180.0)-1800.0*hdu.header['bmin']*np.sin(t)*np.sin((hdu.header['bpa']+90.0)*math.pi/180.0)+xoff
y=1800.0*hdu.header['bmaj']*np.cos(t)*np.sin((hdu.header['bpa']+90.0)*math.pi/180.0)+1800.0*hdu.header['bmin']*np.sin(t)*np.cos((hdu.header['bpa']+90.0)*math.pi/180.0)+yoff

X,Y = np.meshgrid(ra_values,dec_values)
#fig, axs = plt.subplots(1, 4)
f1 = plt.figure(figsize=(17.0,4.0))
gs = gridspec.GridSpec(1, 5, width_ratios=[3.0,3.0,3.0,3.0,0.1])

##Data Image
ax1 = plt.subplot(gs[0])
figure = ax1.imshow(image_data, vmin=np.amin(image_data2), vmax=np.amax(image_data2),extent=[-(.5*pixs1),(.5*pixs1),-(.5*pixs2),(.5*pixs2)],cmap='afmhot',origin ='lower')
ax1.imshow(image_data, vmin=np.amin(image_data2), vmax=np.amax(image_data2),extent=[-(.5*pixs1),(.5*pixs1),-(.5*pixs2),(.5*pixs2)],cmap='afmhot',origin ='lower')
ax1.contour(X,Y,image_data/1e6,levels=np.array([2,4,6,8,10])*rms, colors='purple')
ax1.contour(X,Y,image_data/1e6,levels=np.array([-10, -8, -6, -4,-2])*rms,linestyles='dashed',colors='white')
ax1.tick_params(axis='both', which='major', labelsize=10)
ax1.tick_params(axis='both', which='minor', labelsize=8)
#ax1.set_fontsize(font_size)
ax1.set_xlabel(r'$\Delta \alpha$'   '["]',fontsize=font_size)
ax1.set_ylabel(r'$\Delta \delta$'  '["]',fontsize=font_size)
ax1.annotate('HD 206893',xy=(5, 5), xycoords='data',xytext=(-5, 5), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
ax1.annotate('ALMA $\lambda$'"=1348" r'$\mu$''m',xy=(5, 4), xycoords='data',xytext=(-5, 5), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
ax1.annotate('100 au',xy=(-3, -5.2), xycoords='data',xytext=(-30, 30), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
ax1.fill(x,y,fill=False,hatch='//////',color='white')
ax1.plot([-.7, -5], [-5,-5], color="white")


##Raytrace  Image
ax2 = plt.subplot(gs[1])
#axs[1].imshow((image_data5[0,:,:]), vmin=np.amin(image_data5), vmax=np.amax(image_data5),extent=[-(.5*pixs1),(.5*pixs1),-(.5*pixs2),(.5*pixs2)],cmap='afmhot',origin ='lower')
ax2.imshow(image_data5[0,:,:], cmap='afmhot',extent=[-(.5*pixs1),(.5*pixs1),-(.5*pixs2),(.5*pixs2)], origin ='lower')
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='minor', labelsize=8)

#axs[1].set_xticks([6, 4, 2, 0, -2, -4, -6])
#axs[1].set_yticks([6, 4, 2, 0, -2, -4, -6])
#axs[1].contour(X,Y,image_data3/1e6,levels=np.array([2,4,6,8,10])*rms, colors='purple')
#axs[1].contour(X,Y,image_data3/1e6,levels=np.array([-10, -8, -6, -4,-2])*rms,linestyles='dashed',colors='white')
ax2.set_xlabel(r'$\Delta \alpha$'   '["]',fontsize=font_size)
ax2.annotate('Model \nFull Resolution',xy=(5, 5), xycoords='data',xytext=(-5, 5), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
ax2.annotate('100 au',xy=(-3, -5.2), xycoords='data',xytext=(-30, 30), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
#axs[3].fill(x,y,fill=False,hatch='//////',color='white')
ax2.plot([-.7, -5], [-5,-5], color="white")


##Model Image
ax3 = plt.subplot(gs[2])
ax3.imshow(image_data3, vmin=np.amin(image_data2), vmax=np.amax(image_data2),extent=[-(.5*pixs1),(.5*pixs1),-(.5*pixs2),(.5*pixs2)],cmap='afmhot',origin ='lower')
ax3.tick_params(axis='both', which='major', labelsize=10)
ax3.tick_params(axis='both', which='minor', labelsize=8)

#axs[2].set_xticks([6, 4, 2, 0, -2, -4, -6])
#axs[2].set_yticks([6, 4, 2, 0, -2, -4, -6])
ax3.contour(X,Y,image_data3/1e6,levels=np.array([2,4,6,8,10])*rms, colors='purple')
ax3.contour(X,Y,image_data3/1e6,levels=np.array([-10, -8, -6, -4,-2])*rms,linestyles='dashed',colors='white')
ax3.set_xlabel(r'$\Delta \alpha$'   '["]',fontsize=font_size)
ax3.annotate('Model \nALMA Resolution',xy=(5, 5), xycoords='data',xytext=(-5, 5), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
ax3.annotate('100 au',xy=(-3, -5.2), xycoords='data',xytext=(-30, 30), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
ax3.fill(x,y,fill=False,hatch='//////',color='white')
ax3.plot([-.7, -5], [-5,-5], color="white")

##Residual Image
ax4 = plt.subplot(gs[3])
ax4.imshow(image_data4, vmin=np.amin(image_data2), vmax=np.amax(image_data2),extent=[-(.5*pixs1),(.5*pixs1),-(.5*pixs2),(.5*pixs2)],cmap='afmhot',origin ='lower')
ax4.tick_params(axis='both', which='major', labelsize=10)
ax4.tick_params(axis='both', which='minor', labelsize=8)

#cbar=axs[2].colorbar()
#cbar.set_label( r'$\mu$Jy / beam',fontsize=25)
ax4.contour(X,Y,image_data4/1e6,levels=np.array([2,4,6,8,10])*rms, colors='purple')
ax4.contour(X,Y,image_data4/1e6,levels=np.array([-10, -8, -6, -4,-2])*rms,linestyles='dashed',colors='white')
ax4.set_xlabel(r'$\Delta \alpha$'   '["]',fontsize=font_size)
ax4.annotate('Residual',xy=(5, 5), xycoords='data',xytext=(-5, 5), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
ax4.annotate('100 au',xy=(-3, -5.2), xycoords='data',xytext=(-30, 30), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=15)
ax4.fill(x,y,fill=False,hatch='//////',color='white')
ax4.plot([-.7, -5], [-5,-5], color="white")



ax15 = plt.subplot(gs[4])
cbar = plt.colorbar(figure, cax=ax15, orientation='vertical', label=r'$\mu$Jy/beam')
#plt.set_label(fontsize=15)
#cbar.tick_params(labelsize=15)
ax15.tick_params(axis='both', which='major', labelsize=10)
ax15.tick_params(axis='both', which='minor', labelsize=8)
cbar.set_label(r'$\mu$Jy/beam', fontsize=font_size)
plt.show()
plt.savefig('dmr_doublepp.pdf')
#print()
