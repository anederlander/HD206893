#!/bin/bash
rm -rf residual_oct31.*
rm -rf raytrace_test*
cp -r /Volumes/disks/ava/hd206893/disk_model/disk_model3_new/raytrace_test* /Volumes/disks/ava/hd206893/
#fits in=raytrace_test0.model.vis.fits op=uvin options=varwt out=raytrace_test0.model.vis
#fits in=raytrace_test1.model.vis.fits op=uvin options=varwt out=raytrace_test1.model.vis
#fits in=raytrace_test2.model.vis.fits op=uvin options=varwt out=raytrace_test2.model.vis
#fits in=raytrace_test3.model.vis.fits op=uvin options=varwt out=raytrace_test3.model.vis
#fits in=raytrace_test4.model.vis.fits op=uvin options=varwt out=raytrace_test4.model.vis
#fits in=raytrace_test5.model.vis.fits op=uvin options=varwt out=raytrace_test5.model.vis
#fits in=raytrace_test6.model.vis.fits op=uvin options=varwt out=raytrace_test6.model.vis
#fits in=raytrace_test7.model.vis.fits op=uvin options=varwt out=raytrace_test7.model.vis
#fits in=raytrace_test8.model.vis.fits op=uvin options=varwt out=raytrace_test8.model.vis
#fits in=raytrace_test9.model.vis.fits op=uvin options=varwt out=raytrace_test9.model.vis
#fits in=raytrace_test10.model.vis.fits op=uvin options=varwt out=raytrace_test10.model.vis
#fits in=raytrace_test11.model.vis.fits op=uvin options=varwt out=raytrace_test11.model.vis
#fits in=raytrace_test12.model.vis.fits op=uvin options=varwt out=raytrace_test12.model.vis
#fits in=raytrace_test13.model.vis.fits op=uvin options=varwt out=raytrace_test13.model.vis
#fits in=raytrace_test14.model.vis.fits op=uvin options=varwt out=raytrace_test14.model.vis
#fits in=raytrace_test15.model.vis.fits op=uvin options=varwt out=raytrace_test15.model.vis
#fits in=raytrace_test16.model.vis.fits op=uvin options=varwt out=raytrace_test16.model.vis
#fits in=raytrace_test17.model.vis.fits op=uvin options=varwt out=raytrace_test17.model.vis
#fits in=raytrace_test18.model.vis.fits op=uvin options=varwt out=raytrace_test18.model.vis
#fits in=raytrace_test19.model.vis.fits op=uvin options=varwt out=raytrace_test19.model.vis
#fits in=raytrace_test20.model.vis.fits op=uvin options=varwt out=raytrace_test20.model.vis
#fits in=raytrace_test21.model.vis.fits op=uvin options=varwt out=raytrace_test21.model.vis
#fits in=raytrace_test22.model.vis.fits op=uvin options=varwt out=raytrace_test22.model.vis
#fits in=raytrace_test23.model.vis.fits op=uvin options=varwt out=raytrace_test23.model.vis
invert vis=raytrace_test5.model.vis,raytrace_test6.model.vis,raytrace_test7.model.vis,raytrace_test8.model.vis,raytrace_test9.model.vis,raytrace_test1.model.vis,raytrace_test2.model.vis,raytrace_test3.model.vis,raytrace_test4.model.vis,raytrace_test10.model.vis,raytrace_test11.model.vis,raytrace_test12.model.vis,raytrace_test13.model.vis,raytrace_test14.model.vis,raytrace_test15.model.vis,raytrace_test16.model.vis,raytrace_test17.model.vis,raytrace_test18.model.vis,raytrace_test19.model.vis,raytrace_test20.model.vis,raytrace_test21.model.vis,raytrace_test22.model.vis,raytrace_test23.model.vis map=residual_oct31.mp beam=residual_oct31.bm robust=2 options=mfs,systemp imsize=240 fwhm=0.5 cell=0.05
clean map=residual_oct31.mp beam=residual_oct31.bm out=residual_oct31.cl niters=300
restor map=residual_oct31.mp beam=residual_oct31.bm model=residual_oct31.cl out=residual_oct31.cm
fits in=residual_oct31.cm op=xyout out=residual_oct31.fits


