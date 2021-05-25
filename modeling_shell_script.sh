#!/bin/bash
rm -rf model_jan24.*
rm -rf raytrace_test*
cp -r /Volumes/disks/ava/hd206893/disk_model/disk_model3_new/doublepp_visabilities/raytrace_test* /Volumes/disks/ava/hd206893/.
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
#invert vis=raytrace_test5.model.vis,raytrace_test6.model.vis,raytrace_test7.model.vis,raytrace_test8.model.vis,raytrace_test9.model.vis,raytrace_test1.model.vis,raytrace_test2.model.vis,raytrace_test3.model.vis,raytrace_test4.model.vis,raytrace_test10.model.vis,raytrace_test11.model.vis,raytrace_test12.model.vis,raytrace_test13.model.vis,raytrace_test14.model.vis,raytrace_test15.model.vis,raytrace_test16.model.vis,raytrace_test17.model.vis,raytrace_test18.model.vis,raytrace_test19.model.vis,raytrace_test20.model.vis,raytrace_test21.model.vis,raytrace_test22.model.vis,raytrace_test23.model.vis map=model_jan24.mp beam=model_jan24.bm robust=2 options=mfs,systemp imsize=240 cell=0.05
invert vis=raytrace_test5.model.vis,raytrace_test6.model.vis,raytrace_test7.model.vis,raytrace_test8.model.vis,raytrace_test9.model.vis,raytrace_test1.model.vis,raytrace_test2.model.vis,raytrace_test3.model.vis,raytrace_test4.model.vis,raytrace_test10.model.vis,raytrace_test11.model.vis,raytrace_test12.model.vis,raytrace_test13.model.vis,raytrace_test14.model.vis,raytrace_test15.model.vis,raytrace_test16.model.vis,raytrace_test17.model.vis,raytrace_test18.model.vis,raytrace_test19.model.vis,raytrace_test20.model.vis,raytrace_test21.model.vis,raytrace_test22.model.vis,raytrace_test23.model.vis map=model_jan24.mp beam=model_jan24.bm robust=2 options=mfs,systemp imsize=240 fwhm=0.5 cell=0.05
clean map=model_jan24.mp beam=model_jan24.bm out=model_jan24.cl niters=300
restor map=model_jan24.mp beam=model_jan24.bm model=model_jan24.cl out=model_jan24.cm
fits in=model_jan24.cm op=xyout out=model_jan24.fits


