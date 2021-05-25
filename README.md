Resolving Structure in the Debris Disk around HD 206893 with ALMA


From June 2019 - January 2021, I worked at Wesleyan University Astronomy Department and Van Vleck Observatory under the guidance of Dr. A. Meredith Hughes. This repository contains the code that helped achieve the results in our paper: https://ui.adsabs.harvard.edu/abs/2021arXiv210108849N/abstract 

Note: flat disk and continuous disk refer to the same thing. File paths need to be adjusted accordingly.

Files:

aicbic.py <- calculates the AIC and BIC

allplots_flat.py <- calculates the best-fit, and plots the walker evolution and corner plots for the continuous disk 

allplots_doublepp.py <- calculates the best-fit, and plots the walker evolution and corner plots for the double power law disk 

allplots_triplepp.py <- calculates the best-fit, and plots the walker evolution and corner plots for the triple power law disk 

autocor_doublepp_gap_finalv2.pdf <-autocorrelation plot for double power law with gap data

autocor_flatdisk.pdf <-autocorrelation plot for continuous disk data

autocor_nov30_triplepp.pdf <- autocorrelation plot for the triple power law without gap data

autocorrelation_flat.py <- python code to create the autocorrelation plot for double power law with gap data

autocorrelation_test_doublepp_gap.py <- python code to create the autocorrelation plot for double power law with gap data

autocorrelation_triplepp.py <-python code to create the autocorrelation plot for the triple power law without gap data

best_fit_doublepp.py <- best-fit data for the doublepp data

best_fit_triplepp.py <- best-fit data for the triplepp data

best_fit.py <- best-fit data for the flat disk data

combine_csv.py <- combines two csv files into only one csv file

cornerplot_aug13_doublepp_gap.pdf <- corner plot for doublepp with gap

cornerplot_flatdisk.pdf <- corner plot for flat disk

cornerplot_triplepp.pdf <- corner plot for triplepp without gap

csv_array_gap.py <- histogram plots with gap

csv_array_nogap.py <- histogram plots without gap

debris_disk.py <- Kevin Flaherty’s code for MCMC algorithm. Refer here: https://github.com/kevin-flaherty/disk_model3

dmr_doublepp_nogap.pdf <- dmr plot for doublepp without gap

dmr_doublepp.pdf <- dmr plot for doublepp with gap

dmr_flat.pdf <- dmr plot for flat disk without gap

dmr_gap.pdf <- dmr plot for flat disk with gap

dmr_hd206893.py <- python code for creating all dmr plots

dmr_singlepp_gap.pdf <- dmr plot for singlepp with gap

dmr_singlepp_nogap.pdf<- dmr plot for singlepp without gap

dmr_triplepp_nogap.pdf <- dmr plot for triplepp without gap

histograms_continuous.pdf <- histogram plots for continuous disk

histograms_gap.pdf <- histogram plots for doublepp with gap

histograms_nogap1.pdf <- histogram plots for triplepp without gap

june_6 <- personal notes that started on June 6, 2019

mcmc_doublepp_gap.py <- “rev” may be more up to date, but creates fits for the MCMC data 

mcmc_gap.py <-  mcmc python code

mcmc_nogap.py <-  mcmc python code

modeling_shell_script_residual.sh <- creating the final fits file for residual data from model_imaging python code

modeling_shell_script.sh <- creating the final fits file for model data from model_imaging python code

model_imaging_doublepp.py <- creates fits files based on every spectral window

model_imaging_singlepp.py <- creates fits files based on every spectral window

model_imaging_triplepp.py <-creates fits files based on every spectral window

model_imaging.py <- creates fits files based on every spectral window

raytrace_gaussian.py <- extra code

raytrace.py <- Kevin Flaherty’s code for MCMC algorithm. Refer here: https://github.com/kevin-flaherty/disk_model3

rev_mcmcmodel2_gap_copy.py <- mcmc python code

revmcmc_model1_copy.py <-  mcmc python code

sample_alma.csh <- Kevin Flaherty’s code for MCMC algorithm. Refer here: https://github.com/kevin-flaherty/disk_model3

single_model.py <- Kevin Flaherty’s code for MCMC algorithm. Refer here: https://github.com/kevin-flaherty/disk_model3

walker_aug12_doublepp_gap.pdf <- walker evolution plots for doublepp with gap

walker_flatdisk.pdf <- walker evolution plots for flat disk without gap

walker_nov30_triplepp.pdf <- walker evolution plots for triplepp without gap

walker_triplepp.py <- best-fit calculation and walker evolution plot


