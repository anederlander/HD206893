from matplotlib import pyplot as plt
import numpy as np
import math
#make a variable for lnprob1 and lnprob2
#lnprob1 = -10732993.0 #gap
#lnprob2 = -10733006.4375 #nogap, flat disk
lnprob2 = -10732992.2
lnprob1 = -10733015.1


nvar1 = 13
nvar2= 8

#k is number of parameters (8 for no-gap model, 10 for gap model)

#AIC

AIC_nogap = 2*(nvar2) - 2*(lnprob2) #without p
AIC_gap = 2*(nvar1) - 2*(lnprob1) #with p
AIC_prob = np.exp((AIC_gap-AIC_nogap)/2) #— the probability the model without gap is better than with a gap (AICmin is whichever has lower chi square)
print("AIC without p=", AIC_nogap)
print("AIC with p=", AIC_gap)
print("AIC Probability=", AIC_prob)
#probability = exp((10-8) + (Delta chi^2)/2)



#BIC
nvis = 10190980 #number of visibilities
BIC_nogap = (nvar2)*math.log(nvis*2) - 2*(lnprob2)
BIC_gap = (nvar1)*math.log(nvis*2) - 2*(lnprob1)
delta_BIC = (BIC_gap-BIC_nogap)
print("BIC without p=", BIC_nogap)
print("BIC with p=", BIC_gap)
print("delta_BIC=", delta_BIC)

#Extra
#-10733011.5 #with p
