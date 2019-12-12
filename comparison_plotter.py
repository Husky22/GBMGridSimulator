# -*- coding: utf-8 -*-
import h5py
import numpy as np
import os
import threeML.analysis_results as results

envpath = os.environ.get('SIMULATOR')

def plot(simname, gpname, i, j):
    file=h5py.File(envpath+"/"+simname+"_Files/"+simname+".hdf5","r")
    fisher=file["grid/"+gpname+"/fisher"]
    gptrue=file["grid/"+gpname].attrs["True Position"]
    l=[]
    for f in fisher:
        l.append((f,np.linalg.norm(gptrue-fisher[f].attrs["Position"])))
    print(l)
    l.sort(key=lambda x: x[1])
    print(l)

plot('FitSimulation','gp0',0,0)


