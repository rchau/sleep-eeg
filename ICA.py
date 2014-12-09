from __future__ import division
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as m
from scipy import signal
from matplotlib import font_manager as fm
import random
from sklearn import linear_model

from sklearn.decomposition import FastICA, PCA
import gc

def eeg_epochs(eeg,srate):
    """
    Turns raw eeg data into an array of power spectral density for 30 sec epochs
    """
    bin_size_sec = 30
    (n_col,num_pts)=np.shape(eeg)
    #print ('num_pts'+ str(num_pts))
    num_epochs=num_pts/(bin_size_sec*srate) #rows
    #print ('num_epochs'+ str(num_epochs))
    num_samp=num_pts/num_epochs              #col    
    #print('num-samples' + str(num_samp))
    data_array= np.zeros((n_col,num_epochs,num_samp))

    for i in range(int(num_epochs)) :
        data_array[:,i,:]=eeg[:,i*num_samp:(i+1)*num_samp]
    return data_array

def NREM_eeg_epochs(eeg,srate,stages):
    """
    Turns raw eeg data into an array of of 30 sec epochs that contain just the NREM data
    """
    bin_size_sec = 30
    (n_col,num_pts)=np.shape(eeg)
    #print ('num_pts'+ str(num_pts))
    num_epochs=num_pts/(bin_size_sec*srate) #rows
    #print ('num_epochs'+ str(num_epochs))
    num_samp=num_pts/num_epochs              #col    
    #print('num-samples' + str(num_samp))
    if (num_epochs%2==0):
        limit1,limit2=num_epoches/2
    else :
        limit1=round(num_epochs/2)
        limit2=round(num_epochs/2)-1
    print('limit1 is ')+str(limit1) + (' limit2 is ')+str(limit2)
    data_array1= np.zeros((n_col,limit1,num_samp))
    data_array2= np.zeros((n_col,limit2,num_samp))
    j=0
    for i in range(int(limit1)) :
        if (np.logical_or(stages[i]==1, np.logical_or(stages[i]==2, stages[i]==3))) :
                    data_array1[:,j,:]=eeg[:,i*num_samp:(i+1)*num_samp]
                    j=j+1
            
        elif stages[i]==4 :
                    data_array1[:,j,:]=eeg[:,i*num_samp:(i+1)*num_samp]
                    j=j+1
        else :    
                    data_array1=np.delete(data_array1,j,1)
    j=0
    k=limit1
    for i in range(int(limit2)) :
        if (np.logical_or(stages[k]==1, np.logical_or(stages[k]==2, stages[k]==3))) :
                    data_array2[:,j,:]=eeg[:,k*num_samp:(k+1)*num_samp]
                    j=j+1
                    k=k+1
            
        elif stages[k]==4 :
                    data_array2[:,j,:]=eeg[:,k*num_samp:(k+1)*num_samp]
                    j=j+1
                    k=k+1
        else :    
                    data_array2=np.delete(data_array2,j,1)
                    k=k+1
    
    return data_array1,data_array2

def run_ica(data, comp):
    ica = FastICA(n_components=comp, whiten=True, max_iter=5000)
    data_out=np.zeros((comp,np.shape(data[0,:,0])[0],np.shape(data[0,0,:])[0]))
    for i in range(np.shape(data[0,:,0])[0]):
        print i
        data_out[:,i,:]=np.transpose(ica.fit_transform(np.transpose(data[:,i,:])))
    return data_out

def plot_results(data, title):
 
    time=np.linspace(0,30,3840)
    f,axarr=plt.subplots(4,sharex=True, sharey=True)
    for i in range(2) :
        axarr[i].plot(time,data[i,0,:])
        axarr[i].set_ylabel('eeg ch '+ str(i+1))
  
    
    axarr[0].set_title(title)
    
    for i in range(2) :
        j=i+7
        k=i+2
        axarr[k].plot(time,data[j,0,:])

        axarr[k].set_ylabel('eeg ch' + str(i+8))
    axarr[3].set_xlabel('time (sec)')
    os.chdir('C:\Users\LAurie\Desktop\problem_set4\sleep data')
    plt.savefig('EEG_9comp_max_iter5000.png', bbox_inches='tight')

def plot_other(data, title):
 
    time=np.linspace(0,30,3840)
    f,axarr=plt.subplots(5,sharex=True, sharey=True)
    for i in range(2) :
        j=i+2
        axarr[i].plot(time,data[j,0,:])
        axarr[i].set_ylabel('eog ch '+ str(j+1))
  
    
    axarr[0].set_title(title)
    
    for i in range(3) :
        j=i+4
        k=i+2
        axarr[k].plot(time,data[j,0,:])

        axarr[k].set_ylabel('emg ch' + str(i+8))
    axarr[4].set_xlabel('time (sec)')   
    os.chdir('C:\Users\LAurie\Desktop\problem_set4\sleep data')
    plt.savefig('Other_9comp_maxiter5000.png', bbox_inches='tight')
       
    
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    #YOUR CODE HERE
    
    plt.close('all') #Closes old plots.
    os.chdir('C:\Users\LAurie\Desktop\problem_set4\sleep data\data')
    gc.enable() 
    ##PART 1
    #Load data
    with np.load('S1_BSL.npz') as data:
        S1_BSL_stages=data['stages']
        
    all_data=np.load('S1_BSL_epochs.npy')
    
    data_transform=run_ica(all_data, 9)
  
    np.save('S1_BSL_ICA_9comp_max_iter5000.npy', data_transform)

    plot_results(transform_data, 'Transformed EEG NREM index 0 N=9')

    plot_other(transform_data, 'Other channels Transformed NREM index 0')
