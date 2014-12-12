from __future__ import division
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as m
from matplotlib import font_manager as fm
import gc



def eeg_sleep_epochs(eeg,srate,stages,num):
    """
    Turns raw eeg data into an array of 30 sec epochs for sleep stages 1,2,3
    """
    bin_size_sec = 30
    (n_col,num_pts)=np.shape(eeg)
    #print ('num_pts'+ str(num_pts))
    num_epochs=num_pts/(bin_size_sec*srate) #rows
    #print ('num_epochs'+ str(num_epochs))
    num_samp=num_pts/num_epochs              #col    
    #print('num-samples' + str(num_samp))
    
    data_array= np.zeros((n_col,num,num_samp))

    j=0
    for i in range(int(num_epochs)) :
        if (i%100==0):
            print ('i is '+str(i))
            print ('j is '+str(j))
            print ('stage is ' + str(stages[i]))
        if (np.logical_or(stages[i]==1, np.logical_or(stages[i]==2, stages[i]==3))) :
                    data_array[:,j,:]=eeg[:,i*num_samp:(i+1)*num_samp]
                    j=j+1
            
        elif (np.logical_or(stages[i]==4, stages[i]==5)) :
                    data_array[:,j,:]=eeg[:,i*num_samp:(i+1)*num_samp]
                    j=j+1
        if j == num :
            break
    
    
    return data_array


def get_sleep_stages(stages):
    """
    Finds the number of NREM stages
    """
    n=len(stages) # n is the number of stages, m is the psd
    stages_out=np.zeros((n))
    j=0
    #print ('number of stages is' + str(len(stages)))
    for i in range(n):
           
        if (np.logical_or(stages[i]==0, np.logical_or(stages[i]==7,stages[i]==6))) :
            stages_out=np.delete(stages_out,j,0)
           
        elif stages[i]==4 :
            stages_out[j]=3
            j=j+1
        #np.logical_or(stages[i]!=3, np.logical_or(stages[i]!=2,stages[i]!=1)):  
           
        else :
            stages_out[j]=stages[i]
            j=j+1
    return stages_out


def get_mean3(eeg_epochs):
    (l,m,n)=np.shape(eeg_epochs)
    mean_values=np.zeros((l,m))
    for j in range(l):
        mean_values[j]=np.mean(eeg_epochs[j],1)
    return mean_values

def get_mean2(eeg_epochs):
    (m,n)=np.shape(eeg_epochs)
    mean_values=np.mean(eeg_epochs,1)
    return mean_values



    
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    #YOUR CODE HERE
    
    plt.close('all') #Closes old plots.
    os.chdir('C:\Users\LAurie\Desktop\problem_set4\sleep data\data')
    gc.enable()


 #PART 1
    #Load data
    #Note that I load the REC files in first to avoid some memory problems
    #Also note that i reinitialize the arrays when i don't need them to save some on memory

    with np.load('S2_REC.npz') as data:
        stages=data['stages']
        eeg=data['DATA']
        srate=data['srate']

    REC_sleep_stages=get_sleep_stages(stages) #this one selects the sleep stages
    REC_eeg_stages= eeg_sleep_epochs(eeg,srate,stages,len(REC_sleep_stages)) #this one puts each sleep stage in an eeg epoch 
    stages=[]
    eeg= []
    REC_Ave_eegs=np.average(np.r_[REC_eeg_stages[0:2,:,:],REC_eeg_stages[7:9,:,:]],0)   #This calculates the average eeg for the 4 channels (array of nsamples x n epochs)
    REC_ampl=REC_Ave_eegs*REC_Ave_eegs                                          #This one gets the amplitudes (still have an array for each epoch)

    REC_eeg_stages= []
    
    with np.load('S2_BSL.npz') as data:
        stages=data['stages']
        eeg=data['DATA']
    
    BSL_sleep_stages=get_sleep_stages(stages)
    BSL_eeg_stages= eeg_sleep_epochs(eeg,srate,stages,len(BSL_sleep_stages))
    stages=[]
    eeg= []
    BSL_Ave_eegs=np.average(np.r_[BSL_eeg_stages[0:2,:,:],BSL_eeg_stages[7:9,:,:]],0)   
    BSL_ampl=BSL_Ave_eegs*BSL_Ave_eegs

    BSL_eeg_stages= []

    Mean_BSL_ampl=get_mean2(BSL_ampl)       #This is where I take the mean of each epoch resulting in a single vector

    Mean_REC_ampl=get_mean2(REC_ampl)
    
    time_BSL=np.linspace(0,len(BSL_sleep_stages)*2,len(BSL_sleep_stages))
    time_REC=np.linspace(0,len(REC_sleep_stages)*2,len(REC_sleep_stages))

    #Let's do some plots
    
    f,ax=plt.subplots()
    ax.plot(time_BSL, np.log(Mean_BSL_ampl))
    ax.set_xlabel('time(min)')
    ax.set_ylabel('log(mean squared eeg)')
    ax.set_ylim(3,8)
    ax.set_title('Mean squared EEG for BSL Subject 2')

    ax2=ax.twinx()
    ax2.plot(time_BSL, BSL_sleep_stages, 'r')
    ax2.set_ylim(0.5,5.5)
    ax2.set_ylabel('sleep Stages')
    plt.show()

    f2,axis=plt.subplots()
    axis.plot(time_REC, np.log(Mean_REC_ampl))
    axis.set_xlabel('time(min)')
    axis.set_ylabel('log(mean squared eeg)')
    axis.set_ylim(3,8)
    axis.set_title('Mean EEG for REC Subject 2')
    

    axis2=axis.twinx()
    axis2.plot(time_REC, REC_sleep_stages, 'r')
    axis2.set_ylabel('sleep Stages')
    axis2.set_ylim(0.5,5.5)

    plt.show()
  
