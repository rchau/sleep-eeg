#
#  NAME
#    sleep_data.py
#
#  DESCRIPTION
#    In Problem Set 4, you will classify EEG data into NREM sleep stages and
#    create spectrograms and hypnograms.
#
from __future__ import division
import os
import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as m
from scipy import signal
from matplotlib import font_manager as fm
import random
from sklearn import linear_model
from sklearn import cross_validation

def plot_comp(s1_bsl, s2_bsl, s1_rec, s2_rec):
    """
    This function plots the baseline and sleep deprived curves of to subjects
    """
    
    time1=np.linspace(0,10,np.shape(s1_bsl)[0])
    time2=np.linspace(0,10,np.shape(s2_bsl)[0])
    time3=np.linspace(0,10,np.shape(s1_rec)[0])
    time4=np.linspace(0,10,np.shape(s2_rec)[0])

    
    f,axarr=plt.subplots(2,2,sharex='col', sharey='row')
    axarr[0,0].plot(time1,s1_bsl)
    axarr[0,0].set_title('Subject 2 Baseline')
    axarr[0,0].set_ylabel('sleep stage')
    axarr[0,0].set_xlabel('Time (hours)')
    axarr[0,0].set_ylim(0.5,5.5)
    
    axarr[0,1].plot(time3,s1_rec)
    axarr[0,1].set_title('Subject 2 Sleep Deprived')
    axarr[0,1].set_ylabel('sleep stage')
    axarr[0,1].set_xlabel('Time (hours)')
    axarr[0,1].set_ylim(0.5,5.5)

    axarr[1,0].plot(time2,s2_bsl)
    axarr[1,0].set_title('Subject 3 Baseline')
    axarr[1,0].set_ylabel('sleep stage')
    axarr[1,0].set_xlabel('Time (hours)')
    axarr[1,0].set_ylim(0.5,5.5)

    axarr[1,1].plot(time4,s2_rec)
    axarr[1,1].set_title('Subject 3 Sleep Deprived')
    axarr[1,1].set_ylabel('sleep stage')
    axarr[1,1].set_xlabel('Time (hours)')
    axarr[1,1].set_ylim(0.5,5.5)
    
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    plt.show()



def eeg_epochs(eeg,srate,):
    """
    Turns raw eeg data into an array of power spectral density for 30 sec epochs
    """
    bin_size_sec = 30
    num_pts=len(eeg)
    #print ('num_pts'+ str(num_pts))
    num_epochs=num_pts/(bin_size_sec*srate) #rows
    #print ('num_epochs'+ str(num_epochs))
    num_samp=num_pts/num_epochs              #col    
    #print('num-samples' + str(num_samp))
    data_array = np.zeros((num_epochs,num_samp))
    num_loops=num_epochs
    Pxx_array=np.zeros((num_epochs, 257))
    #NPxx=np.zeros((num_epochs, 257))
    for i in range(int(num_loops)) :
        #data_array[i,:]=eeg[i*num_samp:(i+1)*num_samp]
        data=eeg[i*num_samp:(i+1)*num_samp]
        freq,Pxx_array[i,:]=signal.welch(data,srate,nperseg=512)
        #NPxx[i,:]=((Pxx_array[i,:]))/np.sum(abs(Pxx_array[i,:]))
    return Pxx_array

def get_NREM_stages(stages):
    """
    Finds the number of NREM stages
    """
    n=len(stages) # n is the number of stages, m is the psd
    stages_out=np.zeros((n))
    j=0
    #print ('number of stages is' + str(len(stages)))
    for i in range(n):
        if np.logical_or(stages[i]==0, np.logical_or(stages[i]==7, np.logical_or(stages[i]==6, stages[i]==5))) :
            stages_out=np.delete(stages_out,j,0)
            
        elif stages[i]==4 :
            stages_out[j]=3
            j=j+1
        #np.logical_or(stages[i]!=3, np.logical_or(stages[i]!=2,stages[i]!=1)):  
           
        else :
            stages_out[j]=stages[i]
            j=j+1
    return stages_out

def get_NREM_psd(pxx_array, stages):
    """
    Transforms psd with all stages into psd with just the NREM stages
    also combines stage 3 and stage 4 into a single stage 3
    """
    (n,m)=np.shape(pxx_array) # n is the number of stages, m is the psd
    NREM_Pxx=np.zeros((n,m))
    j=0
    for i in range(n):
      
        if (np.logical_or(stages[i]==1, np.logical_or(stages[i]==2, stages[i]==3))) :
            NREM_Pxx[j,:]=pxx_array[i,:]
            j=j+1
            
        elif stages[i]==4 :
            NREM_Pxx[j,:]=pxx_array[i,:]
            j=j+1
        else :
           NREM_Pxx=np.delete(NREM_Pxx,j,0)

    return NREM_Pxx

def norm_Pxx(Pxx):
    (m,n,l)=np.shape(Pxx)
    NPxx=np.zeros((m,n,l))
    for j in range(m):
        for i in range(n):
            NPxx[j,i,:]=((Pxx[j,i,:]))/np.sum(Pxx[j,i,:])
    return NPxx

def get_mean(ch,index):
    """
    This function takes the mean of 4 PSD channels at each frequency
    """
    ch1=ch[0,index,:]
    ch2=ch[1,index,:]
    ch3=ch[2,index,:]
    ch4=ch[3,index,:]
    average=np.ones((len(ch1)))
#    for i in range(len(ch1)):
    average=np.mean([ch1, ch2, ch3, ch4], axis=0)
    return average
def prepare_data(ch):
    """
    This function considers data from 0-5 and 10-15 Hz, it then computes the average
    and normalizes it.
    """
    ch_1=ch[:,:,0:20]
    ch_2=ch[:,:,40:60]
 
    (m1,n1,l1)=np.shape(ch_1)
    (m2,n2,l2)=np.shape(ch_2)
    nch_1=norm_Pxx(ch_1)
    nch_2=norm_Pxx(ch_2)
    ave_nch_1=np.zeros((n1,l1))
    ave_nch_2=np.zeros((n2,l2))
    for i in range(n1):
        ave_nch_1[i,:]=get_mean(nch_1,i)
    for j in range(n2):
        ave_nch_2[j,:]=get_mean(nch_2,j)
        
    new_ch=np.append(ave_nch_1 , ave_nch_2,1)
    np.shape(new_ch)
    return new_ch


def prepare_data2(ch):
    """
    This function takes the 4 channel data, computes the average and
     normalizes it.
    """
    
    (m,n,l)=np.shape(ch)
    nch=norm_Pxx(ch)
    ave_nch=np.zeros((n,l))
    
    for i in range(n):
        ave_nch[i,:]=get_mean(nch,i)
        
    return ave_nch


def get_rand_indices(stages):
    #determine indices for the different sleep stages
    ones=np.array(np.where(stages==1)[0])
    twos=np.array(np.where(stages==2)[0])
    threes=np.array(np.where(stages==3)[0])
    sleep=[1,2,3]
#random choose an index from each stage
    m=len(ones)-1
    index_1=ones[random.randint(0,m)]
    m=len(twos)-1
    index_2=twos[random.randint(0,m)]
    m=len(threes)-1
    index_3=threes[random.randint(0,m)]
    index=[index_1, index_2, index_3]
    return index

def divide_data(ch, stages, frac):
    #find number points
    (m,n)=np.shape(ch)
    num_train=int(round(m*frac))
    num_test=m-num_train
    x_train=np.zeros((num_train,n))
    x_test=np.zeros((num_test,n))
    y_train=np.zeros((num_train))
    y_test=np.zeros((num_test))

    #find the points in the different stages
    ones=np.array(np.where(stages==1)[0])
    twos=np.array(np.where(stages==2)[0])
    threes=np.array(np.where(stages==3)[0])
    #calculate where to divide the different stages
    one_index=int(round(len(ones)*frac))
    two_index=int(round(len(twos)*frac))
    three_index=int(round(len(threes)*frac))
    #create the indices for the the divided stages and sort them
    train=ones[0:one_index]
    test=ones[one_index:]
    train=np.append(train,twos[0:two_index])
    train=np.append(train,threes[0:three_index])
    test=np.append(test,twos[two_index:])
    test=np.append(test,threes[three_index:])
    test=np.sort(test)
    train=np.sort(train)
    #create the new stages and channels
    for i in range(len(train)):
        x_train[i,:]=ch[train[i],:]
        y_train[i]=stages[train[i]]
    for i in range(len(test)):
        x_test[i,:]=ch[test[i],:]
        y_test[i]=stages[test[i]]
    return x_train, y_train, x_test,y_test




  

        
##########################
#You can put the code that calls the above functions down here    
if __name__ == "__main__":
    #YOUR CODE HERE
    
    plt.close('all') #Closes old plots.
    os.chdir('C:\Users\LAurie\Desktop\problem_set4\sleep data')
        
    ##PART 1
    #Load data
    with np.load('S1_BSL.npz') as data:
        S1_BSL=data['DATA']
        srate=data['srate']
        S1_BSL_stages=data['stages']
        
    #separte into different parts
    S1_BSL_eeg=S1_BSL[0:4,:]
    S1_BSL_Pxx=np.zeros((4,len(S1_BSL_stages),257))
    S1_BSL_NREM_stages=get_NREM_stages(S1_BSL_stages)
    S1_BSL_NREM=np.zeros((4,len(S1_BSL_NREM_stages),257))

    #S1_BSL_eeg_sum=np.zeros(4860928)
    #S1_BSL_eeg_sum=np.sum(S1_BSL_eeg,0)
    #S1_BSL_Pxx_sum=np.zeros((len(S1_BSL_stages),257))

#This next step takes time. Any ideas for making it faster?
    for i in range(4):
        S1_BSL_Pxx[i]=eeg_epochs(S1_BSL_eeg[i,:],srate)
    
    for i in range(4):
        S1_BSL_NREM[i]=get_NREM_psd(S1_BSL_Pxx[i], S1_BSL_stages)
        
    freq,test=signal.welch(S1_BSL_eeg[0,0:3840],srate,nperseg=512)
    freq=freq[0:80]
    
   
    #Prepare data for classifier
    S1_BSL_cl=prepare_data(S1_BSL_NREM)
    #freq_test=np.append(freq[0:20], freq[40:60])

    print(np.shape(S1_BSL_cl))
   
    #Classify data with logistic regression
    x_train, y_train, x_test, y_test=divide_data(S1_BSL_cl, S1_BSL_NREM_stages, 0.6)
    
    logreg=linear_model.LogisticRegression(C=1)
    logreg.fit(S1_BSL_cl,S1_BSL_NREM_stages)
    theta=logreg.coef_
    logreg.score(S1_BSL_cl,S1_BSL_NREM_stages)
    print(logreg.score(S1_BSL_cl,S1_BSL_NREM_stages))

 
