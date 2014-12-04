"""
Created on Wed Nov 30

@author: RChau
Project: Sleep Deprivation

This code is for plotting and analysing the sleep-EEG data.

Note: This code is not working correctly yet.
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import matplotlib.mlab as m
from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from scipy.stats import mode

epoch_dur = 30

eeg_files = np.array([['S1_BSL.npz', 'S1_REC.npz'],
                      ['S2_BSL.npz', 'S2_REC.npz'],
                      ['S3_BSL.npz', 'S3_REC.npz'],
                      ['S4_BSL.npz', 'S4_REC.npz']])

def load_data(filename):
    """
    load_data takes the file name and reads in the data.  It returns an
    array containing EEG data and the sampling rate for
    the data in Hz (samples per second).
    """
    data = np.load(filename)
    return data['DATA'], int(data['srate']), data['stages']

def get_times(data_lth,rate):
    times = np.arange(0, 1.0*data_lth/rate, 1.0/rate)
    return times

def plot_stages(stages,step,title):
    plt.figure()
    times = np.arange(0, stages.size*step, step)
    plt.plot(times, stages, drawstyle='steps')
    plt.ylim(ymin=-0.5,ymax=7.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Stage")
    plt.title("Stages: " + title)
    plt.show()
    return

def plot_all_channels(data,rate,title):
    times = get_times(data[0].size,rate)
    plot_eeg_channels(data,times,title)
    plot_eog_channels(data,times,title)
    plot_emg_channels(data,times,title)
    return

def plot_eeg_channels(data,times,title):
    plot_channeln(data,times,0,title + ": EEG (C3/A2)")
    plot_channeln(data,times,1,title + ": EEG (O2/A1)")
    plot_channeln(data,times,7,title + ": EEG (C4/A1)")
    plot_channeln(data,times,8,title + ": EEG (O1/A2)")
    return

def plot_eog_channels(data,times,title):
    plot_channeln(data,times,2,title + ": EOG (ROC/A2)")
    plot_channeln(data,times,3,title + ": EOG (LOC/A1)")
    return

def plot_emg_channels(data,times,title):
    plot_channeln(data,times,4,title + ": EMG (Chin EMG 1)")
    plot_channeln(data,times,5,title + ": EMG (Chin EMG 2)")
    plot_channeln(data,times,6,title + ": EMG (Chin EMG 3)")
    return

def plot_channeln(data,times,n,title):
    plt.figure()
    plt.plot(times,data[n,:])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.title("Action Potential: " + title + " Channel No. " + str(n))
    plt.show()
    return

def plot_section_channels(data,rate,tstart,tend,title):
    times = get_times(data[0].size,rate)
    for i in range(0,data[:,0].size):
        plot_section_channel(data[i,tstart:tend],times[tstart:tend],title + ": Channel No. " + str(i))
    return

def plot_section_channel(eeg,times,title):
    plt.figure()
    plt.plot(times,eeg)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.title("Action Potential: " + title)
    plt.show()
    return

def plot_section_channel_bound(eeg,times,title,tstart,tend):
    plt.figure()
    plt.plot(times[tstart:tend],eeg[tstart:tend])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.title("Action Potential: " + title)
    plt.show()
    return

def plot_section_channeln(data,times,n,type_name,tstart,tend):
    plt.figure()
    plt.plot(times[tstart:tend],data[n,tstart:tend])
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.title("Action Potential: " + type_name + " Channel No. " + str(n))
    plt.show()
    return

def plot_data_psds_slot(data,rate,slot):
    plt.figure()
    for eeg in data:
        #print "eeg = " + str(eeg) 
        plot_eeg_psd(eeg,rate)
    plt.xlim(xmax=30)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectrum")
    plt.title("Power Spectrum Density")
    plt.show()

def plot_data_epoch_psds_slot(data,rate,slot):
    plt.figure()
    for eeg in data:
        #print "eeg = " + str(eeg) 
        plot_eeg_epoch_psds_slot(eeg,rate,slot)
    plt.xlim(xmax=20)
#    plt.xlim(0,5)
#    plt.ylim(0,1000)
#    plt.xlim(9,15)
#    plt.ylim(0,10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectrum")
    plt.title("Power Spectrum Density")
    plt.show()

def plot_eeg_epoch_psds_slot(eeg,rate,slot):
    lo_eeg = slot*epoch_dur*rate
    hi_eeg = (slot+1)*epoch_dur*rate
    #print "lo_eeg = " + str(lo_eeg) + ", hi_eeg = " + str(hi_eeg)
    plot_eeg_psdn(eeg[lo_eeg:hi_eeg],rate)

def plot_eeg_psd(eeg,rate):
    print "eeg = " + str(eeg) + ", eeg length = " + str(len(eeg))
    pxx, freqs = m.psd(eeg, Fs=rate)
    plt.plot(freqs, pxx, hold=True)

def plot_eeg_psdn(eeg,rate):
    print "eeg = " + str(eeg) + ", eeg length = " + str(len(eeg))
    pxx, freqs = m.psd(eeg, Fs=rate)
    plt.plot(freqs, pxx/sum(pxx), hold=True)

def classify_eeg(eeg,rate):
    """
    classify_eeg takes an array of eeg amplitude values and a sampling rate and 
    breaks it into 30s epochs for classification with the classify_epoch function.
    It returns an array of the classified stages.
    """
    bin_size_samp = epoch_dur*rate
    t = 0
    classified = np.zeros(len(eeg)/bin_size_samp)
    while t + bin_size_samp < len(eeg):
       classified[t/bin_size_samp] = classify_epoch(eeg[range(t,t+bin_size_samp)],rate)
       t = t + bin_size_samp
    return classified

def classify_epoch(eeg_epoch,rate):
    """
    This function returns a sleep stage classification (integers: 1 for NREM
    stage 1, 2 for NREM stage 2, and 3 for NREM stage 3/4) given an epoch of 
    EEG and a sampling rate.
    """

    ###YOUR CODE HERE
    stage = 1

    return stage

def classify_data_kmeans(data,rate):
    """
    classify_data_kmeans takes a 2D array of eeg amplitude values and a sampling rate, and 
    breaks it into 30s epochs for classification with the k-means classifier.
    It returns an array of the classified stages.
    Note: the returned stages are numbered in its own numbering system, which
    is likely to be different to the given stages.
    """
    bin_size_samp = epoch_dur*rate
    t = 0
    features = np.zeros((len(data[0])/bin_size_samp,3))
    while t + bin_size_samp < len(data[0]):
       features[t/bin_size_samp,:] = get_features(data[:,range(t,t+bin_size_samp)],rate)
       t = t + bin_size_samp
    k_means = KMeans(n_clusters=2)  
    k_means.fit(features)
    k_means.predict(features)
    return k_means.labels_

def get_features(data_epoch,rate):
    pxxs = []
    for eeg_epoch in data_epoch:
        pxx, freqs = m.psd(eeg_epoch, Fs=rate)
    pxxs.append(sum(pxx[plt.find((10 <= freqs) & (freqs <= 15))]))
    pxxs.append(max(pxx[plt.find((10 <= freqs) & (freqs <= 15))]))
    pxxs.append(np.mean(pxx[plt.find((10 <= freqs) & (freqs <= 15))]))
    return pxxs    
    #return [sum(pxx[plt.find((10 <= freqs) & (freqs <= 15))]),max(pxx[plt.find((10 <= freqs) & (freqs <= 15))]),np.mean(pxx[plt.find((10 <= freqs) & (freqs <= 15))])]

def norm(V):
    #L = np.math.sqrt( sum( [x*x for x in V] ) )
    L = sum( [x for x in V] )
    return [ x/L for x in V ]

if __name__ == "__main__":
    plt.close('all') #Closes old plots.
    
    # Load the data
    data, srate, stages = load_data('S1_BSL.npz')
    
    # Plot the data
    #plot_all_channels(data,srate,"S1_BSL")
    plot_data_psds_slot(data,srate,0)
    #plot_stages(stages,epoch_dur,"S1_BSL")
    
    # Classify the data using k-means
    k_means_labels = classify_data_kmeans(data,srate)
    
