'''
This is a Code from Renaldas, a programmer in the discussion forum who is helping us.
I tested the first version but there was a bug in it, this version is yet to be tested.
'''
def bin_eeg(eeg, srate):

    bin_size_sec = 30

    bin_size_samp = bin_size_sec*srate




    t = 0

    bins = []

    while t + bin_size_samp < len(eeg):

        bins.append(eeg[range(t,t+bin_size_samp)])

        t = t + bin_size_samp

    return bins




def classify_eeg(binned_eegs, srate):

    classified = []

    for eeg in binned_eegs:

       classified.append(classify_epoch(eeg,srate))

    return classified

    

    

def find_max_amplitude (DATA, srate):

    # iterate over all patients' data

    for patient_eeg in DATA:

        # bin eeg data into 30 seconds packets

        binned_eeg = bin_eeg(patient_eeg, srate)

        # use slightly modified  classify_eeg from ProblemSet4 (see function above)

        # to detect sleep stages

        stages = classify_eeg(binned_eeg, srate)




        # convert to numpy array, useful for further operations

        binned_eeg = np.array(binned_eeg)

        stages = np.array(stages)




        # iterate over NREM stages of sleep

        for stage in range(1,4):

            # find eeg packets that correspond to particular sleep stage

            # where() function returns indices which match criteria (particular sleep stage in this case)

            # we can use result of where() function to select 30 seconds packets from binned_eeg array that correspond to particular sleep stage

            filter_bins_at_particular_stage = binned_eeg[np.where(stages == stage)]

            

            # skip, if no eeg for particular stage found

            if (len(filter_bins_at_particular_stage) == 0):

                break

            

            amplitudes = []

            frequencies = []

            for eeg in filter_bins_at_particular_stage:

                # FFT

                spectrum = m.psd(eeg, 256, srate)

                # gather signal amplitudes into an array

                amplitudes = np.concatenate([amplitudes, spectrum[0]])

                # gather frequency distributions into an array

                frequencies = np.concatenate([frequencies, spectrum[1]])

    

            # find max amplitude and corresponding frequency

            # argmax() function returns index of the maximum element in the array

            max_amplitude_index = np.argmax(amplitudes)

            max_amplitude = amplitudes[max_amplitude_index]

            frequency_at_max_amplitude = frequencies[max_amplitude_index]

    

            # print results

            print (stage)

            print ([max_amplitude, frequency_at_max_amplitude])

        print ('next patient')
'''
Second part of Renalsdas code for Maximum frequency.
'''
# so if you need an actual maximum frequency of EEG, I'd threshold amplitudes, cut below 55Hz (to avoid 60Hz artifacts) and would take the last element that passed the threshold.

# Possible usage:

max_frequencies = []

for eeg in filter_bins_at_particular_stage:

    spectrum = m.psd(eeg, 256, srate)


    max_frequencies.append (max_frequency(spectrum[0], spectrum[1]))
print (max(max_frequencies))



Store max_frequency in array when iterating over the eeg 30 second packets. Afterwords just pick maximum from that array.





FREQ_THRESHOLD = .1

def max_frequency(amplitudes, frequencies):

    amplitudes = np.array(amplitudes)

    frequencies = np.array(frequencies)

        

    frequencies_with_amplitudes_above_threshold = frequencies[np.where(amplitudes > FREQ_THRESHOLD)]

    reasonable_frequencies_with_amplitudes_above_threshold = frequencies_with_amplitudes_above_threshold[

        np.where(frequencies_with_amplitudes_above_threshold < 55)]

    

    max_frequency = 0




    if (len(reasonable_frequencies_with_amplitudes_above_threshold) > 0):

        max_frequency = reasonable_frequencies_with_amplitudes_above_threshold[-1]

    

    return max_frequency
