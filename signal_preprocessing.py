import pyxdf
import mne
import getopt
import sys
import csv
import numpy as np


def show_time_series(eeg_data, title):
    """
    Helper function to plot time series from a raw eeg object
    :param eeg_data:
    Raw object containing eeg data
    :param title:
    The title that should be displayed on the plot
    """
    eeg_data.plot(scalings={'eeg': 1e2},
                  show=True,
                  title=title,
                  block=True)


def run_ica(eeg_data, n_components, method, plot_filter_):
    """
    Function to create and fit ica object.
    :param eeg_data:
    Raw object containing eeg data
    :param n_components:
    Integer or None type.
    If it is an integer, decides the number of components used in the ica.
    If n_components is None, will let mne decide dynamically how many components should be used.
    Look at the 'mne doc for ICA <https://mne.tools/stable/generated/mne.preprocessing.ICA.html>' for the most updated methods
    :param method:
    This parameter should contain a string that describes which method that should be used.
    Legal methods of the time of implementation are {'fastica','infomax','picard'}. This method is only tested for fastica
    :param plot_filter_:
    Boolean that represents if the components should be plotted. If it is True, it will plot a topography of the components and a time seies of the components.
    :return:
    This returns the fitted ICA object
    """
    ica = mne.preprocessing.ICA(n_components=n_components,
                                method=method)
    ica.fit(eeg_data, picks='eeg')
    if plot_filter_:
        ica.plot_components(title='ICA composition')
    ica_sources = ica.get_sources(eeg_data)
    if plot_filter_:
        show_time_series(ica_sources, 'ICA decomposed series')
    return ica

def create_correlation_table(ica, eeg, acc):
    """
    This method will create a correlation matrix between the raw time series and the accelerations.
    Currently only 3 directions (X,Y,Z) and 6 directions (lin_X, lin_Y, lin_Z, ang_X, ang_Y, ang_Z) is allowed.
    :param ica:
    This is a fitted ICA object
    :param eeg:
    Raw object containing eeg data.
    :param acc:
    Raw object containing the acceleration. Must have the shape [n,x] where n is either 3 or 6 and x = eeg length
    :return:
    (correlation_table,eeg_unmixed)
    correlation_table:
    This is a numpy array containing a table of correlations.
    This will be returned as a [independent_components,acceleration_direction+1] numpy array. correlation_table[:,0] will contain the acceleration names.

    eeg_unmixed:
    This is a raw object of the eeg data after being unmixed by the ica object
    """
    corr_matrix = [ica.score_sources(inst=eeg,
                              target=np.expand_dims(accuracy, axis=0),
                              score_func='pearsonr') for accuracy in acc.get_data()]

    if len(acc.get_data()) == 3:
        print("{:<12} {:<20} {:<20} {:<20}".format('ICA COMPONENTS',
                                                   acc.ch_names[0],
                                                   acc.ch_names[1],
                                                   acc.ch_names[2]))
        for data in zip(ica._ica_names, corr_matrix[0], corr_matrix[1], corr_matrix[2]):
            print("{:<12} {:<20} {:<20} {:<20}".format(data[0],data[1],data[2], data[3]))

    elif len(acc.get_data()) == 6:
        print("{:<12} {:<20} {:<20} {:<20} {:<20} {:<20} {:20}".format('ICA COMPONENTS',
                                                                       acc.ch_names[0],
                                                                       acc.ch_names[1],
                                                                       acc.ch_names[2],
                                                                       acc.ch_names[3],
                                                                       acc.ch_names[4],
                                                                       acc.ch_names[5]))


        for data in zip(ica._ica_names, corr_matrix[0], corr_matrix[1], corr_matrix[2],
                        corr_matrix[3], corr_matrix[4], corr_matrix[5]):
            print("{:<12} {:<20} {:<20} {:<20} {:<20} {:<20} {:20}".format(data[0],
                                                                           data[1],
                                                                           data[2],
                                                                           data[3],
                                                                           data[4],
                                                                           data[5],
                                                                           data[6]))
    else:
        raise ValueError("received incorrect number of accuracy directions. Need 3 or 6")
    corr_matrix = np.array(corr_matrix)
    # Creates the cleaned signal
    eeg_raw_unmixed = unmix_ica(ica,corr_matrix, eeg)
    return np.vstack((np.array(acc.ch_names),corr_matrix.T)), eeg_raw_unmixed

def unmix_ica(ica, corr_matrix, eeg):
    """
    Reconstructs the eeg space using the ica object. This method excludes all independent components that are more
    correlated then mean + (2*standard deviation).
    :param ica:
    This is a fitted ICA object
    :param corr_matrix:
    This is the correlation table between each of the independent components and acceleration directions
    :param eeg:
    this is a raw object containing eeg data that should be unmixed. This does not unmix in-place, this raw will be preserved
    :return:
    Raw object containing the unmixed eeg data.
    """
    # reconstructtion of electrode space
    corr_matrix = np.absolute(corr_matrix)
    mean = corr_matrix.mean()
    std = np.std(corr_matrix)
    threshold = std * 2 + mean
    correlated_sources = np.where(np.any(corr_matrix >= threshold, axis=0))[0]
    print('With a threshold of {}; components to be removed: {}'.format(threshold, correlated_sources))
    eeg_raw_unmixed = ica.apply(inst=eeg.copy(), exclude=correlated_sources)
    return eeg_raw_unmixed

def split_raw(full_raw, times, init_time):
    """
    This method creates a list of raw objects that are pieces of the original raw object based on the time stamps given
    :param full_raw:
    This is the original raw object given. This will not be transformed in-place
    :param times:
    This is a list of time stamps that give us a pairwise start and stop time
    :param init_time:
    This is the first timestamp, used to shift the time to start at 0 sek
    :return:
    List of raw objects that are slices of the original raw objec
    """
    res = []
    for i,(start, end) in enumerate(zip(*[iter(times)]*2)): ##  Loop that gets every time stamp pairwise
        res.append({'data':full_raw.copy(),
                    'start_time':start,
                    'end_time':end})
        res[i]['data'].crop(tmin=start-init_time,tmax=end-init_time)
    return res

def full_correlation_check(ica_, sources_, raw_, ref_, hmd_, is_filtered):
    """
    This is a function that creates a correlation tables for both vr headset and reference acceleration.
    Finally it saves the correlation tables and plots to a file.
    :param ica_:
    This is an ica object
    :param sources_:
    This is a raw object containing the source data for each independent component.
    :param raw_:
    This is a raw object containing the eeg data.
    Note: These raw objects must have equal amount of data points
    :param ref_:
    This is a raw object containing the reference acceleration
    Note: These raw objects must have equal amount of data points
    :param hmd_:
    This is a raw object containing the VR headset measured acceleration
    Note: These raw objects must have equal amount of data points
    :param is_filtered:
        This is a boolean representing whether the raw objects went through an extra filter process in the end. This will slightly change the title of the plots.
    """

    # Creating the correlation tables
    ref_correlation_matrix, eeg_ref_unmixed= create_correlation_table(ica_, raw_['data'], ref_['data'])
    hmd_acc_correlation_matrix, eeg_hmd_unmixed = create_correlation_table(ica_, raw_['data'], hmd_['data'])
    ica_names = np.concatenate(([0],ica_._ica_names))
    table_body_ref = np.vstack((ica_names,
                            ref_correlation_matrix.T))
    table_body_hmd = np.vstack((ica_names,
                            hmd_acc_correlation_matrix.T))

    # Saves the correlation tables as csv
    with open('.\\data\\correlation_table_{0}{1}.csv'.format(csv_path_ref,'_filtered' if is_filtered else ''),'a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start time:', raw_['start_time'], 'End time', raw_['end_time']])
        writer.writerows(table_body_ref)
        writer.writerow([])

    with open('.\\data\\correlation_table_{0}{1}.csv'.format(csv_path_hmd,'_filtered' if is_filtered else ''), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start time:', raw_['start_time'], 'End time', raw_['end_time']])
        writer.writerows(table_body_hmd)
        writer.writerow([])
        '''
        print('REF ACCELERATION\n--------\n'
          'max correlation: {0}\n'
          'min correlation: {1}\n'
          'HMD ACCELERATION\n--------\n'
          'max correlation {2}\n'
          'min correlation {3}'.format(np.amax(ref_correlation_matrix), np.amin(ref_correlation_matrix),
                                       np.amax(hmd_acc_correlation_matrix), np.amin(hmd_acc_correlation_matrix)))
        '''

    # Saves the plots as jpeg
    unmixed_ref_fig = eeg_ref_unmixed.plot(scalings={'eeg':3e2},
                                           show=False,
                                           show_options=False,
                                           title='Unmixed {0} unmixed {3} between times {1:.2f}s {2:.2f}s'
                                           .format('filtered' if is_filtered else 'unfiltered',
                                                   raw_['start_time'],
                                                   raw_['end_time'],
                                                   csv_path_ref))
    unmixed_ref_fig.savefig('.\\data\\plots\\unmixed_{0}_unmixed_{3}_between_times_{1:.2f}s_{2:.2f}s.jpeg'
                                           .format('filtered' if is_filtered else 'unfiltered',
                                                   raw_['start_time'],
                                                   raw_['end_time'],
                                                   csv_path_ref))

    unmixed_hmd_fig = eeg_hmd_unmixed.plot(scalings={'eeg': 3e2},
                                           show=False,
                                           show_options=False,
                                           title='Unmixed {0} unmixed {3} between times {1:.2f}s {2:.2f}s'
                                           .format('filtered' if is_filtered else 'unfiltered',
                                                   raw_['start_time'],
                                                   raw_['end_time'],
                                                   csv_path_hmd))
    unmixed_hmd_fig.savefig('.\\data\\plots\\unmixed_{0}_unmixed_{3}_between_times_{1:.2f}s_{2:.2f}s.jpeg'
                                           .format('filtered' if is_filtered else 'unfiltered',
                                                   raw_['start_time'],
                                                   raw_['end_time'],
                                                   csv_path_hmd))
    source_fig = sources_.plot(scalings={'eeg': 3e2},
                  show=False,
                  show_options=False,
                  title='ICA decomposed series on {0} data between {1:.2f}s-{2:.2f}s'.format('filtered' if is_filtered else 'unfiltered',
                                                                                             raw_['start_time'],
                                                                                             raw_['end_time']))
    source_fig.savefig('.\\data\\plots\\ICA_{0}_between_times_{1:.2f}s-{2:.2f}s_{3}.jpeg'.format(plot_source_path,
                                                                                               raw_['start_time'],
                                                                                               raw_['end_time'],
                                                                                               'filtered' if is_filtered else 'unfiltered'))
    component_fig = ica_.plot_components(title='ICA composition for {0} data between {1:.2f}s-{2:.2f}s'.format('filtered' if is_filtered else 'unfiltered',
                                                                                                               raw_['start_time'],
                                                                                                               raw_['end_time']),
                                                                                                               show=False)
    for i, window in enumerate(component_fig):
        window.savefig('.\\data\\plots\\ICA_{0}_between_times_{1:.2f}s_and_{2:.2f}s_{4}_part_{3}.jpeg'.format(plot_component_path,
                                                                                                       raw_['start_time'],
                                                                                                       raw_['end_time'],
                                                                                                       i,
                                                                                                       'filtered' if is_filtered else 'unfiltered'))
# constants
opts = 'm:n:p'
long_opts = 'method:plot'
method_ = 'fastica'
n_components_ = None
plot_filter = False
l_pass = 0.1
h_pass = 60
notch = 50
missing_signals = ['F9','P9','F10','P10']
csv_path_ref = 'ref'
csv_path_hmd = 'hmd'
plot_source_path = 'sources'
plot_component_path = 'components'

eeg_labels = ['Fp1', 'Fz', 'F3', 'F7', 'F9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'P9', 'O1',
              'Oz', 'O2', 'P10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'C4', 'Cz', 'FC2', 'FC6', 'F10', 'F8', 'F4']
ref_acc_labels = ['ACC_X','ACC_Y','ACC_Z']
hmd_acc_labels = ['LIN_ACC_X','LIN_ACC_Y','LIN_ACC_Z','ANG_ACC_X','ANG_ACC_Y','ANG_ACC_Z']

# Create options
try:
    args, _ = getopt.getopt(sys.argv[1:], opts, long_opts)
    for arg, val in args:
        if arg in ('-m', '--method'): # Method to run ICA with. fastica by default
            method_ = val
        elif arg in '-n': # How many components to run ICA with. If None, look at default description of mne doc for ICA: https://mne.tools/stable/generated/mne.preprocessing.ICA.html
            n_components_ = int(val)
        elif arg in ('-p', '--plot'): # Plots step by step
            plot_filter = True
except getopt.error as err:
    print(str(err))

#Pre-load EEG data
data, header = pyxdf.load_xdf('.\\data\\sub-P001_ses-S001_task-Default_run-001_eeg.xdf')


montage = mne.channels.read_custom_montage(fname='./data/Twist-32_REF.bvef')
if plot_filter:
    montage.plot(show_names=True)

# Creating info objects for the raw objects
info = mne.create_info(
        ch_names=eeg_labels,
        sfreq=data[2]['info']['effective_srate'],
        ch_types='eeg',
        verbose=None)

info['bads'] = missing_signals # Disabling electrodes that are missing from the location data
info.set_montage(montage=montage, on_missing='warn')

ref_acc_info = mne.create_info(
        ch_names=ref_acc_labels,
        sfreq=data[2]['info']['effective_srate'],
        ch_types='misc',
        verbose=None)

hmd_acc_info = mne.create_info(
        ch_names=hmd_acc_labels,
        sfreq=data[1]['info']['effective_srate'],
        ch_types='misc',
        verbose=None)

# Creating raw objects
raw_data = mne.io.RawArray(data=data[2]['time_series'].T[:-5], info=info)
raw_ref_acc = mne.io.RawArray(data=data[2]['time_series'].T[32:35], info=ref_acc_info)
hmd_acc = data[1]['time_series'].T
hmd_acc_raw = mne.io.RawArray(data=hmd_acc[:,:-1],info=hmd_acc_info)

print('hmd lin shape: {0}\nhmd ang shape: {1}\nref shape: {2}'.format(hmd_acc[:3].shape,hmd_acc[3:].shape,raw_ref_acc.get_data().shape))
if plot_filter:
    show_time_series(raw_data, 'Raw time series data')

eeg_referenced = mne.set_eeg_reference(
    inst=raw_data,
    ref_channels=['Cz'],
    copy=True,
    projection=False,
    ch_type='eeg')


if plot_filter:
    show_time_series(eeg_referenced[0], 'referenced data')

#  Split EEG data into epochs
first_eeg_time = data[2]['time_stamps'][0]
last_eeg_time = data[2]['time_stamps'][-1]
first_acc_time = data[1]['time_stamps'][0]
last_acc_time = data[1]['time_stamps'][-1]

# selecting the last time to start and the first time to end
first_time = abs(first_eeg_time - first_acc_time)
last_time = (last_eeg_time if last_eeg_time < last_acc_time else last_acc_time) - first_time


eeg_referenced[0].filter(l_freq=l_pass,
                         h_freq=h_pass,
                         picks='eeg',
                         phase='zero-double',
                         fir_window='hamming')
if plot_filter:
    show_time_series(eeg_referenced[0], 'data filtered with butterworth')

eeg_referenced[0].notch_filter(freqs=notch,
                               picks='eeg',
                               phase='zero-double',
                               fir_window='hamming')
if plot_filter:
    show_time_series(eeg_referenced[0], 'data after butterworth[0.1Hz,60Hz] and notch filter[50Hz]')
    show_time_series(raw_ref_acc,'ref acceleration')

# Resampling the raw data down to the vr headset levels
eeg_referenced[0].resample(sfreq=data[1]['info']['effective_srate'])
raw_ref_acc.resample(sfreq=data[1]['info']['effective_srate'])

eeg_raw = eeg_referenced[0].copy()

print('eeg_referenced size: {0}\nref acc size: {1}'.format(eeg_referenced[0].get_data().shape,
                                                           np.tile(raw_ref_acc.get_data()[0],(31,1)).shape))

##    --- SPLIT RAW---
split_lpass = 1
split_hpass = 10
time_stamps = data[0]['time_stamps'][1:-1] # Removing first and last to get the time that matters in stead of the time between press and release button
raw_list = split_raw(eeg_raw, time_stamps, data[0]['time_stamps'][0])
ref_acc_list = split_raw(raw_ref_acc, time_stamps, data[0]['time_stamps'][0])
hmd_acc_list = split_raw(hmd_acc_raw, time_stamps, data[0]['time_stamps'][0])
print(len(raw_list))
for raw,ref,hmd in zip(raw_list,ref_acc_list,hmd_acc_list):
    # Run correlation check on normal data
    ica = run_ica(eeg_data=eeg_referenced[0],
                         n_components=n_components_,
                         method=method_,
                         plot_filter_=plot_filter)
    ica_sources = ica.get_sources(raw['data'])
    full_correlation_check(ica_=ica,sources_=ica_sources,raw_=raw,ref_=ref,hmd_=hmd, is_filtered=False)

    #Filter raw objects
    ica_sources.filter(l_freq=split_lpass,
                         h_freq=split_hpass,
                         picks='all',
                         phase='zero-double',
                         fir_window='hamming')
    raw['data'].filter(l_freq=split_lpass,
                   h_freq=split_hpass,
                   picks='all',
                   phase='zero-double',
                   fir_window='hamming')
    ref['data'].filter(l_freq=split_lpass,
                   h_freq=split_hpass,
                   picks='all',
                   phase='zero-double',
                   fir_window='hamming')
    hmd['data'].filter(l_freq=split_lpass,
                   h_freq=split_hpass,
                   picks='all',
                   phase='zero-double',
                   fir_window='hamming')
    # Full correlation check on filtered data
    full_correlation_check(ica_=ica, sources_=ica_sources, raw_=raw, ref_=ref, hmd_=hmd, is_filtered=True)