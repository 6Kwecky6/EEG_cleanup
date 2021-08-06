import pyxdf
import mne
import getopt
import sys


def show_time_series(eeg_data, title):
    eeg_data.plot(scalings={'eeg': 1e2},
                  show=True,
                  title=title,
                  block=True)


def run_ica(eeg_data, n_components, method):
    ica = mne.preprocessing.ICA(n_components=n_components,
                                method=method)
    ica.fit(eeg_data, picks='eeg')
    ica.plot_components(title='ICA composition')


opts = ['m', 'n', 'f']
long_opts = ['method', 'fast']
method_ = 'fastica'
n_components_ = None
plot_filter = True

try:
    args, _ = getopt.getopt(sys.argv[1:], opts, long_opts)
    for arg, val in args:
        if arg in ('-m', '--method'):
            method_ = val
        elif arg in '-n':
            n_components_ = int(val)
        elif arg in ('-f', '--fast'):
            plot_filter = False
except getopt.error as err:
    print(str(err))
data, header = pyxdf.load_xdf('.\\data\\sub-P001_ses-S001_task-Default_run-001_eeg.xdf')

eeg_labels = ['Fp1', 'Fz', 'F3', 'F7', 'F9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'P9', 'O1',
              'Oz', 'O2', 'P10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'C4', 'Cz', 'FC2', 'FC6', 'F10', 'F8', 'F4']
info = mne.create_info(
        ch_names=eeg_labels,
        sfreq=data[2]['info']['effective_srate'],
        ch_types='eeg',
        verbose=None)
raw_data = mne.io.RawArray(data=data[2]['time_series'].T[:-5], info=info)
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

eeg_referenced[0].filter(l_freq=0.1,
                         h_freq=60,
                         picks='eeg',
                         fir_window='hamming')
if plot_filter:
    show_time_series(eeg_referenced[0], 'data filtered with butterworth')

eeg_referenced[0].notch_filter(freqs=50.,
                               picks='eeg',
                               fir_window='hamming')
if plot_filter:
    show_time_series(eeg_referenced[0], 'data after butterworth[0.1Hz,60Hz] and notch filter[50Hz]')

run_ica(eeg_data=eeg_referenced[0], n_components=n_components_, method=method_)
