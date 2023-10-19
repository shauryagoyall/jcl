import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from jcl.utils import get_last_spike_time, to_ms, compute_bins, compute_frs

def readfromtxt(file_path, conv_fun=str):
    """ Read lines from a text file into a list.

        Args:
            file_path - path to the text file
            conv_fun - conversion to be done on a line
        Return:
            List of elements converted using conv_fun
    """
    with open(file_path) as f:
        lines = [conv_fun(l) for l in f]
    return lines


def spike_times_from_res_and_clu(res_path, clu_path, exclude_clusters=[0, 1], ts_limits=None):
    """ Load spike times for each neuron from provided '.res' and '.clu' files.

        Args:
            res_path - path to res file (sorted in a non-descending order)
            clu_path - path to clu file
            exclude_clusters - clusters to exclude (by default exclude noise clusters: 0 - artifacts and 1 - unassigned spikes)
            ts_limits - timestamps used to cut out only a temporal section of data, tuple (begin_ts, end_ts)
        Return:
            List of firing times for each cell (list of np.arrays)
    """
    # .clu and .res files are big
    # so we use our faster implementation
    # instead of np.genfromtxt
    clu = np.array(readfromtxt(clu_path, conv_fun=int))
    res = np.array(readfromtxt(res_path, conv_fun=int))
    assert len(clu) == len(res) or len(clu) == len(res) + 1

    if len(clu) == len(res) + 1:
        # number of clusters is written in the first line
        num_clusters = clu[0]
        clu = clu[1:]
    else:
        num_clusters = np.max(clu)

    clusters = np.arange(num_clusters)
    clusters = [res[clu == c] for c in clusters if c not in exclude_clusters]
    if ts_limits is not None:
        return slice_spike_times(clusters, *ts_limits)
    return clusters


def make_des(res_path, clu_path, thr=5, sampling_period=0.05):
    """ Determine cell type ('p1' or 'b1') based of firing rates.
        This function assumes that all recorded cells are from the hippocampus.

        Args:
            res_path - path to res file (sorted in a non-descending order)
            clu_path - path to clu file
            thr - firing rate threshold (if thr > 5 -> 'b1', else 'p1')
        Return:
            List of cell types
    """
    spike_times = spike_times_from_res_and_clu(res_path, clu_path)
    frs = compute_frs(spike_times, sampling_period)
    return ["b1" if fr > thr else "p1" for fr in frs]


def slice_spike_times(spike_times, begin_ts, end_ts):
    """ Return only spike times between given timestamps.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            begin_ts - spike times greater or equal than this
            end_ts - spikes times less than this

        Return:
            Sliced spike times (list of np.arrays)
    """
    if begin_ts <= 0 and end_ts >= get_last_spike_time(spike_times):
        # no need to iterate over spike_times
        # if the given range contains all of them
        return spike_times

    def __get_inds(st, b, e):
        g = np.where(st >= b)
        l = np.where(st < e)
        return np.intersect1d(g, l)

    return [st[__get_inds(st, begin_ts, end_ts)] if len(st) > 0 else st for st in spike_times]


def bins_from_spike_times(spike_times, position_range, bin_len=480, sampling_period=1/480, dtype=np.uint16, dense_loading=True, return_mat_type=np.array):
    """ Bin given spike times, each bin contains total number of spikes.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            position_range - tuple or list with first position timestamp and last position time stamp
            bin_len - length of a bin in ms (default 1s/50 = 20ms)
            sampling_period - sampling period in ms (default 1s/24kHz)
            dtype - dtype to use for bins, default np.uint16 (np.uint8 would use less memory, but can store only up to 256 spikes per bin)
            dense_loading - if True (default) load data into a dense np.ndarray then convert to a `return_mat_type` (fast), if False load into a sparse matrix (slow, but memory efficient)
            return_mat_type - type of matrix to be returned (default is sparse `csc_matrix` for efficient storage and relatively fast column slicing)
        Return:
            Matrix with spike count per bin ((neuron num, time bins), `return_mat_type`)
    """
    # leave this two lines here in case we find non-sorted spikes (.res files)
    # maxes = [np.max(st) if len(st) > 0 else 0 for st in spike_times]
    # last_spike_time = to_ms(np.max(maxes), sampling_period)  # in ms
    bin_edges, bin_num = compute_bins(position_range, bin_len, sampling_period)

    if dense_loading:
        # fastest loading (due to indexing), but requires a lot of memory
        binned_data = np.empty((len(spike_times), bin_num), dtype=np.uint16)
    else:
        # lil_matrix is fast for loading rows, convert later to return_mat_type
        binned_data = lil_matrix((len(spike_times), bin_num), dtype=np.uint16)
#     print((bin_edges.shape), binned_data.shape)
    for n, st in enumerate(spike_times):
#         st_ms = to_ms(np.array(st), sampling_period)
        hist = np.histogram(st, bins=bin_edges)[0]
        binned_data[n] = hist

    return return_mat_type(binned_data)


def bins(res_path, clu_path, bin_len=25.6, sampling_period=0.05, dtype=np.uint16, dense_loading=True, return_mat_type=csc_matrix):
    """ Bin spike times given '.res' and '.clu' files, each bin contains total number of spikes (without noise clusters).

        Args:
            res_path - path to res file
            clu_path - path to clu file
            bin_len - length of a bin in ms (default 1s/39.0625 = 25.6ms)
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
            dtype - dtype to use for bins, default np.uint16 (np.uint8 would use less memory, but can store only up to 256 spikes per bin)
            dense_loading - if `return_dense` is True this is ignored and only dense np.ndarray will be used; otherwise: if True (default) load data into a dense array then convert to a sparse matrix (fast), if False load straight into a sparse matrix (requires less memory, but significantly slower)
            return_mat_type - if `return_dense` is True this is ignored; otherwise: type of sparse matrix to be returned (default `csc_matrix` for fast column slicing)
        Return:
            Matrix of shape (neuron num, time bins) with spike count in each bin (np.array or csc_matrix, depending on `return_dense`)
    """
    st = spike_times_from_res_and_clu(res_path, clu_path)
    return bins_from_spike_times(st, bin_len, sampling_period, dense_loading, return_mat_type)


def cell_types(des_path):
    """ Read cell types from '.des' file.
            'p1' - CA1 pyramidal
            'pu' - unknown pyramidal
            'b1' - CA1 interneuron
            'bu' - unknown interneuron

        Args:
            des_path - path to des file
        Return:
            Array with cell types (dtype=str)
    """
    return np.genfromtxt(des_path, dtype=str)


def positions_from_whl(whl_path):
    """ Load animal positions from '.whl' file.

        Args:
            whl_path - path to whl file
        Return:
            Array with positions (N, led * 2)
    """
    return np.genfromtxt(whl_path)


def positions_and_speed_from_whl2(whl2_path):
    """ Load animal positions and speed from '.whl2' file.

        Args:
            whl2_path - path to whl2 file
        Return:
            Array with positions (N, 2), array with speed (N)
    """
    d = np.genfromtxt(whl2_path)
    return d[:, [0, 1]], d[:, 2]


def session_limits(resofs_path):
    """ Read session limits from a .resofs file.

        Args:
            resofs_path - path to the .reso file
        Return:
            session limits (list of tuples)
    """
    resofs = readfromtxt(resofs_path, int)
    num_sessions = len(resofs)
    resofs = [0] + resofs
    return [(resofs[i], resofs[i+1]) for i in range(num_sessions)]


def sw(sw_path):
    """ Read sharp wave timestamps from a .sw file.

        Args:
            sw_path - path to the .sw file
        Return:
            SW timestamps (list of lists)
    """
    return readfromtxt(sw_path,
                       lambda l: [int(ts) for ts in l.strip().split()])


# TODO unit tests


"""
st = [np.array([1,3,5,7,9,11]), np.array([2,4,6,8,10,12,14])]
sts = slice_spike_times(st, 4, 14)
"""
