import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter


def align_measured_data(population_vectors, positions, num_bins, speed=None):
    l = positions.shape[0] // num_bins
    if l < population_vectors.shape[1]:
        pv = population_vectors[:, 0:l]
        pos = positions[0:l*num_bins]
        s = speed[0:l*num_bins] if speed is not None else None
    else:
        ll = population_vectors.shape[1] * num_bins
        pv = population_vectors
        pos = positions[0:ll]
        s = speed[0:ll] if speed is not None else None

    sh = (-1, num_bins, 2) if pos.ndim == 2 else (-1, num_bins)
    pos = pos.reshape(sh).mean(axis=1)
    s = s.reshape(-1, num_bins).mean(axis=1) if speed is not None else None

    if s is not None:
        return pv, pos, s
    else:
        return pv, pos

def __interpolate_linear_position(position, unknown_val=-1, interp_order=1):
    """ Spline interpolation of missing values in animal position data.

        Args:
            position - coordinates
            unknown_val - placeholder for unknown/missing values
            interp_order - interpolation polynom degree, default 1 (linear)
        Return:
            Interpolated data in the same format as provided.
    """
    position = np.array(position)
    ok_inds = np.where(position != unknown_val)[0]
    f = InterpolatedUnivariateSpline(ok_inds, position[ok_inds], k=interp_order)
    return f(np.arange(len(position)))

def interpolate_position(x, y=None, unknown_val=-1, interp_order=1):
    """ Spline interpolation of missing values in animal position data.
        In case of 2d data (y not None), each dimension is interpolated independently.

        Args:
            x - coordinates
            y - coordinates (for 2d data only)
            unknown_val - placeholder for unknown/missing values
            interp_order - interpolation polynom degree, default 1 (linear)
        Return:
            Interpolated data in the same format as provided.
    """
    x_i = __interpolate_linear_position(x, unknown_val, interp_order)
    if y is not None:
        y_i = __interpolate_linear_position(y, unknown_val, interp_order)
        return x_i, y_i
    return x_i


def trial_duration(xy_trajectory, sampling_rate=39.0625):
    """ Compute trial duration in seconds based on given trajectory and sampling rate.

    Args:
        xy_trajectory - animal trajectory during a trial, array of shape (n,2)
        sampling_rate - number of samples per second (Hz), default 39.0625
    Return:
        Duration in seconds
    """
    return len(xy_trajectory) / sampling_rate  # in seconds


def trial_distance(xy_trajectory):
    """ Compute trial duration based on given trajectory.

    Args:
        xy_trajectory - animal trajectory during a trial, array of shape (n,2)
    Return:
        Distance covered by the trajectory (in the same units as xy_trajectory)
    """
    diffs = np.diff(xy_trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return distances.sum()


def calc_speed(positions, bin_len):
    """ Calculate animal speed. Units of positions / s.

        Args:
            positions - array of positions of shape (n,1) or (n,2)
            bin_len - duration of temporal bins in ms
        Return:
            speed - animal speed in each recorded point, same shape as positions
    """
    speed = []
    for i in range(1, len(positions)):
        prev_p = positions[i-1]
        current_p = positions[i]
        dist = np.linalg.norm(current_p-prev_p)
        speed.append(dist / (bin_len / 1000))
    return [speed[0]] + speed


def speed_filter(positions, bin_len, speed_thr, speed_violation = 90):
    """ Filter positions based on speed.
        Return both parts where speed > speed_thr and where speed <= speed_thr.

            positions - array of positions of shape (n,1) or (n,2)
            bin_len - duration of temporal bins in ms
            speed_thr - threshold for speed
            speed_violation - speed in cm/s that is max achievable by rat
        Return:
            positions speed filtered with low speed set to 0, low speed position indices
    """
    assert speed_thr is not None
    
    window_size = 50
    poly_degree = 1
    # Apply the Savitzky-Golay filter
    smooth_pos = position.copy()
    smooth_pos[:,0] = savgol_filter(position[:,0], window_size, poly_degree)
    smooth_pos[:,1] = savgol_filter(position[:,1], window_size, poly_degree)
    
    speed = np.array( calc_speed(smooth_pos,bin_len) )
    
    speed_violation_index = np.where(speed > speed_violation)[0]
    for i in speed_violation_index:
        speed[i]= np.mean(speed[i-10:i])
        
    # Window size and polynomial degree for the Savitzky-Golay filter
    window_size = 50
    poly_degree = 1
    # Apply the Savitzky-Golay filter
    smooth_speed = savgol_filter(speed, window_size, poly_degree)
    speed_l_idx = smooth_speed < speed_thr
    
    speed_filtered_positions = smooth_pos.copy()
    speed_filtered_positions[speed_l_idx] = [0,0]
    return speed_filtered_positions, speed_l_idx, smooth_speed,speed, smooth_pos


def get_last_spike_time(spike_times):
    """ Get the timestamp of the last spike.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
        Return:
            Last spike timestamp (int)
    """
    return np.max([st[-1] for st in spike_times if len(st) > 0])


def get_first_spike_time(spike_times):
    """ Get the timestamp of the first spike.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
        Return:
            First spike timestamp (int)
    """
    return np.min([st[0] for st in spike_times if len(st) > 0])


def compute_bins(position_range, bin_len):
    """ Compute bin edges for given spike times.

        Args:
            position_range - tuple or list with first position timestamp and last position time stamp
            bin_len - length of a bin in ms (default 1s/50 = 20ms which is 480 samples of spikes)
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
        Return:
            List of bin edges (in ms), total number of bins
    """
    
    bin_num = np.ceil( (position_range[-1] - position_range[0]) / bin_len).astype(int)
    bin_edges = np.arange(position_range[0], position_range[-1]+bin_len, bin_len)
    
    return bin_edges, bin_num

def concatenate_spike_times(*all_spike_times, shift_sessions=True):
    """ Concatenate spike times from different sessions (shift appropriately).

        Args:
            all_spike_times - spike times (list of lists) for every session
            shift_sessions - if True (default) shift spike times for every session in `all_spike_times`
        Return:
            concatenated spike times (list of lists)
    """

    unit_nums = np.array([len(st) for st in all_spike_times])
    if not np.all(unit_nums == unit_nums[0]):
        raise ValueError("All given spike times (sessions) must have equal number of units.")

    last_spike_times = [get_last_spike_time(st) for st in all_spike_times]
    if shift_sessions:
        session_shifts = [0] + np.cumsum(last_spike_times)[:-1].tolist()
    else:
        session_shifts = [0 for i in range(len(all_spike_times))]
    assert len(session_shifts) == len(all_spike_times)

    cat_spike_times = []

    for u in range(unit_nums[0]):
        all_st_u = [np.array(st[u]) for st in all_spike_times]
        all_st_u_shifted = [all_st_u[i] + session_shifts[i] for i in range(len(all_st_u))]
        cat_spike_times.append(np.concatenate(all_st_u_shifted))

    return cat_spike_times

def to_ms(x, sampling_period):
    """ Convert time `x` to ms.

        Args:
            x - time to be converted (number or array of numbers)
            sampling_period - sampling period in ms
        Return:
            Converted data of the same type as `x`
    """
    return x * sampling_period

def __elapsed(st, sampling_period):
    """ Compute elapsed time (in s) between the first and the last spike in given list of spike times."""
    return None if len(st) <= 0 else sampling_period * (st[-1]-st[0]) / 1000

def compute_frs(spike_times, sampling_period=0.05):
    """ Compute firing rate (in Hz) for each neuron in given spike times.

        Args:
            spike_times - list of spike times per neuron (list of iterables, pre-sorted in a non-descending order)
            sampling_period - sampling period in ms (default 1s/20kHz = 0.05ms)
        Return:
            List of firing rates.
    """
    return [len(st) / __elapsed(st, sampling_period) if len(st) > 0 else 0 for st in spike_times]

def generate_sliding_windows(bins, window_len, stride=None):
    """ Generate sliding windows over given bins.

        Args:
            bins - matrix with binned data
            window_len - length of each window in number of bins
            stride - stride between windows (defaults to window_len)
        Return:
            Generator with slices of bins.
    """
    if stride is None:
        stride = window_len
    for start in range(0, bins.shape[1], stride):
        yield bins[:, start:start+window_len]

def traj_in_interval_1D(traj, interval):
    """ Check which points of the trajectory are in the given interval.

        Args:
            traj - trajectory, array with `n` elements
            interval - (begin, end), inclusive
        Return:
            True / False array with `n` elements
    """
    return (traj >= interval[0]) & (traj <= interval[1])

def traj_in_rect_2D(traj, rect):
    """ Check which points of the trajectory are in the given rect.

        Args:
            traj - trajectory, array of shape (n, 2)
            rect - coordinates of the rect [(TL_x, TL_y), (BR_x, BR_y)]

        Return:
            True / False array with `n` elements
    """
    (TL_x, TL_y), (BR_x, BR_y) = rect
    x_idx = traj_in_interval_1D(traj[:, 0], (TL_x, BR_x))
    y_idx = traj_in_interval_1D(traj[:, 1], (BR_y, TL_y))
    return np.logical_and(x_idx, y_idx)

def traj_in_circle_2D(traj, center, radius):
    """ Check which points of the trajectory are in the given circle.

        Args:
            traj - trajectory, array of shape (n, 2)
            center - coordinates of the center (c_x, c_y)
            radius - circle radius

        Return:
            True / False array with `n` elements
    """
    return np.linalg.norm(traj - np.array(center), axis=1) ** 2 <= radius ** 2

def res_clu_from_spike_times(st):
    """ Generate res and clu for given spike times.

        Args:
            st - spike times, list of lists

        Return:
            res, clu - np.ndarrays
    """
    clu = np.concatenate([len(st_c) * [c] for c, st_c in enumerate(st)])
    res = np.concatenate(st).astype(np.int64)
    sort_idx = np.argsort(res)
    return res[sort_idx], clu[sort_idx]

# TODO unit tests


"""
# quick test

x = [1, 2, -1, 4, 5, 6]
y = [-1, 2, 3, -1, 5, -1]

x_i, y_i = interpolate_position(x, y)


#pv, pp, ss = align_measured_data(fr, p, 4, speed=s)
"""
