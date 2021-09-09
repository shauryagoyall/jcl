import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


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


def concatenate_spike_times(*all_spike_times):
    """ Concatenate spike times from different sessions (shift appropriately).

        Args:
            all_spike_times - spike times (list of lists) for every session
        Return:
            concatenated spike times (list of lists)
    """
    def __get_last_spike_time(spike_times):
        return np.max([st[-1] for st in spike_times])

    unit_nums = np.array([len(st) for st in all_spike_times])
    if not np.all(unit_nums == unit_nums[0]):
        raise ValueError("All given spike times (sessions) must have equal number of units.")

    last_spike_times = [__get_last_spike_time(st) for st in all_spike_times]
    session_shifts = [0] + np.cumsum(last_spike_times)[:-1].tolist()
    assert len(session_shifts) == len(all_spike_times)
    cat_spike_times = []

    for u in range(unit_nums[0]):
        all_st_u = [np.array(st[u]) for st in all_spike_times]
        all_st_u_shifted = [all_st_u[i] + session_shifts[i] for i in range(len(all_st_u))]
        cat_spike_times.append(np.concatenate(all_st_u_shifted))

    return cat_spike_times

# TODO unit tests
"""
# quick test

x = [1, 2, -1, 4, 5, 6]
y = [-1, 2, 3, -1, 5, -1]

x_i, y_i = interpolate_position(x, y)


#pv, pp, ss = align_measured_data(fr, p, 4, speed=s)
"""
