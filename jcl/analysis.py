from functools import cached_property
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cosine


class Binning:
    def __init__(self, bin_size, maze_size):
        self.bin_size = bin_size
        self.maze_size = maze_size

    def bin_idx(self, p):
        """ Calculate bin index for given position.

            Args:
                p - position
                bin_size - size of bins, single number or same shape as `p`
                maze_size - size of the maze (number of bins per dimension)
            Return:
                bin index - tuple of the same shape as `p`
        """
        idx = tuple((p // self.bin_size).astype(int))
        # origin to lower left corner
        return (self.num_bins[1] - idx[1] - 1, idx[0])

    @cached_property
    def num_bins(self):
        """ Calculate number of spatial bins (per dimension). """
        return 1 + (np.array(self.maze_size) / self.bin_size).astype(int)


class Map:
    def __init__(self, mmap):
        self.__map = mmap
        self.__map_prob = None

    @property
    def map(self):
        """ Return the underlying map. """
        return self.__map

    @property
    def map_prob(self):
        """ Return the underlying map normalized to sum of 1. """
        if self.__map_prob is None:
            self.__map_prob = self.__map / self.__map.sum()
        return self.__map_prob

    # a map is an array
    # so we implement methods that are
    # absolutely essential for an array
    def __getitem__(self, idx):
        return self.__map.__getitem__(idx)

    @property
    def ndim(self):
        return self.__map.ndim

    @property
    def shape(self):
        return self.__map.shape

    @property
    def size(self):
        return self.__map.size

class OccupancyMap(Map):
    def __init__(self, positions, maze_size, bin_size, bin_len, smooth_sd=None):
        """ Produce occupancy map.

            Args:
                positions - array of positions of shape (n,1) or (n,2)
                maze_size - size of the maze in the same units as positions
                bin_size - size of spatial bins in the same units as positions
                bin_len - duration of temporal bins in ms
                smooth_sd - SD in bins for gaussian smoothing
            Return:
                occupancy - time in seconds spent in each spatial bin
        """
        self.binner = Binning(bin_size, maze_size)
        super().__init__(self.__compute_occ(positions, self.binner, bin_len, smooth_sd))
        self.bin_size = bin_size

    @staticmethod
    def __compute_occ(positions, binner, bin_len, smooth_sd=None):
        occupancy = np.zeros(binner.num_bins, dtype= np.int64)

        for p in positions:
            
            bin_idx = binner.bin_idx(p)
            num_bins = binner.num_bins - 1
            if ( abs( bin_idx[0] - bin_idx[1] )   !=   num_bins[0] ) and ( abs( bin_idx[0] - bin_idx[1] )   !=  0 ):
                occupancy[bin_idx] += bin_len
        if smooth_sd is not None:
            occupancy = gaussian_filter(occupancy, smooth_sd)

        return occupancy / 1000  # ms to seconds



class FiringRateMap(Map):
    def __init__(self, spike_train, positions, maze_size, bin_size, bin_len, smooth_sd=3):
        """ Produce firing rate map for the given single cell spike train.

            Args:
                positions - array of positions of shape (n,1) or (n,2)
                maze_size - size of the maze in the same units as positions
                bin_size - size of spatial bins in the same units as positions
                bin_len - duration of temporal bins in ms
                smooth_sd - SD in bins for gaussian smoothing
            """
        __fr_map, __occupancy = self.__compute_frm(spike_train, positions, maze_size, bin_size, bin_len, smooth_sd=3, return_occupancy=True)
        super().__init__(__fr_map)
        self.__occupancy = __occupancy
        self.bin_size = bin_size
        self.__eps = 1e-15
        self.__I_sec = None
        self.__I_spike = None
        self.__sparsity = None
        self.__frs = None # mean, median, max

    @property
    def occupancy(self):
        """ Occupancy map. """
        return self.__occupancy

    @property
    def peak_fr(self):
        """ Peak firing rate in Hz. """
        if self.__frs is None:
            self.__frs = self.__compute_frs()
        return self.__frs[2]

    @property
    def mean_fr(self):
        """ Mean firing rate in Hz. """
        if self.__frs is None:
            self.__frs = self.__compute_frs()
        return self.__frs[0]

    @property
    def I_sec(self):
        """ Information of the firing rate map in bits/s. """
        if self.__I_sec is None:

            good = np.logical_and(self.__good_idx(self.map, self.__eps), self.__good_idx(self.__occupancy.map, self.__eps))
            frm = self.map[good]

            occ = self.__occupancy.map_prob[good]

            self.__I_sec = np.sum(frm * np.log2(frm / self.mean_fr) * occ)
        return self.__I_sec


    @property
    def I_spike(self):
        """ Information of the firing rate map in bits/spike. """
        if self.__I_spike is None:
            self.__I_spike = self.I_sec / self.mean_fr
        return self.__I_spike

    @property
    def sparsity(self):
        """ Sparsity of the firing rate map. """
        if self.__sparsity is None:
            occ_prob = self.__occupancy.map_prob
            frm_good_idx = self.__good_idx(self.map)
            occ_good_idx = self.__good_idx(occ_prob)
            good_idx = np.logical_and(frm_good_idx, occ_good_idx)

            frm = self.map[good_idx]
            occ = occ_prob[good_idx]

            self.__sparsity = np.sum(frm * occ) ** 2 / np.sum(frm ** 2 * occ)
        return self.__sparsity

    @cached_property
    def center(self):
        if (self.map != 0).sum() == 0:
            return None
        return np.squeeze(np.where(self.map == self.peak_fr))

    def correlate(self, other: Map, normalized=False):
        self_ok = self.__good_idx(self.map) & (self.occupancy.map > 0)
        other_ok = self.__good_idx(other.map) & (other.occupancy.map > 0)
        both_ok = (self_ok) & (other_ok)
        if normalized:
            sm = self.map_prob[both_ok].flatten()
            om = other.map_prob[both_ok].flatten()
        else:
            sm = self.map[both_ok].flatten()
            om = other.map[both_ok].flatten()
        return np.corrcoef(sm, om)[0, 1]

    def __compute_frs(self):
        """ Compute mean, median and max firing rates from given firing rate map.
            Ignore nans, infs and values smaller than eps.

            Return:
                (mean, median, max) FR in Hz
        """
        not_nan = np.logical_not(np.isnan(self.map))
        not_inf = np.logical_not(np.isinf(self.map))
        good = np.logical_and(not_nan, not_inf)
        good = np.logical_and(good, self.map >= self.__eps)
        frm = self.map[good]
        if frm.size == 0:
            return 0, 0, 0
        return frm.mean(), np.median(frm), frm.max()

    @staticmethod
    def __compute_frm(spike_train, positions, maze_size, bin_size, bin_len, smooth_sd=3, return_occupancy=False):
        """ Produce firing rate map for the given single cell spike train.

            Args:
                positions - array of positions of shape (n,1) or (n,2)
                maze_size - size of the maze in the same units as positions
                bin_size - size of spatial bins in the same units as positions
                bin_len - duration of temporal bins in ms
                smooth_sd - SD in bins for gaussian smoothing
                return_occupancy - whether to return occupancy map
            Return:
                firing rate map - matrix with firing rate (Hz) in each spatial bin
        """
        assert len(spike_train) == len(positions)
        occupancy = OccupancyMap(positions, maze_size, bin_size, bin_len, smooth_sd)
        frm = np.zeros_like(occupancy.map)

        for p, sn in zip(positions, spike_train):
            bin_idx = occupancy.binner.bin_idx(p)
            frm[bin_idx] += sn

        frm = frm / occupancy.map
        frm[np.isnan(frm)] = 0
        frm[np.isinf(frm)] = 0
        frm = gaussian_filter(frm, smooth_sd)
        return frm, occupancy if return_occupancy else frm

    @staticmethod
    def __good_idx(m, eps=None):
        """ Get indices of elements that are not Nan, Inf and that are >= `eps`. """
        not_nan = np.logical_not(np.isnan(m))
        not_inf = np.logical_not(np.isinf(m))
        good = np.logical_and(not_nan, not_inf)
        if eps is not None:
            return np.logical_and(good, m >= eps)
        else:
            return good


class PopulationVectors:
    def __init__(self, fr_maps):
        """ Get population vectors from provided firing rate maps.

            Args:
                fr_maps - list of FiringRateMap, all of the same shape
            Return:
                population vectors - np.array of shape (*(fr_maps[0].shape), len(fr_maps))
        """
        self.pvs = np.stack([frm.map for frm in fr_maps], axis=-1)

    @property
    def spatial_shape(self):
        return self.pvs.shape[:-1]

    def measure(self, other, measure_fun):
        """ Compute measure `measure_fun` between all pairs of population vectors.

            Args:
                other - other population vectors
                measure_fun - function that accepts two vectors and returns a number
            Return:
                list of measure values
        """
        assert self.pvs.shape == other.pvs.shape
        spatial_shape = self.pvs.shape[:-1]
        measure = np.array([measure_fun(self.pvs[idx], other.pvs[idx]) for idx in np.ndindex(spatial_shape)])
        return measure[~np.isnan(measure)]

    def correlation(self, other):
        """ Compute Pearson correlation coefficients between all pairs of population vectors.

            Args:
                other - other population vectors
            Return:
                correlation coefficients - list of correlation coefficients
        """
        return self.measure(other, lambda pv1, pv2: np.corrcoef(pv1, pv2)[0,1])

    def cosine_distance(self, other):
        """ Compute correlation coefficients between all pairs of population vectors.

            Args:
                other - other population vectors
            Return:
                list of cosine distances
        """
        return self.measure(other, cosine)

