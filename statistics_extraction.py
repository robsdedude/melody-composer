# -*- coding: utf-8 -*-
"""
__author__ = 'kathtn'
__copyright__ = "Fraunhofer IDMT"

"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from collections import defaultdict

from fractions import Fraction

from util import Util
from util import CartesianPowerMathHelper

import numpy as np
from scipy.sparse import lil_matrix as sparse_mat

import logging
log = logging.getLogger("main")


class MelodyStats(object):
    """
    Generates statistics from a training set of 4/4 melodies.
    """
    def __init__(self, pitches, durations, order=1):
        """
        :type pitches: list of lists of ints
        :param pitches: Pitches (ascii encoded) where each inner
            list represents the pitches of a melody.
        :type durations: list of lists of floats
        :param durations: Representing the duration of the corresponding notes
            in pitches parameter.
        :type order: int
        :param order: Specifies an order of the transition matrix that is
            extracted. Order means the number of previous stats that are taken
            into account. Default is 1.

        :returns: MelodyStats object
        """
        self.pitches = pitches
        self.durations = durations
        self.set_of_pitches = list(set([e for l in pitches for e in l]))
        self.set_of_lengths = list(set([e for l in durations for e in l]))
        self.order = order

        # Just some internal storage so that we only have to calculate each
        # statistical value once. Like a cache.
        self._pip = None
        self._ptp = None
        self._pep = None
        self._dip = None
        self._dtp = None
        self._aps = None
        self._als = None

    @property
    def all_pitch_states(self):
        if not self._aps:
            # FIXME: this is not efficient!
            pitch_states_count = 0
            sop_len = len(self.set_of_pitches)
            for exp in range(self.order, 0, -1):
                pitch_states_count += sop_len**exp
            self._aps = [None]*pitch_states_count
            for idx in range(pitch_states_count):
                self._aps[idx] = self._index_to_x_vector(self.set_of_pitches,
                                                         idx)
        return self._aps

    @property
    def all_length_states(self):
        if not self._als:
            # FIXME: this is not efficient!
            length_states_count = 0
            sol_len = len(self.set_of_lengths)
            for exp in range(self.order, 0, -1):
                length_states_count += sol_len**exp
            self._als = [None]*length_states_count
            for idx in range(length_states_count):
                self._als[idx] = self._index_to_x_vector(self.set_of_lengths,
                                                         idx)
        return self._als

    def _calc_probabilities_at_position(self, lists, position, bins=None):
        """
        Counts the nth pitch of the songs then normalize it to get a
        probabilistic distribution of nth pitches. Where n is position argument.
        :param lists: a list of lists to be analyzed
        :param position: the index to count the notes at
        :param bins: the bins are the different elements contained in the lists.
            If not given the method calculates it itself, chooses an order and
            returns it in addition.

        :returns: a normalized list of probabilities (and bins if not specified
            at method call)
        """
        b = bins
        if bins is None:
            b = list(set([e for l in lists for e in l]))
        props = [0]*len(b)
        for list_ in lists:
            props[b.index(list_[position])] += 1
        sum_ = sum(props)
        res = list(map(lambda x: x/sum_, props))
        if bins is not None:
            return res
        else:
            return res, b

    def _x_vector_to_index(self, x, vector, cut_at_order=True):
        """
        Calculate the index in lexicographical order for a vector build ot of a
        given set of elements.

        :type x: list or tuple
        :param x: The (ordered) set of elements the vector is build from
        :type vector: tuple of elements of x
        :param vector: The vector containing only elements of x
        :type cut_at_order: bool
        :param cut_at_order: If set to True only the first n elements of the
            vector will be considered. n is the order passed at init.

        :return: Index int
        """
        len_x = len(x)
        len_v = len(vector)
        index = 0
        for idx in range(len_v-1, -1, -1):
            idx_inv = len_v-idx
            value = len_x**(idx_inv-1)
            index += (x.index(vector[idx])+1)*value
            if cut_at_order and idx_inv >= self.order:
                break
        return index-1

    def _index_to_x_vector(self, x, index):
        """
        Calculate the vector at the index in lexicographical order that is build
        of a given set of elements.

        :type x: list or tuple
        :param x: The (ordered) set of elements the vector is build from
        :type index: int
        :param index: The index in lexicographical order

        :return: a tuple of elements of x (a vector)
        """
        index += 1
        len_x = len(x)
        # calculate needed len of vector
        len_v = CartesianPowerMathHelper.calc_f_inv(index, len_x)[0]
        v = [0]*len_v
        for idx_v in range(len_v):
            inv_idx_v = len_v - idx_v
            r = CartesianPowerMathHelper.calc_f(inv_idx_v, len_x)
            p_r = CartesianPowerMathHelper.calc_f(inv_idx_v-1, len_x)
            d = (r-p_r)/len_x
            m = 1
            while index > p_r+m*d:
                m += 1
            v[idx_v] = x[m-1]
            index -= m*d

        return tuple(v)

    def pitch_vector_to_index(self, pitches, cut_at_order=True):
        """
        Covert a tuple of pitch states to it's according index of the transition
        matrix.

        :type pitches: tuple of ints
        :param pitches: The pitches to convert
        :type cut_at_order: bool
        :param cut_at_order: If set to True only the n most recent states will
            be considered. n is the order passed at init.

        :return: The index (int)
        """
        return self._x_vector_to_index(self.set_of_pitches, pitches,
                                       cut_at_order=cut_at_order)

    def length_vector_to_index(self, lengths, cut_at_order=True):
        """
        Covert a tuple of length states to it's according index of the
        transition matrix.

        :type lengths: tuple of floats
        :param lengths: The lengths to convert
        :type cut_at_order: bool
        :param cut_at_order: If set to True only the n most recent states will
            be considered. n is the order passed at init.

        :return: The index (int)
        """
        return self._x_vector_to_index(self.set_of_lengths, lengths,
                                       cut_at_order=cut_at_order)

    def _calc_pitch_and_length_trans_probs(self):
        """
        Calculates transition probability trajectories for pitch and length.
        first axis is at what count of ab bar (e.g. [1.0, 5.0) in a 4/4 measure)
        second axis is from which state
        third axis is to which state

        :returns: two transition probability trajectories as defaultdicts
            pitch_transition_trajectory, length_transition_trajectory
        """
        # ==== counting pitches and lengths ====
        pitch_count = len(self.set_of_pitches)
        length_count = len(self.set_of_lengths)
        sop_len = CartesianPowerMathHelper.calc_f(self.order,
                                                  len(self.set_of_pitches))
        sol_len = CartesianPowerMathHelper.calc_f(self.order,
                                                  len(self.set_of_lengths))
        pitch_t_tensor = defaultdict(lambda: sparse_mat((sop_len,
                                                         pitch_count),
                                                        dtype=np.float32))
        length_t_tensor = defaultdict(lambda: sparse_mat((sol_len,
                                                          length_count),
                                                         dtype=np.float32))
        for song_index in range(len(self.pitches)):
            count = Fraction(1)  # the position (aka count) in the current bar
            song_length = len(self.pitches[song_index])
            prev_pitches = self.pitches[song_index][:1]
            prev_lengths = self.durations[song_index][:1]
            count += Util.count_to_fraction(self.durations[song_index][0])
            for note_index in range(1, song_length):
                pitch = self.pitches[song_index][note_index]
                length = self.durations[song_index][note_index]
                for i in range(1, min(self.order+1, len(prev_pitches)+1)):
                    prev_pitches_in_scope = prev_pitches[-i:]
                    prev_lengths_in_scope = prev_lengths[-i:]
                    pitch_idx = self.pitch_vector_to_index([pitch])
                    length_idx = self.length_vector_to_index([length])
                    prev_pitches_idx = self.pitch_vector_to_index(
                        prev_pitches_in_scope)
                    prev_lengths_idx = self.length_vector_to_index(
                        prev_lengths_in_scope)
                    pitch_t_tensor[count][prev_pitches_idx, pitch_idx] += 1
                    length_t_tensor[count][prev_lengths_idx, length_idx] += 1
                prev_pitches.append(pitch)
                prev_lengths.append(length)
                # going on in the bar being aware of waring count at the end of
                # the bar
                count = Util.count_to_fraction(((count + length - 1) % 4) + 1)
        # ==== normalizing the counts to be a probabilities distribution ====
        assert (sorted(pitch_t_tensor.keys()) ==
                sorted(length_t_tensor.keys()))
        for key in pitch_t_tensor.keys():
            nz = pitch_t_tensor[key].nnz
            pitch_t_tensor[key] = Util.normalize_lil_mat_by_rowsum(
                pitch_t_tensor[key])
            assert (nz == pitch_t_tensor[key].nnz and
                    "Too much training data, float flipped to zero.")
            nz = length_t_tensor[key].nnz
            length_t_tensor[key] = Util.normalize_lil_mat_by_rowsum(
                length_t_tensor[key])
            assert (nz == length_t_tensor[key].nnz and
                    "Too much training data, float flipped to zero.")
        return pitch_t_tensor, length_t_tensor

    def calc_pitch_initial_probabilities(self):
        """
        Counts the starting pitch of the songs then normalize it to get a
        probabilistic distribution of starting pitches.

        :returns: a normalized list of probabilities
        """
        if self._pip:
            return self._pip
        return self._calc_probabilities_at_position(self.pitches, 0,
                                                    self.set_of_pitches)

    def calc_pitch_transition_probabilities(self):
        """
        Calculates transition probability trajectory for pitch of notes.
        first axis is at what count of ab bar (e.g. [1.0, 5.0) in a 4/4 measure)
        second axis is from which state
        third axis is to which state

        :return: returns the trajectory as defaultdict
        """
        if self._ptp:
            return self._ptp
        self._ptp, self._dtp = self._calc_pitch_and_length_trans_probs()
        return self._ptp

    def calc_pitch_end_probabilities(self):
        """
        Counts the ending pitch of the songs then normalize it to get a
        probabilistic distribution of ending pitches.

        :returns: a normalized list of probabilities
        """
        if self._pep:
            return self._pep
        return self._calc_probabilities_at_position(self.pitches, -1,
                                                    self.set_of_pitches)

    def calc_duration_initial_probabilities(self):
        """
        Counts the starting note' length of the songs then normalize it to get a
        probabilistic distribution of durations.

        :returns: a normalized list of probabilities
        """
        if self._pip:
            return self._pip
        return self._calc_probabilities_at_position(self.durations, 0,
                                                    self.set_of_lengths)

    def calc_duration_transition_probabilities(self):
        """
        Calculates transition probability trajectory for length of notes.
        first axis is at what count of ab bar (e.g. [1.0, 5.0) in a 4/4 measure)
        second axis is from which state
        third axis is to which state

        :return: returns the trajectory as defaultdict
        """
        if self._dtp:
            return self._dtp
        self._ptp, self._dtp = self._calc_pitch_and_length_trans_probs()
        return self._dtp
