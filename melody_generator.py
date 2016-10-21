# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from math import exp
from math import pi
from math import sqrt

from collections import defaultdict
from fractions import Fraction

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from random import shuffle

from util import Util
from util import NoteNumbers
from statistics_extraction import MelodyStats
from mm import MarkovModel
from mm import NotEnoughTrainingData
from contour import MelodyContour
from contour import RhythmContour

import logging
log = logging.getLogger("main")

__author__ = 'rouven'


class MelodyGenerator(object):
    """
    This class defines objects, that, after giving them statistics of training
    data, can create melodies by certain constraints.
    """
    def __init__(self, songs, mm_order=4):
        """
        :type songs: list of lists of lists of pairs
        :param songs: each list stand for a song which contains phrases which
            are lists of pairs (pitch, length).
        :type mm_order: int
        :param mm_order: Specifies the order of the markov model to use.

        :return: object of MelodyGenerator
        """
        self.order = mm_order
        log.debug("Using Markov model of order {}".format(mm_order))
        self.songs = songs
        self.phrases = [phrase for song in songs for phrase in song]
        self.pitches = [[note[0] for phrase in song for note in phrase]
                        for song in songs]
        self.lengths = [[note[1] for phrase in song for note in phrase]
                        for song in songs]

        # calling learn() will fill out these attributes
        self._stats = None
        self.index2ascii = {}
        self.ascii2index = {}
        self.index2length = {}
        self.length2index = {}
        self._pitch_mm_tensor = None
        self._duration_mm_tensor = None
        self._melody_contour = MelodyContour()
        self._rhythm_contour = RhythmContour()

    def learn(self, n_cluster=17):
        """
        Extract statistical information from training data.

        This function must be called to train the model. You cannot call
        generate_melody before this one.

        :type n_cluster: int
        :param n_cluster: the number of phrase contour clusters build.
            s. contour.py

        :return: None
        """
        def _remap_transition_tensor(tensor):
            # remaps the tensor (dict) with first axis elem. [1, 4) to a tensor
            # with first axis elem. [1, 2)
            new_tensor = {}
            for count in tensor:
                offset = Util.count_to_fraction(count % 1 + 1)
                if offset in new_tensor:
                    new_tensor[offset] = (new_tensor[offset] +
                                          csc_matrix(tensor[count]))
                else:
                    new_tensor[offset] = csc_matrix(tensor[count])
            for offset in new_tensor:
                new_tensor[offset] = Util.normalize_lil_mat_by_rowsum(
                    lil_matrix(new_tensor[offset]))
            return new_tensor

        self._stats = MelodyStats(self.pitches, self.lengths, order=self.order)
        log.debug("Starting statistical extraction.")
        pitch_tensor = self._stats.calc_pitch_transition_probabilities()
        duration_tensor = self._stats.calc_duration_transition_probabilities()
        log.debug("Remapping tensors.")
        pitch_tensor = _remap_transition_tensor(pitch_tensor)
        duration_tensor = _remap_transition_tensor(duration_tensor)
        log.debug("Done.")

        for i, pitch in enumerate(self._stats.set_of_pitches):
            self.index2ascii[i] = pitch
            self.ascii2index[pitch] = i
        for i, length in enumerate(self._stats.set_of_lengths):
            self.index2length[i] = length
            self.length2index[length] = i

        log.debug("Calculating init probabilities.")
        pitch_init_mat = self._stats.calc_pitch_initial_probabilities()
        duration_init_mat = self._stats.calc_duration_initial_probabilities()
        log.debug("Done.")

        def pitch_mm_init(trans_mat):
            return MarkovModel(trans_mat, pitch_init_mat,
                               self._stats.set_of_pitches,
                               self._stats.all_pitch_states, self.order)

        def duration_mm_init(trans_mat):
            return MarkovModel(trans_mat, duration_init_mat,
                               self._stats.set_of_lengths,
                               self._stats.all_length_states, self.order)

        class OnDemandMMOffsetDict(dict):
            @staticmethod
            def _transform_key(key):
                # 128th triplets (3*16)
                return Fraction(key).limit_denominator(48)

            def __init__(self, constructor, *args, **kwargs):
                self.const = constructor
                orig_dict = dict(*args, **kwargs)
                new_dict = {}
                for key in orig_dict:
                    if key < 1 or key >= 2:
                        raise ValueError("All keys must be in [1, 2)."
                                         "Keys are\n{}".format(self.keys()))
                    new_dict[OnDemandMMOffsetDict._transform_key(key)] =\
                        orig_dict[key]
                super(OnDemandMMOffsetDict, self).__init__(new_dict)

            def __setitem__(self, key, value):
                key = OnDemandMMOffsetDict._transform_key(key)
                if key < 1 or key >= 2:
                    raise ValueError("Key must be in [1, 2). Got {}".format(
                        key))
                super(OnDemandMMOffsetDict, self).__setitem__(key, value)

            def __getitem__(self, key):
                key = key % 1 + 1
                key = OnDemandMMOffsetDict._transform_key(key)
                res = super(OnDemandMMOffsetDict, self).__getitem__(key)
                if not isinstance(res, MarkovModel):
                    res = self[key] = self.const(res)
                return res

            def get(self, k, d=None):
                if k in self:
                    return self[k]
                else:
                    return d

            def iteritems(self):
                for k in self:
                    yield (k, self[k])

            def itervalues(self):
                for k in self:
                    yield self[k]

            def values(self):
                return list(self.itervalues())

            def pop(self, k, d=None):
                raise NotImplementedError

            def popitem(self):
                raise NotImplementedError

            def viewitems(self):
                raise NotImplementedError

            def viewvalues(self):
                raise NotImplementedError

        self._pitch_mm_tensor = OnDemandMMOffsetDict(pitch_mm_init,
                                                     pitch_tensor)
        self._duration_mm_tensor = OnDemandMMOffsetDict(duration_mm_init,
                                                        duration_tensor)

        log.debug("Learning melody contours")
        self._melody_contour.learn(self.phrases, n_cluster=n_cluster)
        log.debug("Learning rhythm contours")
        self._rhythm_contour.learn(self.phrases, n_cluster=n_cluster)
        log.debug("Done.")

    def generate_melody(self, length):
        """
        Compose a phrase similar to the ones of the training data.

        :type length: int
        :param length: how many 4/4 bars to create

        :return: two lists: pitches (ascii encoded) and lengths (float)
        """

        count = Fraction(1)
        pitches = []
        durations = []
        trunc = MelodyTreeTruncate()

        log.debug("Start building melody.")

        def _kick_all_too_long_durations():
            # assuming global_reachable_states is same for all MMs => pick any
            counts_left = Util.count_to_fraction(length*4+1 - count)
            states = duration_mm.states
            if durations:
                trans_vec_ = duration_mm.get_transition_vector_of_states(
                    durations
                )
            else:
                trans_vec_ = duration_mm.initial_matrix
            for i, state in enumerate(states):
                if state > counts_left:
                    trans_vec_[i] = 0
            return Util.normalize_1d_array_by_sum(trans_vec_)

        def _follow_contour(contour, states, trans_vec, sigma):
            aim_for = contour.get_curve_at(count, 4 * length + 1)
            gauss = lambda x: exp(-0.5*((x-aim_for)/sigma)**2)/sigma*sqrt(2*pi)
            trans_vec = np.array(trans_vec)
            for i, state in enumerate(states):
                if state != 'R':
                    trans_vec[i] *= gauss(state)
            return Util.normalize_1d_array_by_sum(trans_vec)

        def _follow_melody_contour(trans_vec):
            return _follow_contour(self._melody_contour,
                                   self._stats.set_of_pitches, trans_vec, 4)

        def _follow_rhythm_contour(trans_vec):
            return _follow_contour(self._rhythm_contour,
                                   self._stats.set_of_lengths, trans_vec, .33)

        def is_not_tonic(pitch):  # does not allow rest at the end
            if pitch == 'R':
                return False
            is_c = NoteNumbers.ascii_note_has_name(pitch, "C")
            is_e = NoteNumbers.ascii_note_has_name(pitch, "E")
            is_g = NoteNumbers.ascii_note_has_name(pitch, "G")
            return not is_c and not is_e and not is_g

        fail_count = 0
        while count < Util.count_to_fraction(4 * length + 1):  # in 4/4 measure
            cur_path = zip(pitches, durations)
            pitch_mm = self._pitch_mm_tensor[count]
            duration_mm = self._duration_mm_tensor[count]
            pitches_reachable = pitch_mm.local_reachable_states(pitches)
            durations_reachable = duration_mm.local_reachable_states(durations)
            _kick_all_too_long_durations()

            try:
                ignored_pitches, ignored_durations =\
                    trunc.get_forbidden_pitches_and_lengths(cur_path,
                                                            pitches_reachable,
                                                            durations_reachable)
                trans_vec = _kick_all_too_long_durations()
                trans_vec = _follow_rhythm_contour(trans_vec)
                d = duration_mm.generate_next_state(
                    previous_states=durations,
                    transition_vector=trans_vec,
                    states_to_ignore=ignored_durations
                )
                if count + d == length * 4 + 1:
                    # last note => only accept tonic notes (C, E or G)
                    ignored_pitches = [p for p in pitches_reachable
                                       if p in ignored_pitches or
                                       is_not_tonic(p)]
                if pitches:
                    trans_vec = pitch_mm.get_transition_vector_of_states(
                        pitches)
                else:
                    trans_vec = pitch_mm.initial_matrix
                trans_vec = _follow_melody_contour(trans_vec)
                p = pitch_mm.generate_next_state(
                    previous_states=pitches,
                    transition_vector=trans_vec,
                    states_to_ignore=ignored_pitches
                )
                pitches += [p]
                durations += [d]
                count += Util.count_to_fraction(d)
                fail_count = 0
            except NotEnoughTrainingData as e:
                # roll back last note
                log.error(e)
                if not cur_path:
                    assert not pitches and not durations
                    raise RuntimeError("No path in the tree is long enough!")
                trunc.truncate(cur_path)
                count -= Util.count_to_fraction(durations[-1])
                pitches = pitches[:-1]
                durations = durations[:-1]
                fail_count += 1
                if fail_count >= 400:
                    raise RuntimeError("I give up on this one. Too many errors "
                                       "due to insufficient training data.")
        return pitches, durations

    def generate_melody_with_pitch_included(self, ascii_pitch, *args, **kwargs):
        """
        Generates a melody and makes sure that the given pitch is included in
        it.

        :param ascii_pitch: the pitch to be included
        :type ascii_pitch: int
        :param args: passed to generate_melody
        :param kwargs: passed to generate_melody

        :return: passed from generate_melody
        """
        # FIXME: Random position is not a good idea in musical terms
        pitches, durations = self.generate_melody(*args, **kwargs)
        if ascii_pitch not in pitches:
            position = np.random.randint(0, len(pitches))
            pitches[position] = ascii_pitch
        return pitches, durations

    def generate_melody_with_pitches_included(self, ascii_pitches,
                                              *args, **kwargs):
        """
        Generates a melody and makes sure that the given pitches are included in
        it.

        :param ascii_pitches: list of integers (ascii encoded pitches)
            if the same pitch is given n times it will appear at least n times
            in the melody.
        :type ascii_pitches: list
        :param args: passed to generate_melody
        :param kwargs: passed to generate_melody

        :return: passed from generate_melody
        """
        # FIXME: Random positions is not a good idea in musical terms
        pitches, durations = self.generate_melody(*args, **kwargs)
        p = pitches[:]
        ascii_pitches = list(ascii_pitches)
        for pitch in ascii_pitches:  # don't add pitches that are already there
            if pitch in p:
                ascii_pitches.remove(pitch)
                p[p.index(pitch)] = -1
        positions = np.random.choice(len(pitches), len(ascii_pitches),
                                     replace=False)
        shuffle(ascii_pitches)
        for i, position in enumerate(positions):
            pitches[position] = ascii_pitches[i]
        return pitches, durations

    def generate_melody_from_set_of_pitches(self, ascii_pitches,
                                            *args, **kwargs):
        raise NotImplementedError

    def generate_melody_with_pattern_included(self, ascii_pitches, lengths,
                                              *args, **kwargs):
        raise NotImplementedError


class MelodyTreeTruncate(object):
    """
    Keeps track of the leafs found while traversing the composition tree.
    It suggests next children to explore by sticking to the duration that has
    been started to explore and eliminates the pitches that have turned out to
    lead to leafs.
    """
    def __init__(self):
        self._truncated = defaultdict(lambda:
                                      defaultdict(lambda: []))
        self._pitch_working_on = defaultdict(lambda: None)
        self._forbidden_pitches = defaultdict(lambda: [])

    def truncate(self, path):
        """
        Remember the passed path as leaf node

        :type path: a list of (pitch, length)-tuples
        :param path: The path that was identified as ending in a leaf.

        :return: None
        """
        # path is [(pitch, length), ... ]
        cur_pitch, cur_length = path[-1]
        self._truncated[tuple(path[:-1])][cur_pitch].append(cur_length)
        self._pitch_working_on[tuple(path[:-1])] = cur_pitch

    def get_forbidden_pitches_and_lengths(self, path, reachable_pitches,
                                          reachable_lengths):
        """Return the list of pitches and durations that have been truncated
        for the current path.

        :type path: a list of (pitch, length)-tuples
        :param path: The path to the node currently examined.

        :return: list of forbidden pitches, list of forbidden durations"""
        pwo = self._pitch_working_on[tuple(path)]
        if pwo:
            reachable_lengths_set = set(reachable_lengths)
            assert len(reachable_lengths) == len(reachable_lengths_set)
            forbidden_pitches = [p for p in reachable_pitches if p != pwo]
            tried_lengths = self._truncated[tuple(path)][pwo]
            forbidden_lengths = [l for l in reachable_lengths
                                 if l in tried_lengths]
            if set(forbidden_lengths) == reachable_lengths_set:
                # tried all durations => go on to next pitch
                self._forbidden_pitches[tuple(path)].append(pwo)
                self._pitch_working_on[tuple(path)] = None
            else:
                return forbidden_pitches, forbidden_lengths

        forbidden_pitches = self._forbidden_pitches[tuple(path)]
        return forbidden_pitches, []
