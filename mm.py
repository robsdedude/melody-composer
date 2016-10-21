# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy as sp
from util import Util


class NotEnoughTrainingData(RuntimeError):
    pass


class MarkovModel(object):
    """
    This class will generate a sequence of states based on first order
    Markov Model

    :type transition_matrix: numpy.ndarray
    :param transition_matrix: A 2d list representing the probabilities for
        each transition. Must be normalized! Must be element Mat(n x n)
        where n is the number of states.
    :type initial_matrix: numpy.ndarray
    :param initial_matrix: A list representing the probabilities for each
        state at the beginning. Must be normalized! Must have a length of n
        where n is the number of states.
    :type states: list
    :param states: A list naming the states. Enumeration by default.

    :return: Instance of Markovmodel
    """

    def __init__(self, transition_matrix, initial_matrix, states=None,
                 state_combinations=None, order=1):
        self.__transition_matrix = sp.sparse.lil_matrix(transition_matrix)
        if not isinstance(initial_matrix, np.ndarray):
            self.__initial_matrix = np.array(initial_matrix)
        else:
            self.__initial_matrix = initial_matrix
        if states is None:
            states = list(range(len(self.initial_matrix)))
        if not state_combinations:
            state_combinations = Util.cartesian_power(states, order)
        self.states = states
        self.state_combinations = state_combinations
        self.order = order
        self.__global_reachable_states = None

    def super_assertion(self):
        for r in self.transition_matrix:
            s = sum(r)
            if not (np.isclose(s, 0) or np.isclose(s, 1)):
                raise AssertionError("Nooope {}\nsums: {}".format(
                    self.transition_matrix, self.transition_matrix.sum(axis=1)))
        if not (sum(self.initial_matrix) in (0, 1)):
            raise AssertionError("Nooope {}".format(self.initial_matrix))

    @property
    def transition_matrix(self):
        return self.__transition_matrix.copy()

    @transition_matrix.setter
    def transition_matrix(self, transition_matrix):
        self.__transition_matrix = transition_matrix
        self.update_global_reachable_states()

    @property
    def initial_matrix(self):
        return self.__initial_matrix.copy()

    @property
    def global_reachable_states(self):
        if self.__global_reachable_states is not None:
            return self.__global_reachable_states[:]
        tm = self.transition_matrix
        reachable_mask = np.not_equal(np.array(tm.sum(axis=0))[0, :],
                                      np.zeros((tm.shape[1])))
        reachable = []
        for i in range(len(self.states)):
            if reachable_mask[i]:
                reachable += [self.states[i]]
        self.__global_reachable_states = reachable[:]
        return reachable

    def update_global_reachable_states(self):
        self.__global_reachable_states = None

    def local_reachable_states(self, previous_states=None):
        if not previous_states and previous_states is not None:  # empty list
            row = list(self.initial_matrix)
        else:
            prev_states_idx = self.state_combinations.index(
                tuple(previous_states[-self.order:]))
            row = list(self.transition_matrix[prev_states_idx].toarray()[0, :])
        return [self.states[idx] for idx, prop in enumerate(row)
                if prop != 0]

    def generate_next_state(self, previous_states=None, states_to_ignore=[],
                            transition_vector=None):
        """
        Generates the next state

        :type previous_states: list
        :param previous_states: name of all state we've seen to far. If None we
            use the initial matrix to generate the first state.
        :type states_to_ignore: list
        :param states_to_ignore: list of states you do not want to be returned.
            Default is empty list.
        :type transition_vector: numpy.ndarray
        :param transition_vector: if given, the param previous_states is
            ignored. Transition probabilities of transition_vector are used
            instead.
        :return: next state (element of states passed at initialization i.e.
            state name)
        """

        def _draw_from_categories(cat_probs):
            """
            Draws randomly a category

            :param cat_probs: the probabilities (normalized) of the categories.
            :return: category index

            e.g. _draw_from_categories([.25, .75]) will return 0 with a
            probability of 1/4 and 1 with a probability of 3/4.
            """
            r = np.random.uniform(0, 1)
            categories = np.cumsum(cat_probs)
            for x in range(len(cat_probs)):
                if categories[x] - r >= 0:
                    return x
            assert cat_probs.sum() == 0
            raise NotEnoughTrainingData("Can't generate following state. "
                                        "I've never seen what's happening "
                                        "after this state: {} of {}"
                                        .format(previous_states, self.states))

        if transition_vector is None:
            if previous_states:
                categories = self.get_transition_vector_of_states(
                    previous_states)
            else:
                categories = self.initial_matrix
        else:
            assert isinstance(transition_vector, np.ndarray)
            assert len(transition_vector.shape) == 1
            assert transition_vector.shape[0] == len(self.states)
            categories = np.array(transition_vector)

        for state_to_ignore in states_to_ignore:
            categories[self.states.index(state_to_ignore)] = 0
        if categories.sum():  # normalization
            categories /= categories.sum()

        return self.states[_draw_from_categories(categories)]

    def generate_sequence(self, n_states):
        """
        calls generate_next_state n_states times

        :param n_states: (int) how many states to be in the sequence
        :returns: list of states (state names)
        """
        res = []
        current_state = None
        for i in range(n_states):
            current_state = self.generate_next_state(current_state)
            res += [current_state]
        return res

    def drop_state(self, state):
        """
        Removes a state by setting transition probabilities to that state to
        zero so that it's impossible to got to that state again.
        Attention! This can't be undone.

        :type state: what ever type your state names have (default is int)
        :param state: The name of the state to be dropped.
        """
        state_index = self.states.index(state)
        mask = np.ones((self.transition_matrix.shape[1],))
        mask[state_index] = 0
        m = sp.sparse.lil_matrix((self.transition_matrix.shape[1],
                                  self.transition_matrix.shape[1]))
        m.setdiag(mask)
        new_trans_mat = sp.sparse.lil_matrix(self.transition_matrix * m)
        self.transition_matrix = Util.normalize_lil_mat_by_rowsum(
            new_trans_mat)
        self.__initial_matrix[state_index] = 0
        self.__initial_matrix = Util.normalize_1d_array_by_sum(
            self.initial_matrix)

    def get_transition_vector_of_states(self, states):
        """
        Return transition (probability) vector for previously generated states.

        :type states: list
        :param states: The previously generated states.

        :return: Transition probabilities in a numpy array.
            The order of states can be found in self.states
        """
        return self.get_transition_vector_of_index(
            self.state_combinations.index(tuple(states[-self.order:]))
        )

    def get_transition_vector_of_index(self, idx):
        """
        Return transition (probability) vector with a given index.

        :type idx: int
        :param idx: The index of the transition vector. Assume two states s1 and
            s2. They are enumerated in lexicographical order like this:
            0 - (s1)
            1 - (s2)
            2 - (s1, s1)
            3 - (s1, s2)
            4 - (s2, s1)
            ...

        :return: Transition probabilities in a numpy array.
            The order of states can be found in self.states
        """
        return np.array(self.transition_matrix[idx].todense())[0, :]

    def get_transition_vector_of_indexes(self, idx_tuple):
        """
        Return transition (probability) vector for the indices of the previously
        generated states.

        :type idx_tuple: tuple of ints
        :param idx_tuple: Indices in self.states of the previous states.

        :return: Transition probabilities in a numpy array.
            The order of states can be found in self.states
        """
        return self.get_transition_vector_of_index(
            Util.index_of_vector_in_cartesian_power_set(self.states,
                                                        idx_tuple)
        )
