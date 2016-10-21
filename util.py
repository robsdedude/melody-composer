from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from fractions import gcd
from itertools import product

from fractions import Fraction

import numpy as np
from scipy.sparse import lil_matrix

import config

__author__ = 'rouven'


class Util(object):
    @staticmethod
    def gcd(*numbers):
        """Return the greatest common divisor of the given integers"""
        # https://gist.github.com/endolith/114336
        return reduce(gcd, numbers)

    @staticmethod
    def lcm(*numbers):
        """Return lowest common multiple."""
        # https://gist.github.com/endolith/114336
        def lcm(a, b):
            return (a * b) // gcd(a, b)

        return reduce(lcm, numbers, 1)

    @staticmethod
    def normalize_lil_mat_by_rowsum(np_2d_array):
        """
        Normalizes a 2d array i.e. Matrix so that the sum of each row equals
        one. If all entries in a row are zero this is not possible for obvious
        reasons. In this case the algorithm just keeps the row as it is.

        :param np_2d_array: numpy array with two axis
        :type np_2d_array: lil_matrix
        :returns: normalized 2d array
        """
        mat = np_2d_array.copy()
        row_sums = np.array(mat.sum(axis=1))[:, 0]
        row_indices, col_indices = mat.nonzero()
        for row in set(row_indices):
            mat.data[row] /= row_sums[row]
        return mat

    @staticmethod
    def normalize_2d_array_by_rowsum(np_2d_array):
        """
        Normalizes a 2d array i.e. Matrix so that the sum of each row equals
        one. If all entries in a row are zero this is not possible for obvious
        reasons. In this case the algorithm just keeps the row as it is.

        :param np_2d_array: numpy array with two axis
        :type np_2d_array: numpy.array
        :returns: normalized 2d array
        """
        mat = np_2d_array.copy()
        row_sums = np.array(mat).sum(axis=1)[:, np.newaxis]
        zero_mask = np.equal(row_sums, np.zeros(row_sums.shape))
        # this avoids division by zero but can cause rows to have a sum
        # of zero. i.e. no following state can be generated.
        row_sums += zero_mask
        return mat / row_sums

    @staticmethod
    def normalize_1d_array_by_sum(np_1d_array):
        """
        Normalizes an 1d array so that the sum of all elements equals one.
        If all entries are zero this is not possible for obvious reasons.
        In this case the algorithm just keeps the array as it is.

        :param np_1d_array: numpy array with one axis
        :type np_1d_array: numpy.array
        :returns: normalized 2d array
        """
        return Util.normalize_2d_array_by_rowsum(
            np_1d_array[np.newaxis, :])[0, :]

    class InvalidAsciiNote(ValueError):
        def __init__(self, val, *args, **kwargs):
            ValueError.__init__(self,
                                "invalid literal for ascii_pitch '{}' must be "
                                "'R' or int between 0 and 127 (both included)"
                                .format(val))

    @staticmethod
    def str_to_ascii_pitch(string):
        """
        Convert a ascii pitch to an int or 'R'.

        Takes either 'R' or a string containing an integer within [0, 127].

        :param string: the string to convert.

        :return: int or 'R'
        :raise Util.InvalidAsciiNote if the input is not 'R' or within
            ['0', '128'].
        """
        try:
            i = int(string)
            if 0 <= i <= 127:
                return i
            else:
                raise Util.InvalidAsciiNote(string)
        except ValueError:
            if string == 'R':
                return string
            else:
                raise Util.InvalidAsciiNote(string)

    @staticmethod
    def cartesian_power(set_, pow_):
        """len_set = len(set_)
        # FIXME: this is not efficient!
        len_res = CartesianPowerMathHelper.calc_f(pow_, len_set)
        res = [None]*len_res
        for idx in range(len_res):
            res[idx] = Util.vector_at_index_in_cartesian_power_set(set_, idx)
        return res"""
        res = []
        for i in range(1, pow_ + 1):
            res += list(product(set_, repeat=i))
        return res

    @staticmethod
    def index_of_vector_in_cartesian_power_set(set_, vector):
        len_x = len(set_)
        len_v = len(vector)
        index = 0
        for idx in range(len_v - 1, -1, -1):
            idx_inv = len_v - idx
            value = len_x ** (idx_inv - 1)
            index += (set_.index(vector[idx]) + 1) * value
        return index - 1

    @staticmethod
    def vector_at_index_in_cartesian_power_set(set_, index):
        index += 1
        len_x = len(set_)
        # calculate needed len of vector
        len_v = CartesianPowerMathHelper.calc_f_inv(index, len_x)[0]
        v = [0] * len_v
        for idx_v in range(len_v):
            inv_idx_v = len_v - idx_v
            r = CartesianPowerMathHelper.calc_f(inv_idx_v, len_x)
            p_r = CartesianPowerMathHelper.calc_f(inv_idx_v - 1, len_x)
            d = (r - p_r) / len_x
            m = 1
            while index > p_r + m * d:
                m += 1
            v[idx_v] = set_[m - 1]
            index -= m * d
        return tuple(v)

    @staticmethod
    def count_to_fraction(count):
        return Fraction(count).limit_denominator(config.MAX_LENGTH_DENOMINATOR)

    @staticmethod
    def quantize_duration(duration):
        return float(Util.count_to_fraction(duration))


class CartesianPowerMathHelper(object):
    @staticmethod
    def calc_f(x, b):
        """Calculates Sum over b**i from i=1 to x

        :type x: int
        :type b: int

        :return: result"""
        r = 0
        for i in range(1, x + 1):
            r += b ** i
        return r

    @staticmethod
    def calc_f_inv(x, b):
        """Calculates the inverse function of sum over b**i from i=1 to x.
        Function returns the result (int)(floored) and the rest.

        :type x: int
        :type b: int

        :return: result, reset

        result is ceiled to next int and rest is how much was added. """
        r = 0
        i = 1
        while True:
            r += b ** i
            if r >= x:
                break
            i += 1
        return i, r - x


class NoteNumbers(object):
    names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#",
                   "B"]
    names_flat = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb",
                  "B"]

    @staticmethod
    def ascii_note_to_name(note, sharp=True):
        if sharp:
            return NoteNumbers.names_sharp[note % 12]
        else:
            return NoteNumbers.names_flat[note % 12]

    @staticmethod
    def ascii_note_has_name(note, name):
        name = name.capitalize()
        return (NoteNumbers.names_sharp[note % 12] == name or
                NoteNumbers.names_flat[note % 12] == name)
