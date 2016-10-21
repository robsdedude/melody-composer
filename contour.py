# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import math
from fractions import Fraction
import numpy as np
import scipy as sp
import scipy.cluster.hierarchy
from util import Util
import matplotlib.pyplot as plt
import warnings

import logging
log = logging.getLogger("main")

TRANSPOSE_BY_MEAN = 'mean'
TRANSPOSE_BY_FIRST_PITCH = 'first'
NO_TRANSPOSE = False


class MelodyContour(object):
    """
    Class to help you analyze melody contours of phrases.
    """
    def __init__(self):
        self.phrases = None
        self.fft_phrases = None
        self.low_passed_fft_phrases = None
        self.z = None
        self.flat = None
        self.clusters = None
        self.funcs = None
        self.transition_matrix = None
        self.initial_matrix = None
        self.curve_func = None
        self.curve_cluster = None

    @staticmethod
    def _gauss(x, max_=1, min_=0, center=0):
        """
        Calculate gaussian function (bell curve) phi(x).
        sigma (standard deviation) is given implicitly by max_ and min_.

        :type x: float
        :param x: x value
        :type max_: float
        :param max_: global maximum of the phi(x)
        :type min_: float
        :param min_: limes phi(x) for x -> +- inf
        :type center: float
        :param center: x of global maximum (mean of gauss distribution)

        :return: phi(x) (float)
        """
        a = max_ - min_
        b = center
        c = 1  # c is the standard deviation
        return a*math.exp(-((x-b)**2)/(2*c**2)) + min_

    def get_curve_at(self, count, total_counts):
        """
        After learning "the contour" from the training data return the value of
        the contour for a given count.

        :type count: int
        :param count: The count at which to get the value of the contour.
        :param total_counts: The total number of counts.
            This is needed so that the function knows the relative position
            in the phrase / contour.

        :return: float
        :raise RuntimeError: if 'learn' has not been called earlier.
        """
        if self.funcs is not None:
            return self.curve_func(count/total_counts)
        else:
            raise RuntimeError("Call 'learn' before first call of "
                               "'get_curve_at'")

    # --- learning algorithms -----------------------------------------------
    def learn(self, phrases, lowcut=6, cluster_method='ward', n_cluster=17,
              transpose=True):
        """
        Learn a contour from the phrases.

        This is done by extracting the contours of the phrases, clustering them
        and selecting the most representative contour.

        :type phrases: list of lists of (pitch, length)-tuples
        :param phrases: The phrases to learn the contours from.
        :type lowcut: int
        :param lowcut: The number of lowest frequencies to keep of the phrases
            contour.
        :type cluster_method: basestring
        :param cluster_method: Clustering method of to use in
            scipy.cluster.hierarchy.linkage (see their documentation).
        :type n_cluster: int
        :param n_cluster: The number of clusters to build.
        :param transpose: s. MelodyContour.transpose_to_zero

        :return: None
        """
        self.phrases = phrases
        self.low_passed_fft_phrases = []
        self.fft_phrases = []
        self.funcs = []
        self.clusters = []
        features = []
        for phrase in phrases:
            phrase = self.mirror_phrase(phrase)
            fft_phrase = self.phrase_to_fft(phrase, min_length=lowcut*2)
            self.fft_phrases.append(fft_phrase)
            fft_phrase = self.low_pass(fft_phrase, lowcut)
            self.low_passed_fft_phrases.append(fft_phrase)

            phrase_transposed = self.transpose_to_zero(phrase, transpose)
            fft_phrase_transposed = self.phrase_to_fft(phrase_transposed,
                                                       min_length=lowcut*2)
            fft_phrase_transposed = self.low_pass(fft_phrase_transposed, lowcut)
            features.append([x.real for x in fft_phrase_transposed] +
                            [x.imag for x in fft_phrase_transposed])

        self.z = sp.cluster.hierarchy.linkage(
            features, method=cluster_method.encode("utf-8"))
        if len(features) > 1:
            self.flat = sp.cluster.hierarchy.fcluster(self.z,
                                                      min(n_cluster,
                                                          len(phrases)),
                                                      criterion='maxclust')
        else:
            self.flat = np.array([1])

        for i in range(n_cluster):
            cluster = [phrases[idx] for idx in range(len(self.flat))
                       if self.flat[idx] == i+1]
            if cluster:
                self.clusters.append(cluster)
                ffts = [self.low_passed_fft_phrases[idx]
                        for idx in range(len(self.flat))
                        if self.flat[idx] == i+1]

                mean = np.mean(ffts, 0)

                def unmirror_ifft(ifft_func):
                    fp = ifft_func(.5)

                    def new_ifft(x, first_pitch=None):
                        y = ifft_func((x % 1)/2+0.5)
                        if first_pitch is None:
                            return y
                        else:
                            return y-fp+first_pitch

                    return new_ifft

                self.funcs.append(unmirror_ifft(
                    self.ifft(mean, norm_x=True, norm_y=False)))

        if len(self.clusters) != n_cluster:
            warnings.warn("Formed only %i clusters instead of %i"
                          % (len(self.clusters), n_cluster), RuntimeWarning)

        idx = self.choose_cluster(map(len, self.clusters), self.funcs)
        self.curve_func = self.funcs[idx]
        self.curve_cluster = self.clusters[idx]

        self.learn_debug_out()

    def learn_debug_out(self):
        """
        Helper function that visualizes the learned contour.

        This function does nothing it the logging level is set above 'debug'.
        """
        if log.getEffectiveLevel() <= logging.DEBUG:
            ContourVisualizer.draw_phrases(self.curve_cluster, MelodyContour,
                                           TRANSPOSE_BY_MEAN)
            ContourVisualizer.draw_contour(self.curve_func, transpose=False)
            plt.show()

    @classmethod
    def phrase_to_fft(cls, phrase, min_length=None):
        """
        Convert a phrase into a normalized fft frequency array.

        Internally the phrase is sampled (rasterized) and interpolated at rests
        before the Fourier transform is applied and normalized.

        :type phrase: list of (pitch, length)-tuples
        :param phrase:  The phrase to convert
        :type min_length: int
        :param min_length: The minimal number of fft coefficients to return.

        :return: frequency array
        """
        sr = cls.get_greatest_sampling_rate(phrase)
        r = cls.rasterize_phrase(phrase, sr, min_length)
        r = cls.interpolate_rests(r)
        fft = np.fft.fft(np.array(r))
        return fft/fft.shape[0]

    @staticmethod
    def low_pass(fft, l):
        """
        Apply a low pass filter to a Fourier spectrum. Keep the lowest l
        frequencies. The energy-loss is amortized.

        :type fft: np.array
        :param fft: The FFT frequency array
        :type l: int
        :param l: The lowers l frequencies to keep.

        :return: Low-passed and thus shorter frequency array
        """
        if 2*l > len(fft):
            raise ValueError("l (cut off threshold) must be greater than the "
                             "length of fft")
        kept_energy = np.abs(fft[1:l]).sum()
        lost_energy = np.abs(fft[l:]).sum()
        if kept_energy != 0:
            f = (kept_energy+lost_energy)/kept_energy  # fix energy-loss
            res = fft[:l] * f
            res[0] /= f
        else:
            res = fft[:l]
        return res

    @staticmethod
    def get_greatest_sampling_rate(phrases):
        """
        Calculate the greatest common possible sampling rate of given phrases.

        :type phrases: list of lists of (pitch, length)-tuples
        :param phrases: The phrases to calculate the sampling rate from.

        :return: The sampling rate in quarter notes (floats)
        """
        if not isinstance(phrases[0][0], (list, tuple)):
            phrases = [phrases]
        l = [e[1] for phrase in phrases for e in phrase]
        # smallest value are 128th triplets
        l = map(lambda x: Fraction(x).limit_denominator(3**9).denominator, l)
        l = list(set(l))
        return 1/Util.lcm(*l)

    @staticmethod
    def rasterize_phrase(phrase, resolution=1, min_length=None):
        """
        Sample / raster a phrase.

        :type phrase: list of (pitch, length)-tuples
        :param phrase: The phrase to sample
        :type resolution: float
        :param resolution: The sampling rate in quarter notes.
        :type min_length: int or None
        :param min_length: If provided the sampled phrase will iteratively be
            subsampled to double it's length until the min_length is reached.

        :return: A list of pitches.
        """
        result = []
        for note in phrase:
            pitch, duration = note
            assert duration % resolution < 1e-6  # fix float precision issue
            result += [pitch] * int((duration / resolution)+1e-6)
        if min_length is not None:
            q = min_length/len(result)
            result = [x for pitch in result for x in [pitch]*int(math.ceil(q))]
            assert len(result) >= min_length
        return result

    @staticmethod
    def interpolate_rests(phrase):
        """
        Interpolate the rests of a rasterized phrase.

        Rests at the beginning and at the end are interpolated with the
        respectively the first or the last pitch. Rests between pitches are
        linearly interpolated.

        :type phrase: list of ints (pitches)
        :param phrase: The rasterized phrase

        :return: Return the phrase with interloped rests. No rests are
            contained anymore but only pitches.
        """
        found_tone = False
        last_pitch = None
        buf = 0
        i = 0
        res = []
        while i < len(phrase):
            if phrase[i] != 'R':
                if not found_tone:
                    found_tone = True
                if buf:
                    if last_pitch is not None:
                        f = lambda x: last_pitch+(phrase[i]-last_pitch)*x/(buf+1)
                        res += [f(x) for x in range(1, buf+1)]
                    else:
                        res += [phrase[i]]*buf
                    buf = 0
                res.append(phrase[i])
                last_pitch = phrase[i]
            else:
                buf += 1
            i += 1
        if buf:
            res += [last_pitch] * buf
        assert found_tone
        assert len(phrase) == len(res)
        return res

    @staticmethod
    def mirror_phrase(phrase):
        """
        Prepend the inverted phrase to it. Works on resterized phrases.

        :type phrase: list of ints (pitches)
        :param phrase: The rasterized phrase

        :return: phrase.reverse()+phrase
        """
        reversed_phrase = phrase[:]  # shallow copy
        reversed_phrase.reverse()
        return reversed_phrase + phrase

    @staticmethod
    def transpose_to_zero(phrase, method=TRANSPOSE_BY_MEAN):
        """
        Transpose a phrase to be "around zero" depending on the method specified

        :type phrase: list of (pitch, length)-tuples
        :param phrase: The phrase to be transposed
        :type method: One one True, False,
            TRANSPOSE_BY_MEAN, TRANSPOSE_BY_FIRST_PITCH, NO_TRANSPOSE
        :param method:
            If True or TRANSPOSE_BY_MEAN:
                Transpose the phrase so that the mean of the pitches (ignoring
                their durations) is zero.
            If TRANSPOSE_BY_FIRST_PITCH:
                Transpose the phrase so that the first pitch is zero
            If False or NO_TRANSPOSE:
                This function does nothing.

        :return: The transposed phrase (list of (pitch, length)-tuples)
        """
        if method is TRANSPOSE_BY_MEAN or method is True:
            pitches = [note[0] for note in phrase if note[0] != "R"]
            mean = np.mean(pitches)
            return [(note[0]-mean if note[0] != "R" else "R", note[1])
                    for note in phrase]
        elif method is TRANSPOSE_BY_FIRST_PITCH:
            pitches = [note[0] for note in phrase if note[0] != "R"]
            fp = pitches[0]
            return [(note[0]-fp if note[0] != "R" else "R", note[1])
                    for note in phrase]
        else:
            return phrase

    @staticmethod
    def ifft(fft, norm_x=False, norm_y=True):
        """
        Transform Fourier coefficients into a continuous function

        :param fft: list of Fourier coefficients (Fourier spectrum)
        :param norm_x: squeeze the function to repeat within [0, 1]
        :param norm_y: normalize amplitudes by the number of frequencies

        :return: a function f(x)
        """
        # find formula here:
        # http://www.engineeringproductivitytools.com/stuff/T0001/PT01.HTM
        fft = np.array(fft, copy=True)
        n = len(fft)
        if norm_y:
            fft *= 2/n
        if not norm_x:
            factor = 2j*np.pi*np.arange(n)/n
        else:
            factor = 2j*np.pi*np.arange(n)  # /n *n
        return lambda x: np.sum(fft*np.exp(x*factor)).real

    @staticmethod
    def choose_cluster(cluster_sizes, funcs):
        """
        Selects a the "best" cluster.

        The best cluster is a large one with a high difference of the global
        maximum and global minimum of it's representative curve (mean contour).

        :type cluster_sizes: list of ints
        :param cluster_sizes: The sizes of the clusters.
        :type funcs: list of functions f_i(x)
        :param funcs: The representative curves of the clusters.

        :return: Index of the chosen cluster (int)
        """
        sizes = np.array(cluster_sizes, float)
        x = np.linspace(0, 1, 1000)
        # calculate widths
        widths = np.zeros_like(funcs, float)
        for i, func in enumerate(funcs):
            f = np.vectorize(func)
            y = f(x)
            m = np.mean(y)
            widths[i] = np.mean(np.abs(y-m))
        # normalize widths
        # widths -= np.min(widths)
        widths /= np.max(widths)

        # normalize sizes
        sizes /= np.max(sizes)

        return np.argmax(sizes+widths**(1/3))


class RhythmContour(MelodyContour):
    """
    Works like MelodyContour but for rhythm (note length) contours.
    """
    def learn(self, phrases, lowcut=7, cluster_method='ward', n_cluster=8,
              transpose=True):
        return super(RhythmContour, self).learn(
            phrases, lowcut, cluster_method, n_cluster, transpose
        )

    @staticmethod
    def rasterize_phrase(phrase, resolution=1, min_length=None):
        result = []
        for note in phrase:
            pitch, duration = note
            assert duration % resolution < 1e-6
            result += [duration] * int((duration / resolution)+1e-6)
        if min_length is not None:
            q = min_length/len(result)
            result = [x for duration in result for x in [duration]*int(math.ceil(q))]
            assert len(result) >= min_length
        return result

    @staticmethod
    def interpolate_rests(phrase):
        # there are no "rhythm rests"
        return phrase

    @staticmethod
    def transpose_to_zero(phrase, method=TRANSPOSE_BY_MEAN):
        if method in (TRANSPOSE_BY_FIRST_PITCH, TRANSPOSE_BY_MEAN):
            raise NotImplementedError("Can't transpose a rhythm.")
        return phrase

    def learn_debug_out(self):
        if log.getEffectiveLevel() <= logging.DEBUG:
            ContourVisualizer.draw_phrases(self.curve_cluster, RhythmContour)
            ContourVisualizer.draw_contour(self.curve_func, transpose=False)
            plt.show()


class ContourVisualizer(object):
    """
    Collection of helper function to visualize clusters and their representative
    curves.
    """
    @staticmethod
    def draw_phrase(phrase, cls, transpose=False):
        """
        Draw a phrase a step function.

        :type phrase: list of (pitch, length)-tuples
        :param phrase: The phrase to visualize
        :type cls: MelodyContour or RhythmContour
        :param cls: Determines if the melody or the rhythm of phrase should be
            visualized.
        :param transpose: s. MelodyContour.transpose_to_zero

        :return: None
        """
        phrase = cls.transpose_to_zero(phrase, transpose)
        m = cls.get_greatest_sampling_rate(phrase)
        r = cls.rasterize_phrase(phrase, m)
        r = [x if x != 'R' else None for x in r]
        # m = np.mean([e for e in r if e is not None])
        # r = [e-m if e is not None else e for e in r]
        fac = int(math.ceil(1000.0/len(r)))
        y = [e for tup in zip(*([r]*fac)) for e in tup]
        x = np.linspace(0, 1, len(y))
        plt.plot(x, y)

    @staticmethod
    def draw_phrases(phrases, cls, transpose=False):
        """
        Calls ContourVisualizer.draw_phrase for each phrase in phrases (list).
        """
        for phrase in phrases:
            ContourVisualizer.draw_phrase(phrase, cls, transpose)

    @staticmethod
    def draw_contour(func, transpose=False):
        """
        Visualizes a function f(x) defined within [0, 1]

        :param func: the function f(x) to visualize
        :param transpose: s. MelodyContour.transpose_to_zero

        :return: None
        """
        f = np.vectorize(func)
        x = np.linspace(0, 1, 1000)
        y = f(x)
        if transpose is True or transpose is TRANSPOSE_BY_MEAN:
            y -= np.mean(y)
        elif transpose is TRANSPOSE_BY_FIRST_PITCH:
            y -= y[0]
        plt.plot(x, y, lw=3)

    @staticmethod
    def draw_sampled_phrase(phrase):
        """
        Like ContourVisualizer.draw_phrase but for a sampled phrase

        :type phrase: list of pitches (int in [0, 127] or 'R')
        :param phrase: The sampled phrase to visualize.

        :return: None
        """
        y = [y if y != 'R' else None for y in phrase]
        x = np.linspace(0, 1, len(y))
        plt.plot(x, y, 'x')


def main():
    CLUSTERMETHODS = ["single",
                     "complete",
                     "weighted",
                     "average",
                     "centroid",
                     "median",
                     "ward"]
    LOWCUT = 6
    CLUSTERMETHOD = CLUSTERMETHODS[6]
    NUMBER_CLUSTERS = 17
    CLUSTER_TRANSPOSED = TRANSPOSE_BY_MEAN
    DRAW_TRANSPOSED = TRANSPOSE_BY_MEAN

    from midi import MtcMidi
    import os

    basepath = os.path.join("MTC-FS-1.0", "good_midi")
    files = os.listdir(basepath)
    phrases = []
    for fn in files:
        fn = os.path.join(basepath, fn)
        r = MtcMidi.read_midi(fn)
        phrases += r

    cv = ContourVisualizer

    mc = MelodyContour()
    mc.learn(phrases, LOWCUT, CLUSTERMETHOD, NUMBER_CLUSTERS, CLUSTER_TRANSPOSED)
    print("chose cluster #" + str(mc.funcs.index(mc.curve_func)+1))
    for i in range(len(mc.funcs)):
        plt.title("Cluster #"+str(i+1)+", size: "+str(len(mc.clusters[i])))
        cv.draw_phrases(mc.clusters[i], MelodyContour, DRAW_TRANSPOSED)
        cv.draw_contour(mc.funcs[i], DRAW_TRANSPOSED)
        plt.show()

    LOWCUT = 7
    CLUSTERMETHOD = CLUSTERMETHODS[6]
    NUMBER_CLUSTERS = 8

    rc = RhythmContour()
    rc.learn(phrases, LOWCUT, CLUSTERMETHOD, NUMBER_CLUSTERS, NO_TRANSPOSE)
    print("chose cluster #" + str(rc.funcs.index(rc.curve_func)+1))
    for i in range(len(rc.funcs)):
        plt.title(CLUSTERMETHOD+" - Cluster #"+str(i+1)+", size: " +
                  str(len(rc.clusters[i])))
        cv.draw_phrases(rc.clusters[i], RhythmContour, False)
        cv.draw_contour(rc.funcs[i], False)
        plt.show()


if __name__ == '__main__':
    main()
