#!/usr/bin/python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
from random import shuffle
from glob import glob

from util import Util
from melody_generator import MelodyGenerator
from midi import MtcMidi
from midi import MIDI
w
import logging
log = logging.getLogger("main")


def get_training_set(src, learn_limit=None, shuffled=True):
    """
    Read training input files.

    :type src: unicode
    :param src: the input source to learn from. May be a single .mid, .midi
        file or a folder. If a folder is given, the algorithm will learn
        from all .mid/.midi files inside it.
    :type learn_limit: int
    :param learn_limit: if folder is specified as input, this will tell the
        learning algorithm not to take more files into account as specified.
        The files will be chosen randomly. If None (default) is specified,
        the algorithm will use all files.
    :type shuffled: bool
    :param shuffled: If set to True the input melodies will be returned in
        random order.

    :return A list of of lists of lists of pairs (type tuple). First dimension
        stands for the songs. Second for the phrases of each song. The third for
        the notes of the song. Notes are pairs in the form
        (int - ascii pitch, float - duration where 1.0 is one quarter note).
    """

    if os.path.isdir(src):
        log.debug("Searching dir {}".format(src))
        file_names = glob(os.path.join(src, "*.mid")) +\
            glob(os.path.join(src, "*.midi"))
    else:
        file_names = [src]
    if learn_limit:
        if shuffled:
            shuffle(file_names)
        file_names = file_names[:learn_limit]
    if len(file_names) == 0:
        raise RuntimeError("No files found to get_training_set from.")

    log.debug("Reading in files...\n{}".format(file_names))
    songs = []

    for fn in file_names:
        phrases = MtcMidi.read_midi(fn)
        songs.append(phrases)

    return songs


def generate_melody(songs, bars, *constraint_args, **kwargs):
    """
    Generate a melody by using constraint satisfaction problem and markov models

    :param songs: A list of songs represented as lists of phrases.
    :type songs: list
    :param bars: int giving the length of the melody to be generated in bars.
    :type bars: int
    :param constraint_args: depending on constraint param:
        0: none needed
        1: index of state representing the note in the markov model
        2: list of notes like specified for constraint=1
        3: two lists: list of ascii notes to be replaced with ascii notes.
            e.g. [64, 66], [66, 68] would transpose the the tho notes 64 and 66
            by a whole tone.
        4: pattern_notes, pattern_durations where:
            pattern_notes is a list of ascii notes
            patter_duration is a list of floats representing the length of each
                note
    :type constraint_args: int
    :param constraint: Choose a constraint (default is 0)
        0: No further constraints (besides the markov model)
        1: include a specific tone
        2: include specific tones
        3: replace tones with given tones
        4: include a specified pattern
    :type constraint: int
    :param mm_order: the order of the markov model to use.
    :type mm_oder: int

    :return Two lists: notes (ascii notes) and durations (floats)
    """
    # FIXME: this only works if songs parameter is the same for each call
    global mg
    if 'mg' not in globals():
        mg = MelodyGenerator(songs, mm_order=kwargs.pop('mm_order', 1))
        mg.learn(n_cluster=kwargs.pop('n_cluster'))

    constraint = kwargs.get('constraint', 0)

    if constraint == 0:
        return mg.generate_melody(bars)
    elif constraint == 1:
        return mg.generate_melody_with_pitch_included(constraint_args[0], bars)
    elif constraint == 2:
        return mg.generate_melody_with_pitches_included(constraint_args[0],
                                                        bars)
    elif constraint == 3:
        return mg.generate_melody_from_set_of_pitches(constraint_args[0], bars)
    elif constraint == 4:
        return mg.generate_melody_with_pattern_included(constraint_args[0],
                                                        constraint_args[1],
                                                        bars)
    else:
        raise ValueError("Constraint must be in range [0, 4].")


def insert_suffix_before_file_ending(basename, suffix, ending):
    """
    Inserts a number right before the file ending.

    :param basename: the filename in which to insert the number
    :type basename: unicode
    :param suffix: the number to insert
    :type suffix: unicode
    :param ending: the file ending
    :type ending: unicode

    :return string with insertion

    Examples:
    >>> insert_suffix_before_file_ending('foo.bar', '7', '.bar')
    'foo7.bar'

    >>> insert_suffix_before_file_ending('foo.bar', '1', '.bar')
    'foo1.bar'

    >>> insert_suffix_before_file_ending('foo.bar', '42', '.test')
    'foo.bar42.test
    """
    head, sep, tail = basename.rpartition(ending)
    if not tail:
        return head + unicode(suffix) + sep
    else:
        return basename + unicode(suffix) + ending


def main(args):
    """
    As the name "main" suggest this method evaluates the arguments and runs all
    the rest of the code according to the passed args.

    :param args: the arguments passed to the program
    :type args: argparse.Namespace
    :return: None
    """
    log.debug(args)
    input_phrases = get_training_set(args.input, args.l, not args.no_shuffle)
    constraint = 0
    constraint_args = []
    rw = lambda x: "".join(x.split())  # removes whitespaces from strings
    if args.c1 is not None:
        # include pitch
        constraint = 1
        constraint_args = [int(rw(args.c1))]
    elif args.c2 is not None:
        # include pitches
        constraint = 2
        constraint_args = list(
            map(Util.str_to_ascii_pitch, rw(args.c2).split(",")))
    elif args.c3 is not None:
        # only use given pitches
        constraint = 3
        constraint_args = list(
            map(Util.str_to_ascii_pitch, rw(args.c3).split(",")))
    elif args.c4 is not None:
        # include pattern
        constraint = 4
        p, l = rw(args.c4).split(";")
        p = list(map(Util.str_to_ascii_pitch, p.split(",")))
        l = list(map(Util.str_to_ascii_pitch, l.split(",")))
        constraint_args = [p, l]

    log.debug("Generating melody with constraint {} ({})".format(
        constraint, constraint_args))
    bars = args.b
    for i in range(args.m):
        infix = str(i) if args.m > 1 else ''
        fn_out = insert_suffix_before_file_ending(args.output, infix, '.mid')
        pitches, rhythm = generate_melody(input_phrases, bars, *constraint_args,
                                          constraint=constraint,
                                          mm_order=args.mm_order,
                                          n_cluster=args.n_cluster)
        MIDI.write_midi(fn_out, pitches, rhythm)
        if args.play:
            MIDI.play_midi(fn_out)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Auto generates melodies by learning.",
                            epilog="Notes are written as integer "
                                   "(see midi values) and separated by spaces "
                                   ".\nDurations are given as floats. "
                                   "Separate list elements with ',' and lists "
                                   "with ';'")
    parser.add_argument("--constraint1", "-c1", action="store",
                        dest="c1", required=False, metavar="NOTE",
                        help="Melody will include this note.")
    parser.add_argument("--constraint2", "-c2", action="store",
                        dest="c2", required=False, metavar="NOTES",
                        help="Melody will include these notes.")
    parser.add_argument("--constraint3", "-c3", action="store",
                        dest="c3", required=False, metavar="NOTES",
                        help="Melody will be made of this notes only.")
    parser.add_argument("--constraint4", "-c4", action="store",
                        dest="c4", required=False, metavar="NOTES&LENGTHS",
                        help="Melody will include the phrase. Notes and "
                             "Lengths may be a list each. E.g. 12,13;1,.5")
    parser.add_argument("--bars", "-b", action="store", default="4", dest="b",
                        type=int, help="How many 4/4 bars to fill with melody")
    parser.add_argument("--input", "-i", action="store",
                        default="MTC-FS-1.0/good_midi/",
                        help="The set to learn from. May be a file or a dir."
                             '\nDefault is "MTC-FS-1.0/good_midi/"')
    parser.add_argument("--learn-limit", "-l",
                        action="store", type=int, dest="l", metavar="N",
                        help="Algorithm learn from at most N files found, "
                             "if input is specified as a dir. Defaults to "
                             "learn from all files found.")
    parser.add_argument("--output", "-o", action="store", default="output.mid",
                        help="Filename to store the result in.\nDefault is "
                             '"output.mid"')
    parser.add_argument("-v", "--verbose", action="store_const",
                        const=logging.INFO, dest="ll",
                        help="Set log level to verbose.")
    parser.add_argument("-d", "--debug", action="store_const",
                        const=logging.DEBUG, dest="ll",
                        help="Set log level to debug.")
    parser.add_argument("--play", "-p", action="store_const", const='p',
                        help="Flag: if set program will play the "
                             "newly generated melody.")
    parser.add_argument("--multi-output", "-m", default="1", dest="m", type=int,
                        help="If set the algorithm will create m (different) "
                        "melodies. The files will be named after given"
                        "output parameter but suffixed with numbers from 0 to "
                        "m-1.")
    parser.add_argument("--no-shuffle", action="store_true", dest="no_shuffle",
                        help="If the flag is set and the number of training "
                             "melodies is limited (-l / --learn-limit), the "
                             "melodies learns from will be the first n ones in "
                             "of the input folder.")
    parser.add_argument("--order", default=4, dest="mm_order", type=int,
                        metavar="O",
                        help="Specify the order of the markov model, that is "
                             "used. Default is 4.")
    parser.add_argument("--cluster-count", default=17, dest='n_cluster',
                        type=int, metavar='CC',
                        help="Specifies the number of clusters used to for the "
                             "contours. Defaults to 17.")

    args = parser.parse_args()
    if args.ll is not None:
        log.setLevel(args.ll)

    # strange issue with logger or I simply don't get it -.-'
    # if you leave out the next line the main logger won't print to stdout
    logging.debug("Staring debugger main")


    def count_non(*args):
        return sum([a is not None for a in args])
    if count_non(args.c1, args.c2, args.c3, args.c4) > 1:
        raise ValueError("Cannot define more than one constraint at a time!")

    main(args)
