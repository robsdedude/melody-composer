# -*- coding: utf-8 -*-
"""
__author__ = 'Rouven Bauer, kathtn'
__copyright__ = "Fraunhofer IDMT"
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import re

from midiutil.MidiFile3 import MIDIFile
from mido import MidiFile
import mido

import logging
log = logging.getLogger("main")


class MIDI(object):
    """
    Read method not used in current version of this program. May be used in the
    future for reading generic monophonic midi files. Some bugs must be fixed
    then.

    This class will help to play, read and write MIDI files
    """

    @staticmethod
    def play_midi(fn):
        """
        Play back a midi file from disk.

        :type fn: basestring
        :param fn: Path to the midi file that should be played.
        :return: None
        """
        output = mido.open_output()
        for message in MidiFile(fn).play():
            output.send(message)
            log.debug(message)
        output.close()

    @staticmethod
    def read_midi(fn):
        """
        Read a midi file from disk.

        :type fn: basestring
        :param fn: Path to the file that is to be read.

        :return: list
        """
        # FIXME: can break if there are CCs in the middle of the track
        # this is because the time stamp of such messages is ignored.
        mid = MidiFile(fn)
        notes = []
        duration_ticks = []
        status = []
        pitch = []

        for i, track in enumerate(mid.tracks):
            for message in track:

                if message.type == 'note_on':
                    pitch = pitch + [message.note]
                    notes = notes + [message.note]
                    status = status + [message.type]
                    duration_ticks = duration_ticks + [message.time]

                elif message.type == 'note_off':
                    notes = notes + [message.note]
                    status = status + [message.type]
                    duration_ticks = duration_ticks + [message.time]
        duration_sec = []
        for i in range(len(duration_ticks)):
            if duration_ticks[i] >= 100:
                duration_sec = duration_sec + [duration_ticks[i] / 1920]
        # for finding the presence of rest in the MIDI sequence
        rest_index = []
        rest_value = []
        notes_rest = []
        for i in range(len(duration_ticks)):
            if status[i] == 'note_on':
                if duration_ticks[i - 1] > 100:
                    if duration_ticks[i] > 100:
                        if duration_ticks[i + 1] > 100:
                            rest_index = rest_index + [i]
                            rest_value = rest_value + [duration_ticks[i]]
                            notes_rest = notes_rest + ['R']
            else:
                notes_rest = notes_rest + [notes[i]]
        return [pitch, notes_rest, duration_ticks, duration_sec, status,
                rest_index, rest_value]

    @staticmethod
    def write_midi(fn, notes, duration):
        """
        Creates a midi file.

        :param fn: (str) path to file, where to store
        :param notes: (list) Containing the notes. Integers for the pitch or 'R'
            for rests.
        :param duration: (list) must have the same length as notes containing
            the durations of the notes/rest as floats where 1.0 is a quarter,
            0.5 an eighth not and so on.

        :return: None
        """

        midi = MIDIFile(1)
        track = 0
        bpm = 120
        log.debug("Writing Midi file {}:".format(fn))
        log.debug(notes)
        log.debug(duration)
        time = sum(duration) * 60 / bpm
        midi.addTrackName(track, time, "Sample Track")
        midi.addTempo(track, time, bpm)
        track = 0
        channel = 0
        volume = 100
        time = float(0)
        pitch = notes
        for x in range(len(notes)):
            if notes[x] == 'R':
                time += float(duration[x])
            else:
                midi.addNote(track, channel, pitch[x], time, duration[x],
                             volume)
                time += float(duration[x])
        binfile = open(fn, 'wb')
        midi.writeFile(binfile)
        binfile.close()


class MtcMidi(MIDI):
    """
    Class that helps to read midi files from the MTC data set
    http://www.liederenbank.nl/mtc/downloads/MeertensTuneCollections.pdf
    and splits the melodies into phrases.
    """

    phrase_regex = re.compile(r"!! verse \d+")

    @staticmethod
    def read_midi(fn, auto_transpose=True):
        """
        Read a midi file from the MTC data set and return it split into phrases.

        :type fn: basestring
        :param fn: The path to the midi file to be read.
        :type auto_transpose: bool
        :param auto_transpose: If set to True look for infixes like '_p3'
            between filename and type ending. End of fn must match
            "_([mp])(\d+)\.midi?$". _m(\d+) will cause the input to be
            transposed down wards by $0 semitones while _p(\d+) means transpose
            upwards by $0 semitones.

        :return: list of phrases
            Each phrase is a list of notes
            Each note is a (pitch, duration)-tuple.
        """
        if auto_transpose:
            m = re.findall("_([mp])(\d+)\.midi?$", fn)
            if len(m) == 1:
                m = m[0]
                d = int(m[1])
                d *= -1 if m[0] == "m" else 1

                def transpose(p):
                    return p+d if p != 'R' else p

        phrases = []

        def add_to_phrases(pitches, lengths):
            if 'transpose' in locals():
                phrases.append(zip(map(transpose, pitches), lengths))
            else:
                phrases.append(zip(pitches, lengths))

        log.debug("Reading %s" % fn)
        f = MidiFile(fn)
        tracks = f.tracks
        if len(tracks) < 2:
            raise IOError("Expected track %s to have at most 2 tracks but "
                          "found %i" % (fn, len(tracks)))
        phrase_annotation, melody = tracks[:2]

        # process the phrase track
        phrase_starts = []
        time = 0
        mspq = None  # microseconds per quarter note
        for message in phrase_annotation:
            time += message.time
            if (message.type == 'text' and
                    MtcMidi.phrase_regex.match(message.text)):
                phrase_starts.append(time)
            if message.type == 'set_tempo':
                if mspq:
                    raise IOError("Can only handle static tempo")
                if time:
                    raise IOError("Tempo must be set at the beginning")
                mspq = message.tempo
        if not mspq:
            mspq = 500000  # default value for midi
        bpm = 60000000/mspq
        # bpm *= 4  # quantize quarter note to equal 1

        # process the track with the melody
        current_note = None
        last_note_on = 0
        last_note_off = 0
        current_phrase = 0
        time = 0
        pitches = []
        lengths = []
        for message in melody:
            time += message.time

            if message.type == 'note_on' and message.velocity:
                if current_note:
                    raise IOError("Polyphonic tune, sorry...")
                if (current_phrase < len(phrase_starts) - 1 and
                        time == phrase_starts[current_phrase+1]):
                    # new phrase start is here
                    if last_note_off < time:  # there was a rest in last phrase
                        pitches.append('R')
                        lengths.append((time - last_note_off) / bpm)
                    add_to_phrases(pitches, lengths)
                    pitches = []
                    lengths = []
                    current_phrase += 1
                elif (current_phrase < len(phrase_starts) - 1 and
                        time > phrase_starts[current_phrase+1]):
                    # new phrase start was between this an last note event
                    if last_note_off < time:
                        # there was at least one rest
                        if last_note_off < phrase_starts[current_phrase+1]:
                            # there was a rest in last phrase
                            pitches.append('R')
                            lengths.append((phrase_starts[current_phrase+1] -
                                            last_note_off) / bpm)
                    add_to_phrases(pitches, lengths)
                    pitches = []
                    lengths = []
                    if time > phrase_starts[current_phrase+1]:
                        # there is also a rest in new phrase
                        pitches.append('R')
                        lengths.append((time -
                                        phrase_starts[current_phrase+1]) / bpm)
                    current_phrase += 1

                else:
                    # no new phrase
                    if last_note_off < time:  # there was a rest
                        pitches.append('R')
                        lengths.append((time - last_note_off) / bpm)
                current_note = message
                last_note_on = time

            if message.type == 'note_off' and message.velocity:
                if not current_note:
                    raise IOError("Can't stop not existing tones...")
                if (current_note.channel != message.channel or
                        current_note.note != message.note or
                        current_note.velocity != message.velocity):
                    raise IOError("Can't stop a different note...")
                pitches.append(current_note.note)
                lengths.append((time - last_note_on) / bpm)
                current_note = None
                last_note_off = time

        add_to_phrases(pitches, lengths)

        if current_note:
            raise IOError("Last note_off message is missing")

        return phrases

if __name__ == '__main__':
    print(MtcMidi.read_midi('MTC-FS-1.0/good_midi/NLB071052_01_p5.midi'))
