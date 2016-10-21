from collections import defaultdict
from fractions import Fraction

__author__ = 'rouven'

import os
import sys
import shutil
import ly.pitch
import ly.document
import ly.music
import ly.music.items
import ly.music.event

if __name__ != '__main__':
    sys.stderr.write('Script cannot be imported!\n')
    sys.exit(1)

halfsteps = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7, 5: 9, 6: 11}

modes = defaultdict(lambda: 0)

fns = os.listdir(os.path.join("MTC-FS-1.0", "ly"))
max_en = len(fns)
hits = 0
ly_path = os.path.join('MTC-FS-1.0', 'ly')
midi_path = os.path.join('MTC-FS-1.0', 'midi')
good_midi_path = os.path.join('MTC-FS-1.0', 'good_midi')
timesigs = defaultdict(lambda: 0)
for i, fn in enumerate(fns):
    path = os.path.join(ly_path, fn)
    name, ext = os.path.splitext(os.path.basename(fn))
    song = ly.music.document(ly.document.Document.load(path))
    nodes = [node for node in song.iter_music() if
             isinstance(node, ly.music.items.Partial) or
             (isinstance(node, ly.music.items.TimeSignature) and
             node.token == '\\time')]
    ts = None
    ky = None
    for node in nodes:
        if isinstance(node, ly.music.items.TimeSignature):
            ts = str(node.numerator())+"/"+str(node.fraction().denominator)
            timesigs[ts] += 1
            break

    ks = song.find_child(ly.music.items.KeySignature)
    key_ext = ""
    if ks:
        note, mode = list(ks.descendants())
        modes[str(mode.token)] += 1
        ky = str(mode.token)
        if ts == "4/4":
            if str(mode.token) == "\\major":
                d = ((halfsteps[note.pitch.note]+note.pitch.alter*2+5) % 12) - 5
                d = int(d) * -1
                m = "m" if d < 0 else "p"
                key_ext = "_" + m + str(abs(d))
                hits += 1
                print path
                src = os.path.join(midi_path, name+".mid")
                dst = os.path.join(good_midi_path, name+key_ext+".midi")
                shutil.copyfile(src, dst)
                print ""
                print "========================================================"
                print "NEXT"
                print "========================================================"
    print "%i/%i    hits: %i  (%5s, %10s)" % (i, max_en, hits, ts, ky)

print timesigs
print modes

