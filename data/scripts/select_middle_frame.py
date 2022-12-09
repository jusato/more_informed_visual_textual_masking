#!/usr/bin/env python

import os
import sys
from pathlib import Path

if __name__ == '__main__':
    splits = [[]]
    tot = 0
    with open(sys.argv[1]) as f:
        s = 0
        c = 0
        for line in f:
            frames = list(Path(line.strip()).resolve().glob('*.jpg'))
            middle_frame = int(len(frames)/2)
            if middle_frame < 1:
                middle_frame = 1
            frames2 = list(Path(line.strip()).resolve().glob('0000{:02d}.jpg'.format(middle_frame)))
            splits[s].extend(sorted([str(f) for f in frames2]))
            c += 1

            if c > 99000:
                tot += c
                print(tot)
                s += 1
                c = 0
                splits.append([])

    for i, s in enumerate(splits):
        if len(s) > 0:
            with open('tmp3/seglist.{:02d}'.format(i), 'w') as f:
                for x in s:
                    f.write('{}\n'.format(x))
