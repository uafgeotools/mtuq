#!/usr/bin/env python

import csv

colormaps = [
    'lajolla',
    'viridis',
    ]

if __name__=='__main__':
    for colormap in colormaps:

        lines = []
        with open(colormap+'.cpt') as file:
            reader = csv.reader(file,
                delimiter='\t',
                skipinitialspace=True)

            for row in reader:
                if row[0].startswith(('#','N','B','F')):
                    continue

                row[0] = str(1.-float(row[0]))
                row[2] = str(1.-float(row[2]))

                lines += ['\t'.join([row[2], row[1], row[0], row[3]])+'\n']

        lines.reverse()
        with open(colormap+'_r.cpt', "w") as file:
            file.writelines(lines)


