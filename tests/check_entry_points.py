#!/usr/bin/env python

from pkg_resources import iter_entry_points

if __name__ == '__main__':

    for key in ['readers', 'greens_tensor_clients']:

        print key.upper()

        for entry_point in iter_entry_points(key):

            try:
                print '    %s' % entry_point.name
            except:
                raise Exception

            entry_point.load()

        print ''





