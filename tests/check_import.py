#!/usr/bin/env python

import sys
from importlib import import_module
from pkgutil import iter_modules

try:
    import matplotlib
except:
    matplotlib = False

try:
    import h5py
except:
    h5py = False


# how much detail in output messages?
verbosity = 1


# global variables
succeeded = []
excluded = []
failed = []


def check_package(pkgname):
    if pkgname in excluded:
        return
    package = import_module(pkgname)
    for _, modname, ispkg in iter_modules(package.__path__):
        full_dotted_name = pkgname+'.'+modname
        if ispkg:
            check_package(full_dotted_name)
        else:
            check_module(full_dotted_name)


def check_module(name):
    global succeeded
    global excluded
    global failed

    if name in excluded:
        return
    try:
        import_module(name)
    except NotImplementedError:
        excluded += [name]
    except Exception as e:
        if verbosity > 0:
            print name.upper()
            print e
            print '\n'
        failed += [name]
    else:
        succeeded += [name]


if __name__ == '__main__':
    # recursively check package
    check_package('mtuq')

    # disply results
    print '\n'
    print '%d modules were excluded' % len(excluded)
    print '\n'
    print '%d modules imported successfully' % len(succeeded)
    print '\n'
    print '%d modules failed to import' % len(failed)
    print '\n'

    if verbosity > 1:
        if excluded:
            print 'The following modules were excluded:'
            for name in excluded: print '  %s' % name
            print '\n'

        if succeeded:
            print 'The following modules imported successfully:'
            for name in succeeded: print '  %s' % name
            print '\n'

        if failed:
            print 'The following modules failed to import:'
            for name in failed: print '  %s' % name
            print '\n'
        
    if failed:
        sys.exit(-1)
    else:
        print 'SUCCESS\n'


