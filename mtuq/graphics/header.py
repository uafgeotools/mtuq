
import numpy as np
import os
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from mtuq.event import MomentTensor
from mtuq.graphics.beachball import gray, plot_beachball
#from mtuq.util.moment_tensor.TapeTape2015 import from_mij
from obspy.core import AttribDict


class Base(object):
    """ Base class for storing and writing text to a matplotlib figure
    """
    def __init__(self):
        raise NotImplementedError("Must be implemented by subclass")


    def _get_axis(self, height, fig=None):
        """ Returns matplotlib axes of given height along top of figure
        """
        if fig is None:
            fig = pyplot.gcf()
        width, figure_height = fig.get_size_inches()

        assert height < figure_height, Exception(
             "Header height exceeds entire figure height. Please double check "
             "input arguments.")
               
        x0 = 0.
        y0 = 1.-height/figure_height

        ax = fig.add_axes([x0, y0, 1., height/figure_height])
        ax.set_xlim([0., width])
        ax.set_ylim([0., height])

        # hides axes lines, ticks, and labels
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        return ax


    def write(self):
        raise NotImplementedError("Must be implemented by subclass")



class SimpleTextHeader(Base):
    """ Stores header text in a list [[text, position]], where text is a string
    and position is a (x,y) tuple
    """
    def __init__(self, items):
        # validates dictionary
        for text, p in items:
            assert type(text) in [str, unicode]
            assert 0. <= pos[0] <= 1.
            assert 0. <= pos[1] <= 1.
            pass

        self.items = items


    def write(self, ax):
        for key, val in self.items:
            text = key
            xp, yp = val
            _write_text(text, xp, yp, ax)


class UAFStyleHeader(Base):
    """ Stores information from a moment tensor inversion and writes UAF-style
    text to the top of a matplotlib figure
    """
    def __init__(self, event_name, process_bw, process_sw, misfit_bw, misfit_sw,
        model, solver, mt, origin):

        self.event_name = event_name
        self.magnitude = MomentTensor(mt).magnitude()
        self.depth_in_m = origin.depth_in_m
        self.depth_in_km = origin.depth_in_m/1000.
        self.model = model
        self.solver = solver
        self.mt = mt

        self.process_bw = process_bw
        self.process_sw = process_sw
        self.misfit_bw = process_bw
        self.misfit_sw = process_sw
        self.norm = misfit_bw.norm

        self.bw_T_min = process_bw.freq_max**-1
        self.bw_T_max = process_bw.freq_min**-1
        self.sw_T_min = process_sw.freq_max**-1
        self.sw_T_max = process_sw.freq_min**-1
        self.bw_win_len = process_bw.window_length
        self.sw_win_len = process_sw.window_length


    def add_beachball(self, ax, height, offset):

        #
        # If ObsPy plotted focal mechanisms correctly we could do the following
        #

        #from obspy.imaging.beachball import beach
        ## beachball size
        #diameter = 0.75*height
        #xp = 0.50*diameter + offset
        #yp = 0.45*height
        #ax.add_collection(
        #    beach(self.mt, xy=(xp, yp), width=diameter,
        #    linewidth=0.5, facecolor=gray))


        #
        # Instead, we must use this workaround
        #

        # beachball size
        diameter = 0.75*height

        # beachball placement
        xp = offset
        yp = 0.075*height

        plot_beachball('tmp.png', self.mt)
        img = pyplot.imread('tmp.png')

        os.remove('tmp.png')
        os.remove('tmp.ps')

        ax.imshow(img, extent=(xp,xp+diameter,yp,yp+diameter))


    def write(self, height, offset):
        """ Writes header text to current figure
        """
        ax = self._get_axis(height)

        # calculate focal mechanism
        kappa, sigma, theta, _, gamma, delta = 0., 0., 0., 0., 0., 0.

        # add beachball to upper left corner
        self.add_beachball(ax, height, offset)


        # write text line #1
        px = 0.125
        py = 0.65
        line = 'Event %s  Model %s  Depth %d km' % (
            self.event_name, self.model, self.depth_in_km)
        _write_text(line, px, py, ax, fontsize=14)


        # write text line #2
        px = 0.125
        py -= 0.175
        line = u'FM %d %d %d    $M_w$ %.1f   %s %d   %s %d   rms %.1e   VR %.1f' %\
                (kappa, sigma, theta, self.magnitude, u'\u03B3', gamma, u'\u03B4', delta, 0, 0)
        _write_text(line, px, py, ax, fontsize=14)


        # write text line #3
        px = 0.125
        py -= 0.175
        line = 'passbands (s):  bw %.1f - %.1f,  sw %.1f - %.1f   ' %\
                (self.bw_T_min, self.bw_T_max, self.sw_T_min, self.sw_T_max)
        line += 'win. len. (s):  bw %.1f,  sw %.1f   ' %\
                (self.bw_win_len, self.sw_win_len)
        _write_text(line, px, py, ax, fontsize=14)


        # write text line #4
        px = 0.125
        py -= 0.175
        line = 'norm %s   N %d Np %d Ns %d' %\
                (self.norm, 0, 0, 0,)
        _write_text(line, px, py, ax, fontsize=14)


class Header(UAFStyleHeader):
    """ Stores information from a moment tensor inversion and writes text to 
    the top of a matplotlib figure
    """
    def write(self, height, offset):
        """ Writes header text to current figure
        """
        ax = self._get_axis(height)
        self.add_beachball(ax, height, offset)


        # write text line #1
        px = 0.125
        py = 0.65
        line = '%s  $M_w$ %.1f  Depth %d km' % (
            self.event_name, self.magnitude, self.depth_in_km)
        _write_bold(line, px, py, ax, fontsize=16)


        # write text line #2
        px = 0.125
        py -= 0.175
        line = u'Model %s   Solver %s   %s norm %.1e   VR %1.3f' %\
                (self.model, self.solver, self.norm, 1., 0.)
        _write_text(line, px, py, ax, fontsize=14)


        # write text line #3
        px = 0.125
        py -= 0.175
        line = 'passbands (s):  bw %.1f - %.1f ,  sw %.1f - %.1f   ' %\
                (self.bw_T_min, self.bw_T_max, self.sw_T_min, self.sw_T_max)
        line += 'win. len. (s):  bw %.1f ,  sw %.1f   ' %\
                (self.bw_win_len, self.sw_win_len)
        _write_text(line, px, py, ax, fontsize=14)


        # write text line #4
        px = 0.125
        py -= 0.175
        line = '$M_{ij}$ = [%.2e  %.2e  %.2e  %.2e  %.2e  %.2e]' % tuple(self.mt)
        _write_text(line, px, py, ax, fontsize=12)



#
# utility functions
#

def _write_text(text, x, y, ax, fontsize=12):
    pyplot.text(x, y, text, fontsize=fontsize, transform=ax.transAxes)


def _write_bold(text, x, y, ax, fontsize=14):
    font = FontProperties()
    #font.set_weight('bold')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize,
        transform=ax.transAxes)


def _write_italic(text, x, y, ax, fontsize=12):
    font = FontProperties()
    font.set_style('italic')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize,
        transform=ax.transAxes)


