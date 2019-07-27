
import numpy as np
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from mtuq.event import MomentTensor


class TextHeader(dict):
    """ Stores text and writes it to a matplotlib figure

    Stores header text in a dictionary {text: position}, where text is a string
    and position is a (x,y) tuple
    """
    def __init__(self, items):
        # validates
        for text, p in items:
            # assert type(text) in [str, unicode]
            # assert type(pos[0]) in [float]
            # assert type(pos[1]) in [float]
            pass

        super(Header, self).__init__(items)


    def write(self, axis):
        for key, val in self.items():
            text = key
            xp, yp = val
            _write_text(text, xp, yp)



class OldStyleHeader(object):
    """ Stores information from a CAP-style figure header and writes stored
    information to a matplotlib figure
    """
    def __init__(self, event_name, process_bw, process_sw, misfit_bw, misfit_sw,
        model, solver, mt, origin):

        self.event_name = event_name
        self.process_bw = process_bw
        self.process_sw = process_sw
        self.misfit_bw = process_bw
        self.misfit_sw = process_sw
        self.mt = mt
        self.magnitude = MomentTensor(mt).magnitude()
        self.depth_in_m = origin.depth_in_m
        self.depth_in_km = origin.depth_in_m/1000.
        self.model = ''
        self.norm = '$L%d$' % misfit_bw.norm_order


    def write(self, height=None):
        """ Writes header text to current figure
        """
        # create matplotlib axis
        fig = pyplot.gcf()
        width, figure_height = fig.get_size_inches()
        x0 = 0.
        y0 = 1.-height/figure_height
        ax = fig.add_axes([x0, y0, 1., height/figure_height])
        ax.set_xlim([0., width])
        ax.set_ylim([0., height])
        _hide_axes(ax)

        bw_T_min = self.process_bw.freq_max**-1
        bw_T_max = self.process_bw.freq_min**-1
        sw_T_min = self.process_sw.freq_max**-1
        sw_T_max = self.process_sw.freq_min**-1
        bw_win_len = self.process_bw.window_length
        sw_win_len = self.process_sw.window_length

        # write beachball
        from obspy.imaging.beachball import beach
        ax.add_collection(beach(self.mt,
            xy=(1.15, 1.), width=1.75, linewidth=0.5, facecolor=_light_gray))

        # write line #1
        line = '%s    $M_w$ %.1f    %d km    Model $%s$' % (
            self.event_name, self.magnitude, self.depth_in_km, self.model)

        px = 0.15
        py = 0.7
        _write_bold(line, px, py, ax, fontsize=16)

        # write line #2
        py -= 0.15
        px = 0.15
        line = u'FM %d %d %d   \u03B3 %d   \u03B4 %d   rms %.1e   VR %.1f' %\
                (0, 0, 0, 0, 0, 0, 0)
        _write_text(line, px, py, ax, fontsize=14)

        # write line #3
        py -= 0.15
        line = 'passbands (s): bw  %.1f - %.1f,  sw  %.1f - %.1f' %\
                (bw_T_min, bw_T_max, sw_T_min, sw_T_max)
        _write_text(line, px, py, ax, fontsize=14)

        # write line #4
        py -= 0.15
        line = 'norm %s   bw %d   sw %d   N %d Np %d Ns %d' %\
                (self.norm, bw_win_len, sw_win_len, 0, 0, 0,)
        _write_text(line, px, py, ax, fontsize=14)



class NewStyleHeader(OldStyleHeader):
    """ Stores information from a CAP-style figure header and writes stored
    information to a matplotlib figure
    """
    def write(self, height=None):
        """ Writes header text to current figure
        """
        # create matplotlib axis
        fig = pyplot.gcf()
        width, figure_height = fig.get_size_inches()
        x0 = 0.
        y0 = 1.-height/figure_height
        ax = fig.add_axes([x0, y0, 1., height/figure_height])
        ax.set_xlim([0., width])
        ax.set_ylim([0., height])
        _hide_axes(ax)

        bw_T_min = self.process_bw.freq_max**-1
        bw_T_max = self.process_bw.freq_min**-1
        sw_T_min = self.process_sw.freq_max**-1
        sw_T_max = self.process_sw.freq_min**-1
        bw_win_len = self.process_bw.window_length
        sw_win_len = self.process_sw.window_length

        # write beachball
        from obspy.imaging.beachball import beach
        ax.add_collection(beach(self.mt,
            xy=(1.15, 1.), width=1.75, linewidth=0.5, facecolor=_light_gray))

        # write line #1
        line = '%s    $M_w$ %.1f    %d km' % (
            self.event_name, self.magnitude, self.depth_in_km)

        px = 0.15
        py = 0.7
        _write_bold(line, px, py, ax, fontsize=16)

        # write line #2
        py -= 0.15
        px = 0.15
        line = 'b.w. passband: %.1f - %.1f s' % (bw_T_min, bw_T_max)
        _write_text(line, px, py, ax, fontsize=14)

        px = 0.35
        line = 's.w. passband: %.1f - %.1f s' % (sw_T_min, sw_T_max)
        _write_text(line, px, py, ax, fontsize=14)

        # write line #3
        py -= 0.15
        px = 0.15
        line = 'b.w. win. len.: %.1f s' % bw_win_len
        _write_text(line, px, py, ax, fontsize=14)

        px = 0.35
        line = 's.w. win. len.: %.1f s' % sw_win_len
        _write_text(line, px, py, ax, fontsize=14)



#
# utility functions
#

def _write_text(text, x, y, ax, fontsize=12):
    pyplot.text(x, y, text, fontsize=fontsize, transform=ax.transAxes)


def _write_bold(text, x, y, ax, fontsize=12):
    font = FontProperties()
    font.set_weight('bold')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize,
        transform=ax.transAxes)


def _write_italic(text, x, y, ax, fontsize=12):
    font = FontProperties()
    font.set_style('italic')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize,
        transform=ax.transAxes)

def _hide_axes(ax):
    # hides axes lines, ticks, and labels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


_light_gray = [0.667, 0.667, 0.667]
_dark_gray = [0.333, 0.333, 0.333]

