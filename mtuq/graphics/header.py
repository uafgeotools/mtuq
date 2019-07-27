
import numpy as np
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
from mtuq.event import MomentTensor




class Header(dict):
    """ A dictionary-like object containing header text

    Stores header text in a dictionary {position: text}, where position is an
    integer and text is a string. (This syntax is inspired by 
    matplotlib.pyplot.subplot.)

    .. example:

       Create a text header with one row and two columns:

           Header({1: 'text of column1', 2: 'text of column2'}, shape=[1,2])


    """
    def __init__(self, items, shape=np.array([])):
        super(Header, self).__init__(items)
        self.shape = shape


def attach_header(title=None, header=None, mt=None, height=None):
    """ Adds header object to current figure
    """
    fig = pyplot.gcf()
    width, figure_height = fig.get_size_inches()

    x0 = 0.
    y0 = 1.-height/figure_height
    ax = fig.add_axes([x0, y0, 1., height/figure_height])
    ax.set_xlim([0., width])
    ax.set_ylim([0., height])
    ax = pyplot.gca()

    if title:
        _bold(title, 0.15, 0.7, ax, fontsize=16)

    for _i, text in header.items():
        ix = _i % header.shape[1]
        iy = _i / header.shape[1]
        px = float(ix) / header.shape[1]
        py = 1. - float(iy) / header.shape[0]
        px *= 0.4
        px += 0.15
        if title:
            py -= 0.05
            py *= 0.5
        _text(text, px, py, ax, fontsize=14)

    # add beachball
    from obspy.imaging.beachball import beach
    beach = beach(mt, xy=(1.15, 1.), width=1.75, linewidth=0.5, facecolor=_light_gray)
    ax = pyplot.gca()
    ax.add_collection(beach)

    _hide_axes(ax)


def generate_header(event_name, process_bw, process_sw, misfit_bw, misfit_sw,
    model, solver, mt, depth_in_m):
    """ Creates header object with CAP-style text
    """
    M0 = np.sqrt(0.5*np.sum(mt[0:3]**2.) + np.sum(mt[3:6]**2.))
    Mw = (np.log10(M0) - 9.1)/1.5

    norm_order = misfit_bw.norm_order
    assert norm_order==misfit_sw.norm_order
    norm = '$L%s$' % norm_order

    bw_T_min = process_bw.freq_max**-1
    bw_T_max = process_bw.freq_min**-1
    sw_T_min = process_sw.freq_max**-1
    sw_T_max = process_sw.freq_min**-1

    bw_win_len = process_bw.window_length
    sw_win_len = process_sw.window_length

    return Header(
        shape=np.array([5,4]),
        items={
            0: '$M_w$: %3.2f' % Mw,
            1: 'depth: %.1f km' % (depth_in_m/1000.),
            #2: 'CLVD: %.0f' % 0.,
            #3: 'ISO: %.0f' % 0.,
            4: 'model: %s' % model,
            5: 'solver: %s' % solver,
            12: 'b.w. passband: %.1f - %.1f s' % (bw_T_min, bw_T_max),
            14: 's.w. passband: %.1f - %.1f s' % (sw_T_min, sw_T_max),
            16: 'b.w. window: %.1f s' % bw_win_len,
            18: 's.w. window: %.1f s' % sw_win_len})



_light_gray = [0.667, 0.667, 0.667]
_dark_gray = [0.333, 0.333, 0.333]

def _text(text, x, y, ax, fontsize=12):
    pyplot.text(x, y, text, fontsize=fontsize, transform=ax.transAxes)


def _bold(text, x, y, ax, fontsize=12):
    font = FontProperties()
    font.set_weight('bold')
    pyplot.text(x, y, text, fontproperties=font, fontsize=fontsize,
        transform=ax.transAxes)


def _italic(text, x, y, ax, fontsize=12):
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

