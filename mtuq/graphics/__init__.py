

from mtuq.graphics.attrs import\
    plot_time_shifts, plot_amplitude_ratios

from mtuq.graphics.beachball import\
    plot_beachball, misfit_vs_depth

from mtuq.graphics.beachball_pygmt import\
    beachball_pygmt

from mtuq.graphics.summary import\
    plot_summary1, plot_summary2

from mtuq.graphics.uq.lune import\
    plot_misfit_lune, plot_likelihood_lune, plot_marginal_lune,\
    plot_variance_reduction_lune, plot_magnitude_tradeoffs_lune, _plot_lune

from mtuq.graphics.uq.vw import\
    plot_misfit_vw, plot_likelihood_vw, plot_marginal_vw, \
    _misfit_vw_regular, _likelihoods_vw_regular, _marginals_vw_regular, \
    _plot_vw, _product_vw

from mtuq.graphics.uq.double_couple import\
    plot_misfit_dc, plot_likelihood_dc, plot_marginal_dc

from mtuq.graphics.uq.force import\
    plot_misfit_force, plot_likelihood_force, plot_marginal_force,\
    plot_magnitude_tradeoffs_force, _plot_force

from mtuq.graphics.uq.origin_depth import\
    plot_misfit_depth, plot_likelihood_depth, plot_marginal_depth

from mtuq.graphics.uq.origin_xy import\
    plot_misfit_xy, plot_mt_xy

from mtuq.graphics.waveforms import\
    plot_waveforms1, plot_waveforms2, plot_data_greens1, plot_data_greens2

from mtuq.graphics.uq import\
    likelihood_analysis


# use Helvetica if available
for fontname in ['Helvetica', 'Arial']:
    try:
        from matplotlib.font_manager import find_font
        find_font(fontname, fallback_to_default=False)
        matplotlib.rcParams['font.sans-serif'] = fontname
        matplotlib.rcParams['font.family'] = "sans-serif"
        break
    except:
        continue
