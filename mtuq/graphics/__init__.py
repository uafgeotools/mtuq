
import shutil
import subprocess

from mtuq.graphics.beachball import plot_beachball, misfit_vs_depth

from mtuq.graphics.uq.moment_tensor import plot_misfit_mt, plot_likelihood_mt, plot_marginal_mt
from mtuq.graphics.uq.double_couple import plot_misfit_dc, plot_likelihood_dc, plot_marginal_dc
from mtuq.graphics.uq.vw import plot_misfit_vw, plot_likelihood_vw, plot_marginal_vw
from mtuq.graphics.uq.force import plot_misfit_force, plot_likelihood_force, plot_marginal_force

from mtuq.graphics.waveform import plot_data_synthetics, plot_data_greens


