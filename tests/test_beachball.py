#!/usr/bin/env python

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import numpy as np
from matplotlib.colors import Normalize, to_rgba
from matplotlib.markers import MarkerStyle

from mtuq import MomentTensor
from mtuq.graphics import plot_sp_ratios
from mtuq.graphics.beachball import (
    _plot_beachball_matplotlib, _plot_beachball_pygmt, _plot_stations)


def _observations(ratios, polarities=None, weights=None):
    ratios = np.asarray(ratios, dtype=float)
    n = ratios.size
    return (np.ones(n) if polarities is None else np.asarray(polarities),
        np.zeros(n), ratios, np.ones(n), np.ones(n),
        np.ones(n) if weights is None else np.asarray(weights))


def _predictions(log_ratios, polarities=None):
    log_ratios = np.atleast_2d(np.asarray(log_ratios, dtype=float))
    return (np.ones_like(log_ratios) if polarities is None else np.asarray(polarities),
        np.zeros_like(log_ratios), log_ratios)


class TestBeachballStationValues(unittest.TestCase):

    def setUp(self):
        self.mt = MomentTensor([1., -1., 0., 0., 0., 1.])
        self.origin = SimpleNamespace(
            latitude=0., longitude=0., depth_in_m=10000.)
        self.stations = [SimpleNamespace(
            station='S%d' % i, azimuth=azimuth, takeoff_angle=takeoff)
            for i, (azimuth, takeoff) in enumerate(
                [(90., 30.), (180., 120.), (0., 60.)])]

    def tearDown(self):
        pyplot.close('all')

    def test_log_residuals_and_missing_pairs(self):
        observed = _observations([1., 1., 0., np.nan, -1., 1.e300, 1.e-300])
        predicted = _predictions([0., 1., 0., 0., 0., -1000., 1000.])
        with patch('mtuq.graphics.beachball.plot_beachball') as plot:
            plot_sp_ratios(None, observed, predicted,
                [None]*7, self.origin, self.mt)
        values = plot.call_args.kwargs['station_values']
        np.testing.assert_allclose(values[[0, 1, 5, 6]], [0., 1., -1300., 1300.])
        self.assertTrue(np.isnan(values[2:5]).all())

    def test_ratio_weights_select_stations_without_scaling_colors(self):
        observed = _observations([1.]*8,
            weights=[1., 0., -1., np.nan, np.inf, 2., 1., 1.])
        predicted = _predictions([0., 1., 1., 1., 1., -0.5, 1., np.nan])
        with patch('mtuq.graphics.beachball.plot_beachball') as plot:
            plot_sp_ratios(None, observed, predicted, [None]*8, self.origin, self.mt)
        values = plot.call_args.kwargs['station_values']
        np.testing.assert_allclose(values[[0, 5, 6]], [0., -0.5, 1.])
        self.assertTrue(np.isnan(values[[1, 2, 3, 4, 7]]).all())

    def test_three_element_observations_assume_unit_weights(self):
        observed = _observations([1., 10., 0.], polarities=[1., -1., 0.])[:3]
        predicted = _predictions([0., 0., 0.], polarities=[[1., 1., 1.]])
        with patch('mtuq.graphics.beachball.plot_beachball') as plot:
            plot_sp_ratios(None, observed, predicted, self.stations, self.origin, self.mt)
        np.testing.assert_allclose(plot.call_args.kwargs['station_values'],
            [0., -1., np.nan], equal_nan=True)
        np.testing.assert_array_equal(plot.call_args.kwargs['polarity_data'],
            [[1., -1., 0.], [1., 1., 1.]])

    def test_invalid_inputs(self):
        with self.assertRaisesRegex(TypeError, 'observation bundle'):
            plot_sp_ratios(None, [1.], [1.], self.stations, self.origin, self.mt)
        with self.assertRaisesRegex(TypeError, 'get_predicted output'):
            plot_sp_ratios(None, _observations([1.]*3), [1.],
                self.stations, self.origin, self.mt)
        with self.assertRaisesRegex(ValueError, 'one value per station'):
            plot_sp_ratios(None, _observations([1.]), _predictions([0.]),
                self.stations, self.origin, self.mt)
        with self.assertRaisesRegex(ValueError, 'No usable'):
            plot_sp_ratios(None, _observations([0., np.nan, -1.]), _predictions([0.]*3),
                self.stations, self.origin, self.mt)
        with self.assertRaises(NotImplementedError):
            plot_sp_ratios(None, _observations([1.]*3), _predictions([0.]*3),
                self.stations, self.origin, self.mt, backend=_plot_beachball_pygmt)
        with self.assertRaisesRegex(ValueError, 'one source'):
            plot_sp_ratios(None, _observations([1.]*3), _predictions(np.zeros((2, 3))),
                self.stations, self.origin, self.mt)
        with self.assertRaisesRegex(ValueError, 'require stations'):
            _plot_beachball_matplotlib(None, self.mt,
                None, self.origin, station_values=[0.]*3)

    def test_supplied_geometry_and_symmetric_colors(self):
        fig, ax = pyplot.subplots()
        with patch('mtuq.graphics.beachball.TauPyModel') as taup:
            artist = _plot_stations(self.stations, self.origin, 'ak135', ax,
                station_values=[2., -1., np.nan])
        taup.assert_not_called()
        # Equal-area projection, including an upward ray folded downward.
        np.testing.assert_allclose(artist.get_offsets(),
            [[np.sqrt(2.)*np.sin(np.pi/12.), 0.], [0., 1./np.sqrt(2.)]],
            atol=1.e-15)
        np.testing.assert_array_equal(artist.get_array(), [2., -1.])
        self.assertEqual((artist.norm.vmin, artist.norm.vmax), (-2., 2.))
        self.assertEqual([text.get_text() for text in ax.texts], ['S0', 'S1'])
        np.testing.assert_array_equal(artist.get_sizes(), [25])

    def test_zero_residuals_and_optional_labels(self):
        fig, ax = pyplot.subplots()
        artist = _plot_stations(self.stations, self.origin, 'ak135', ax,
            station_values=[0.]*3, add_station_labels=False)
        self.assertEqual((artist.norm.vmin, artist.norm.vmax), (-1., 1.))
        self.assertEqual(len(ax.texts), 0)

    def test_geometry_falls_back_to_station_coordinates(self):
        fig, ax = pyplot.subplots()
        station = SimpleNamespace(station='NORTH', latitude=1., longitude=0.)
        with patch('mtuq.graphics.beachball.TauPyModel'), \
                patch('mtuq.graphics.beachball._takeoff_angle_taup', return_value=30.):
            artist = _plot_stations([station], self.origin, 'ak135', ax,
                station_values=[0.5])
        np.testing.assert_allclose(artist.get_offsets(),
            [[0., np.sqrt(2.)*np.sin(np.pi/12.)]], atol=1.e-15)

    def test_polarity_categories_keep_their_colors(self):
        fig, ax = pyplot.subplots()
        stations = [SimpleNamespace(station='S%d' % i,
            azimuth=60.*i, takeoff_angle=30.) for i in range(5)]
        _plot_stations(stations, self.origin, 'ak135', ax,
            polarity_data=np.array([[1., -1., 1., -1., 0.],
                                    [1., -1., -1., 1., 1.]]))
        expected = ['chartreuse', 'chartreuse', 'red', 'red', 'black']
        for artist, color in zip(ax.collections, expected):
            np.testing.assert_allclose(artist.get_facecolors()[0], to_rgba(color))
        self.assertEqual(len(ax.collections), 5)

    def test_colorbar_uses_the_station_normalization(self):
        norm = Normalize(vmin=-1., vmax=1.)
        plot_sp_ratios(None, _observations([1.]*3), _predictions([1., -1., 0.]),
            self.stations, self.origin, self.mt, station_norm=norm)
        fig = pyplot.gcf()
        self.assertEqual(len(fig.axes), 2)
        self.assertIs(fig.axes[0].collections[-1].norm, norm)
        np.testing.assert_allclose(fig.axes[1].get_xlim(), [-1., 1.])
        self.assertIn('log', fig.axes[1].get_xlabel())
        cmap = fig.axes[0].collections[-1].cmap
        self.assertEqual(cmap.name, 'blue_gray_orange')
        self.assertLess(max(cmap(norm(0.))[:3]), 0.4)

    def test_scalar_colors_use_fixed_size_circles_and_crosses_without_outlines(self):
        fig, ax = pyplot.subplots()
        _plot_stations(self.stations, self.origin, 'ak135', ax,
            station_values=[1., -1., 0.3],
            polarity_data=np.array([[1., -1., 0.], [-1., 1., 1.]]))
        fig.canvas.draw()
        for artist, marker, values in zip(ax.collections, ['o', 'x'], [[1., -1.], [0.3]]):
            np.testing.assert_array_equal(artist.get_sizes(), [25])
            np.testing.assert_array_equal(artist.get_array(), values)
            style = MarkerStyle(marker)
            np.testing.assert_allclose(artist.get_paths()[0].vertices,
                style.get_path().transformed(style.get_transform()).vertices)
            np.testing.assert_allclose(artist.get_facecolors(),
                artist.cmap(artist.norm(values)))
            if marker != 'x':
                self.assertEqual(len(artist.get_edgecolors()), 0)
        self.assertEqual(ax.collections[0].cmap.name, 'blue_gray_orange')
        self.assertIs(ax.collections[0].norm, ax.collections[1].norm)

    def test_ratios_render_a_separate_figure_with_colorbar(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory)/'ratios.png'
            plot_sp_ratios(str(filename), _observations([1.]*3, polarities=[1., -1., 1.]),
                _predictions([1., -1., 0.], polarities=[1., 1., 1.]),
                self.stations, self.origin, self.mt,
                add_station_labels=False)
            image = pyplot.imread(filename)
        self.assertGreater(np.max(image[:, :, 0] - image[:, :, 2]), 0.3)
        self.assertGreater(np.max(image[:, :, 2] - image[:, :, 0]), 0.3)
        self.assertEqual(pyplot.get_fignums(), [])


if __name__ == '__main__':
    unittest.main()
