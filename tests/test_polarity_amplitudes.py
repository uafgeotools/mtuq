#!/usr/bin/env python

#
# Tests PolarityPampSPampRatioMisfit
#
# The radiation patterns are checked against analytic results that are
# independent of the implementation (Aki and Richards, 2ed):
#
#   - an explosion radiates no S
#   - R_P and total S radiation obey R_P^2 + R_S^2 = |M.gamma|^2
#   - averaged over the focal sphere, a double couple with M_0 = 1 gives
#     rms radiation coefficients sqrt(4/15) for P and sqrt(2/5) for S
#
# Recovery observations use NED matrix projections rather than the USE
# radiation calculation under test.
# Runs offline; no data files are required.
#

import unittest
from unittest.mock import patch

import numpy as np
import obspy

from mtuq import Origin, Station, MomentTensor
from mtuq.greens_tensor.base import GreensTensor, GreensTensorList
from mtuq.grid import DoubleCoupleGridRegular, FullMomentTensorGridSemiregular,\
    UnstructuredGrid
from mtuq.grid_search import grid_search
from mtuq.misfit.polarity import (
    PolarityMisfit, _radiation_coef_P, _takeoff_angle_taup,
    _takeoff_angles_taup, _polarities_mt)
from mtuq.misfit.polarity_amplitudes import (
    PolarityPampSPampRatioMisfit, polarities_pamp_spamp_ratio_from_dict,
    _misfit_term, _radiation_coef_S, _hypocentral_distances,
    _medium_from_taup)
from obspy.taup import TauPyModel


EPSVAL = 1.e-12


def _ned_matrices(mt):
    """ Independent matrix representation used only by the reference tests
    """
    return np.array([[
        [m[1], -m[5], m[3]],
        [-m[5], m[2], -m[4]],
        [m[3], -m[4], m[0]],
        ] for m in mt])


def _greens(n=4, depth_in_m=10000.):
    """ Builds Green's tensors carrying real stations and origin

    Only the geometry is used by this misfit function, so the traces
    themselves are placeholders.
    """
    origin = Origin({
        'time': obspy.UTCDateTime(2009, 4, 7),
        'latitude': 61.4542, 'longitude': -149.7428,
        'depth_in_m': depth_in_m})

    trace = obspy.Trace(np.zeros(64))
    trace.stats.delta = 0.1

    tensors = []
    for _i in range(n):
        station = Station({
            'network': 'XX', 'station': 'S%d' % _i, 'location': '',
            'latitude': 61.4542 + 0.4*(_i + 1),
            'longitude': -149.7428 + 0.3*_i,
            'npts': 64, 'delta': 0.1,
            'starttime': origin.time, 'endtime': origin.time + 6.3,
            'id': 'XX.S%d.' % _i})

        traces = []
        for component in ('Z', 'R', 'T'):
            for _j in range(6):
                copy = trace.copy()
                copy.stats.channel = '%s%d' % (component, _j)
                copy.stats.network = station.network
                copy.stats.station = station.station
                traces += [copy]

        tensors += [GreensTensor(traces=traces, station=station,
            origin=origin, tags=['units:m', 'type:displacement',
            'model:ak135'])]

    return GreensTensorList(tensors, id='test'), origin


class TestRadiationPatterns(unittest.TestCase):

    def test_explosion_radiates_no_shear(self):
        explosion = np.array([[1., 1., 1., 0., 0., 0.]])
        takeoff = np.linspace(1., 179., 37)
        azimuth = np.linspace(0., 350., 37)

        rs = _radiation_coef_S(explosion, takeoff, azimuth)

        self.assertLess(rs.max(), EPSVAL)


    def test_orthonormal_basis_identity(self):
        # R_P, R_SV and R_SH are the components of M.gamma in an
        # orthonormal basis, so their squares must sum to |M.gamma|^2
        rng = np.random.default_rng(3)
        mt = rng.normal(size=(5, 6))
        takeoff = rng.uniform(1., 179., 9)
        azimuth = rng.uniform(0., 360., 9)

        rp = _radiation_coef_P(mt, takeoff, azimuth)
        rs = _radiation_coef_S(mt, takeoff, azimuth)
        ned = _ned_matrices(mt)

        for _i, (theta, phi) in enumerate(
                zip(np.deg2rad(takeoff), np.deg2rad(azimuth))):

            gamma = np.array([
                np.sin(theta)*np.cos(phi),
                np.sin(theta)*np.sin(phi),
                np.cos(theta)])

            total = rp[:, _i]**2 + rs[:, _i]**2
            expected = np.sum(
                np.einsum('nab,b->na', ned, gamma)**2, axis=1)

            self.assertTrue(np.allclose(total, expected, rtol=1.e-10))


    def test_focal_sphere_averages(self):
        # Aki and Richards 2ed: a double couple with M_0 = 1 has rms
        # radiation coefficients sqrt(4/15) for P and sqrt(2/5) for S
        double_couple = np.array([[0., 0., 0., 0., 0., 1.]])

        rng = np.random.default_rng(11)
        n = 200000
        takeoff = np.rad2deg(np.arccos(rng.uniform(-1., 1., n)))
        azimuth = np.rad2deg(rng.uniform(0., 2.*np.pi, n))

        rp = _radiation_coef_P(double_couple, takeoff, azimuth)[0]
        rs = _radiation_coef_S(double_couple, takeoff, azimuth)[0]

        self.assertAlmostEqual(
            np.sqrt(np.mean(rp**2)), np.sqrt(4./15.), places=2)
        self.assertAlmostEqual(
            np.sqrt(np.mean(rs**2)), np.sqrt(2./5.), places=2)


    def test_frame_agrees_with_polarity_module(self):
        # Check the USE coefficients against an independent NED matrix
        # contraction, including the signs of off-diagonal components.
        rng = np.random.default_rng(7)
        mt = rng.normal(size=(6, 6))
        takeoff = rng.uniform(1., 179., 8)
        azimuth = rng.uniform(0., 360., 8)

        expected = _radiation_coef_P(mt, takeoff, azimuth)

        ned = _ned_matrices(mt)
        observed = np.zeros_like(expected)
        for _i, (theta, phi) in enumerate(
                zip(np.deg2rad(takeoff), np.deg2rad(azimuth))):
            gamma = np.array([
                np.sin(theta)*np.cos(phi),
                np.sin(theta)*np.sin(phi),
                np.cos(theta)])
            observed[:, _i] = np.einsum('nab,a,b->n', ned, gamma, gamma)

        self.assertTrue(np.allclose(observed, expected, rtol=1.e-10))


    def test_polarity_is_sign_of_radiation(self):
        rng = np.random.default_rng(5)
        mt = rng.normal(size=(9, 6))
        takeoff = rng.uniform(1., 179., 5)
        azimuth = rng.uniform(0., 360., 5)

        self.assertTrue(np.array_equal(
            _polarities_mt(mt, takeoff, azimuth),
            np.sign(_radiation_coef_P(mt, takeoff, azimuth))))



class TestRayGeometry(unittest.TestCase):

    def test_S_takeoff_angle_differs_from_P(self):
        # Direct P and S arrivals can have different takeoff angles.
        taup = TauPyModel('ak135')

        for depth_in_km, distance_in_deg in [(10., 1.5), (30., 2.0)]:
            angle_P = _takeoff_angle_taup(
                taup, depth_in_km, distance_in_deg, phase_list=['p', 'P'])
            angle_S = _takeoff_angle_taup(
                taup, depth_in_km, distance_in_deg, phase_list=['s', 'S'])

            self.assertGreater(abs(angle_P - angle_S), 1.)


    def test_default_phase_list_is_unchanged(self):
        taup = TauPyModel('ak135')

        self.assertEqual(
            _takeoff_angle_taup(taup, 10., 1.5),
            _takeoff_angle_taup(taup, 10., 1.5, phase_list=['p', 'P']))


    def test_distance_is_hypocentral(self):
        greens, origin = _greens(n=3, depth_in_m=10000.)

        expected = np.array([np.sqrt(
            tensor.distance_in_m**2 + origin.depth_in_m**2)
            for tensor in greens])

        self.assertTrue(np.allclose(
            _hypocentral_distances(greens), expected))

        # depth must actually enter, otherwise this is epicentral distance
        shallow, _ = _greens(n=3, depth_in_m=1.)
        self.assertFalse(np.allclose(
            _hypocentral_distances(greens),
            _hypocentral_distances(shallow)))



class TestMisfitTerms(unittest.TestCase):

    def test_polarity_term_eq4(self):
        # three stations, one mismatched; the 1/2 factor turns a
        # difference of 2 into a count of 1
        observed = np.array([1., -1., 1.])
        predicted = np.array([[1., -1., -1.]])
        weights = np.ones(3)

        # sum(0.5*tau*|pol_obs-pol_theo|) / sum(tau*|pol_obs|) = 1/3
        self.assertAlmostEqual(
            _misfit_term(observed, predicted, weights, 'L1', 'polarity',
                scale=0.5)[0],
            1./3.)


    def test_polarity_term_weighted(self):
        observed = np.array([1., -1., 1.])
        predicted = np.array([[1., -1., -1.]])
        weights = np.array([1., 1., 3.])

        # numerator 0.5*3*2 = 3, denominator 1+1+3 = 5
        self.assertAlmostEqual(
            _misfit_term(observed, predicted, weights, 'L1', 'polarity',
                scale=0.5)[0],
            3./5.)


    def test_amplitude_term_eq5(self):
        observed = np.array([2., -4.])
        predicted = np.array([[3., -1.]])
        weights = np.ones(2)

        # (|3-2| + |-1+4|) / (2+4) = 4/6
        self.assertAlmostEqual(
            _misfit_term(observed, predicted, weights, 'L1',
                'P amplitude')[0],
            4./6.)


    def test_ratio_term_eq6(self):
        # log10 values, one of which is zero and must still be used
        observed = np.array([0., 1.])
        predicted = np.array([[0.5, 0.5]])
        weights = np.ones(2)

        # (|0.5-0| + |0.5-1|) / (0+1) = 1.0
        self.assertAlmostEqual(
            _misfit_term(observed, predicted, weights, 'L1',
                'S/P amplitude ratio', allow_zero=True)[0],
            1.0)


    def test_zero_marks_unused_except_for_ratios(self):
        observed = np.array([0., 1.])
        predicted = np.array([[5., 2.]])
        weights = np.ones(2)

        # the zero entry is skipped, leaving |2-1|/1
        self.assertAlmostEqual(
            _misfit_term(observed, predicted, weights, 'L1',
                'P amplitude')[0],
            1.0)


    def test_enabled_term_without_data_raises(self):
        with self.assertRaises(ValueError):
            _misfit_term(np.zeros(3), np.ones((1, 3)), np.ones(3), 'L1',
                'P amplitude')



class TestMisfitFunction(unittest.TestCase):

    def test_returns_column_of_scalars(self):
        greens, _ = _greens(n=4)
        sources = DoubleCoupleGridRegular(npts_per_axis=5, magnitudes=[4.5])

        misfit = PolarityPampSPampRatioMisfit(
            lambda_pol=1., lambda_pamp=1., lambda_sprat=1.)

        observed = (
            np.array([1., -1., 1., -1.]),
            np.array([2.e-6, -3.e-6, 1.e-6, -4.e-6]),
            np.array([1.2, 0.8, 1.5, 0.9]))

        values = misfit(observed, greens, sources)

        self.assertEqual(values.shape, (len(sources), 1))
        self.assertEqual(values.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(values)))


    def test_weights_combine_linearly(self):
        greens, _ = _greens(n=4)
        sources = DoubleCoupleGridRegular(npts_per_axis=4, magnitudes=[4.5])

        observed = (
            np.array([1., -1., 1., -1.]),
            np.array([2.e-6, -3.e-6, 1.e-6, -4.e-6]),
            np.array([1.2, 0.8, 1.5, 0.9]))

        def evaluate(*weights):
            return PolarityPampSPampRatioMisfit(
                lambda_pol=weights[0], lambda_pamp=weights[1],
                lambda_sprat=weights[2])(observed, greens, sources)

        total = evaluate(1., 1., 1.)
        separate = (evaluate(1., 0., 0.) + evaluate(0., 1., 0.) +
            evaluate(0., 0., 1.))

        self.assertTrue(np.allclose(total, separate))


    def test_single_moment_tensor_accepted(self):
        greens, _ = _greens(n=3)
        source = MomentTensor(np.array([1., -1., 0., 0., 0., 0.5]))

        misfit = PolarityPampSPampRatioMisfit(
            lambda_pol=1., lambda_pamp=0., lambda_sprat=0.)

        values = misfit(
            (np.array([1., -1., 1.]), np.zeros(3), np.zeros(3)),
            greens, source)

        self.assertEqual(values.shape, (1, 1))


    def test_mismatched_lengths_raise(self):
        greens, _ = _greens(n=4)
        sources = DoubleCoupleGridRegular(npts_per_axis=3, magnitudes=[4.5])

        misfit = PolarityPampSPampRatioMisfit(
            lambda_pol=1., lambda_pamp=0., lambda_sprat=0.)

        with self.assertRaises(ValueError):
            misfit((np.ones(3), np.zeros(3), np.zeros(3)), greens, sources)


    def test_ratios_are_linear_not_logarithmic(self):
        # Input [1, 10] becomes [0, 1] in the residual, giving
        # (|0.5-0| + |0.5-1|) / (|0| + |1|) = 1.
        greens, _ = _greens(n=2)
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pol=0., lambda_pamp=0., lambda_sprat=1.)
        predicted = (np.ones((1, 2)), None, np.full((1, 2), 0.5))
        with patch.object(misfit, '_get_predicted', return_value=predicted):
            values = misfit((np.zeros(2), np.zeros(2), np.array([1., 10.])),
                greens, MomentTensor([1., -1., 0., 0., 0., 0.]))
        np.testing.assert_allclose(values, [[1.]])


    def test_all_unit_ratios_have_degenerate_normalization(self):
        greens, _ = _greens(n=2)
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pol=0., lambda_pamp=0.)
        with self.assertRaisesRegex(ValueError, 'normalizing sum is zero'):
            misfit((np.zeros(2), np.zeros(2), np.ones(2)), greens,
                MomentTensor([1., -1., 0., 0., 0., 0.5]))


    def test_negative_ratio_rejected(self):
        greens, origin = _greens(n=2)
        stations = [tensor.station for tensor in greens]

        with self.assertRaises(ValueError):
            polarities_pamp_spamp_ratio_from_dict(
                {'S0': 1., 'S1': -1.}, {'S0': 1.e-6, 'S1': 1.e-6},
                {'S0': 1.0, 'S1': -0.5}, stations)



class TestIndependentValidation(unittest.TestCase):

    def test_l2_station_weights_enter_linearly(self):
        # One unit residual at weight 1; normalization is 1 + 3.
        values = _misfit_term(np.ones(2), np.array([[2., 1.]]),
            np.array([1., 3.]), 'L2', 'P amplitude')
        np.testing.assert_allclose(values, [0.25])


    def test_polarity_only_agrees_with_existing_misfit(self):
        greens, _ = _greens(n=5)
        sources = DoubleCoupleGridRegular(npts_per_axis=4, magnitudes=[4.5])
        polarities = np.array([1., -1., 0., 1., -1.])
        data = (polarities, np.zeros(5), np.zeros(5))
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pamp=0., lambda_sprat=0.)

        expected = PolarityMisfit()(polarities, greens, sources)/4.
        np.testing.assert_allclose(misfit(data, greens, sources), expected)


    def test_grid_search_preserves_source_order_and_values(self):
        greens, origin = _greens(n=4)
        regular = DoubleCoupleGridRegular(
            npts_per_axis=3, magnitudes=[4.4, 4.5])
        coordinates = regular.to_array()
        # Deliberately reorder sources so sorting errors cannot hide.
        coordinates = coordinates[::-1]
        unstructured = UnstructuredGrid(dims=regular.dims,
            coords=tuple(coordinates[:, i] for i in range(6)),
            callback=regular.callback)
        observed = (np.array([1., -1., 1., -1.]),
            np.array([2.e-6, -3.e-6, 1.e-6, -4.e-6]),
            np.array([1.2, 0.8, 1.5, 0.9]))
        misfit = PolarityPampSPampRatioMisfit()

        for sources in (regular, unstructured):
            with self.subTest(grid=type(sources).__name__):
                direct = misfit(observed, greens, sources)
                results = grid_search(observed, greens, misfit, origin,
                    sources, timed=False, verbose=0, msg_interval=0)
                np.testing.assert_allclose(
                    results.values.ravel(), direct.ravel())
                self.assertEqual(results.source_idxmin(), int(direct.argmin()))


    def test_recovery_from_independent_matrix_predictions(self):
        greens, origin = _greens(n=8, depth_in_m=33000.)
        sources = FullMomentTensorGridSemiregular(
            npts_per_axis=3, magnitudes=[4.2, 4.5, 4.8])
        true_source = sources.get(len(sources)//2 + 17)
        matrix = _ned_matrices(true_source.as_vector()[None, :])[0]
        taup = TauPyModel('ak135')
        angles_p = _takeoff_angles_taup(taup, greens, phase_list=['p', 'P'])
        angles_s = _takeoff_angles_taup(taup, greens, phase_list=['s', 'S'])
        azimuths = np.deg2rad([tensor.azimuth for tensor in greens])

        def ray_directions(angles):
            theta = np.deg2rad(angles)
            return np.column_stack((np.sin(theta)*np.cos(azimuths),
                np.sin(theta)*np.sin(azimuths), np.cos(theta)))

        ray_p, ray_s = ray_directions(angles_p), ray_directions(angles_s)
        theta_s = np.deg2rad(angles_s)
        sv_basis = np.column_stack((np.cos(theta_s)*np.cos(azimuths),
            np.cos(theta_s)*np.sin(azimuths), -np.sin(theta_s)))
        sh_basis = np.column_stack((-np.sin(azimuths), np.cos(azimuths),
            np.zeros(len(greens))))
        rp = np.einsum('ni,ij,nj->n', ray_p, matrix, ray_p)
        rsv = np.einsum('ni,ij,nj->n', sv_basis, matrix, ray_s)
        rsh = np.einsum('ni,ij,nj->n', sh_basis, matrix, ray_s)

        rho, vp, vs = 2300., 5400., 3300.
        distance = np.sqrt(np.array([g.distance_in_m for g in greens])**2 +
            origin.depth_in_m**2)
        observed = (np.sign(rp), rp/(4.*np.pi*rho*vp**3*distance),
            (vp/vs)**3*np.sqrt(rsv**2 + rsh**2)/np.abs(rp))
        self.assertTrue(all(np.isfinite(array).all() for array in observed))
        # Keep this recovery away from nodes: it tests the unfloored theory.
        source_norm = np.linalg.norm(matrix)
        self.assertGreater(np.min(np.abs(rp)), 1.e-6*source_norm)
        self.assertGreater(np.min(np.hypot(rsv, rsh)), 1.e-6*source_norm)

        misfit = PolarityPampSPampRatioMisfit(rho=rho, vp=vp, vs=vs)
        np.testing.assert_allclose(misfit(observed, greens, true_source),
            [[0.]], rtol=0., atol=EPSVAL)
        results = grid_search(observed, greens, misfit, origin, sources,
            timed=False, verbose=0, msg_interval=0)
        self.assertLess(float(results.min()), EPSVAL)
        recovered = sources.get(results.source_idxmin())
        # Equivalent parameterizations can share a tensor; compare tensors,
        # rather than requiring a particular index in the parameter grid.
        np.testing.assert_allclose(recovered.as_vector()/source_norm,
            true_source.as_vector()/source_norm, rtol=0., atol=EPSVAL)


class TestPredictionDependencies(unittest.TestCase):

    def test_polarity_only_needs_no_s_geometry_or_medium(self):
        greens, _ = _greens(n=3)
        source = MomentTensor([1., -1., 0., 0., 0., 0.5])
        data = (np.ones(3), np.zeros(3), np.zeros(3))
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pamp=0., lambda_sprat=0.)
        module = 'mtuq.misfit.polarity_amplitudes.'

        with patch(module + '_takeoff_angles_taup',
                return_value=np.full(3, 45.)) as angles, \
                patch.object(misfit, '_medium') as medium, \
                patch(module + '_radiation_coef_S') as shear, \
                patch(module + '_hypocentral_distances') as distances:
            self.assertTrue(np.isfinite(misfit(data, greens, source)).all())
            attrs = misfit.collect_attributes(data, greens)

        self.assertEqual(len(angles.call_args_list), 2)
        for call in angles.call_args_list:
            self.assertEqual(call.kwargs['phase_list'], ['p', 'P'])
        medium.assert_not_called()
        shear.assert_not_called()
        distances.assert_not_called()
        self.assertEqual(attrs[0].station, greens[0].station.station)
        self.assertNotIn('takeoff_angle_S', attrs[0])


    def test_p_amplitude_does_not_request_s_geometry(self):
        greens, _ = _greens(n=3)
        source = MomentTensor([1., -1., 0., 0., 0., 0.5])
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pol=0., lambda_sprat=0.)
        data = (np.zeros(3), np.ones(3), np.zeros(3))
        module = 'mtuq.misfit.polarity_amplitudes.'
        with patch(module + '_takeoff_angles_taup',
                return_value=np.full(3, 45.)) as angles:
            self.assertTrue(np.isfinite(misfit(data, greens, source)).all())
        self.assertEqual(angles.call_count, 1)
        self.assertEqual(angles.call_args.kwargs['phase_list'], ['p', 'P'])


    def test_ratio_only_needs_no_spreading_distance(self):
        greens, _ = _greens(n=3)
        source = MomentTensor([1., -1., 0., 0., 0., 0.5])
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pol=0., lambda_pamp=0.)
        with patch('mtuq.misfit.polarity_amplitudes.'
                '_hypocentral_distances') as distances:
            values = misfit((np.zeros(3), np.zeros(3), np.full(3, 2.)),
                greens, source)
        self.assertTrue(np.isfinite(values).all())
        distances.assert_not_called()


    def test_public_predictions_include_disabled_terms(self):
        greens, _ = _greens(n=3)
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pamp=0., lambda_sprat=0.)
        predicted = misfit.get_predicted(greens,
            MomentTensor([1., -1., 0., 0., 0., 0.5]))
        for array in predicted:
            self.assertEqual(array.shape, (1, 3))
            self.assertTrue(np.isfinite(array).all())


    def test_floor_uses_frobenius_norm(self):
        greens, _ = _greens(n=1)
        misfit = PolarityPampSPampRatioMisfit(
            rho=2300., vp=5400., vs=3300., radiation_floor=0.1)
        # Pure off-diagonal tensor: ||M||_F = sqrt(2), ||m||_2 = 1.
        # P is nodal along the vertical; |S| = 1/sqrt(2) at these S angles.
        source = MomentTensor([0., 0., 0., 0., 0., 1.])
        greens[0].azimuth = 0.
        with patch('mtuq.misfit.polarity_amplitudes.'
                '_takeoff_angles_taup', side_effect=[np.array([0.]),
                    np.array([45.])]):
            _, _, ratios = misfit.get_predicted(greens, source)
        expected = np.log10(5.) + 3.*np.log10(5400./3300.)
        np.testing.assert_allclose(ratios, [[expected]], atol=EPSVAL)


    def test_source_scaling_and_zero_s_radiation(self):
        greens, _ = _greens(n=3)
        misfit = PolarityPampSPampRatioMisfit()
        for mt in ([1., -2., 0.5, 0.3, -0.4, 0.7],
                [1., 1., 1., 0., 0., 0.]):
            source = np.array(mt)*1.e15
            a = misfit.get_predicted(greens, MomentTensor(source))
            b = misfit.get_predicted(greens, MomentTensor(10.*source))
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_allclose(10.*a[1], b[1], rtol=1.e-12, atol=0.)
            np.testing.assert_allclose(a[2], b[2], rtol=0., atol=EPSVAL)
            self.assertTrue(np.isfinite(a[2]).all())


    def test_zero_source_has_no_defined_s_p_ratio(self):
        greens, _ = _greens(n=1)
        source = MomentTensor(np.zeros(6))
        with self.assertRaisesRegex(ValueError, 'nonzero moment tensors'):
            PolarityPampSPampRatioMisfit().get_predicted(greens, source)
        # Its zero P prediction remains usable when the ratio is disabled.
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pamp=0., lambda_sprat=0.)
        self.assertTrue(np.isfinite(misfit(
            (np.ones(1), np.zeros(1), np.zeros(1)), greens, source)).all())


class TestInputHandling(unittest.TestCase):

    def test_dataset_is_rejected_clearly(self):
        # a Dataset subclasses list, so it must not be mistaken for a
        # bundle of observation arrays
        import mtuq
        dataset = mtuq.Dataset([obspy.Stream() for _ in range(6)])

        misfit = PolarityPampSPampRatioMisfit()

        with self.assertRaises(NotImplementedError):
            misfit.get_observed(dataset)


    def test_three_element_bundle_gets_unit_weights(self):
        misfit = PolarityPampSPampRatioMisfit()

        observed = misfit.get_observed(
            (np.array([1., -1.]), np.array([1., 2.]), np.array([1., 1.])))

        self.assertEqual(len(observed), 6)
        for weights in observed[3:]:
            self.assertTrue(np.array_equal(weights, np.ones(2)))


    def test_from_dict_prefers_network_prefixed_key(self):
        greens, _ = _greens(n=2)
        stations = [tensor.station for tensor in greens]

        polarities = polarities_pamp_spamp_ratio_from_dict(
            {'XX.S0': 1., 'S0': -1., 'S1': 1.}, None, None, stations)[0]

        self.assertEqual(polarities[0], 1.)


    def test_bad_parameters_rejected(self):
        for parameters in (
                {'method': 'FK_metadata'},
                {'norm': 'L3'},
                {'rho': 0.},
                {'vp': -1.},
                {'vs': np.nan}):

            with self.assertRaises((TypeError, ValueError)):
                PolarityPampSPampRatioMisfit(**parameters)


    def test_invalid_term_weights_and_floor(self):
        for name in ('lambda_pol', 'lambda_pamp', 'lambda_sprat'):
            for value in (-1., np.nan, np.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        PolarityPampSPampRatioMisfit(**{name: value})

        for value in (0., -1., np.nan, np.inf):
            with self.subTest(radiation_floor=value):
                with self.assertRaises(ValueError):
                    PolarityPampSPampRatioMisfit(radiation_floor=value)


    def test_multidimensional_observations_and_weights_rejected(self):
        misfit = PolarityPampSPampRatioMisfit()
        for index in range(6):
            observed = [np.ones(2) for _ in range(6)]
            observed[index] = np.ones((2, 1))
            with self.subTest(array=index):
                with self.assertRaisesRegex(ValueError, 'one-dimensional'):
                    misfit.get_observed(observed)


    def test_log_zero_and_missing_ratios_remain_distinct(self):
        greens, _ = _greens(n=3)
        stations = [tensor.station for tensor in greens]
        with self.assertWarns(UserWarning):
            observed = polarities_pamp_spamp_ratio_from_dict(
                None, None, {'S0': 0., 'S1': -1.}, stations,
                spamp_ratio_is_log10=True)

        np.testing.assert_allclose(observed[2], [1., 0.1, 0.])
        missing = polarities_pamp_spamp_ratio_from_dict(
            None, None, None, stations, spamp_ratio_is_log10=True)
        np.testing.assert_array_equal(missing[2], np.zeros(3))


    def test_linear_and_log_ratio_inputs_give_identical_misfits(self):
        greens, _ = _greens(n=3)
        stations = [tensor.station for tensor in greens]
        source = MomentTensor([1., -2., 0.5, 0.3, -0.4, 0.7])
        misfit = PolarityPampSPampRatioMisfit(
            lambda_pol=0., lambda_pamp=0.)
        linear = polarities_pamp_spamp_ratio_from_dict(
            None, None, {'S0': 1., 'S1': 10., 'S2': 0.1}, stations)
        logarithmic = polarities_pamp_spamp_ratio_from_dict(
            None, None, {'S0': 0., 'S1': 1., 'S2': -1.}, stations,
            spamp_ratio_is_log10=True)
        np.testing.assert_allclose(
            misfit(linear, greens, source), misfit(logarithmic, greens, source))


    def test_medium_defaults_to_velocity_model(self):
        # rho, vp and vs left unset are read from taup_model at the source
        # depth, so the medium and the takeoff angles share one model
        greens, origin = _greens(n=4, depth_in_m=33000.)
        misfit = PolarityPampSPampRatioMisfit()

        self.assertIsNone(misfit.rho)
        self.assertIsNone(misfit.vp)
        self.assertIsNone(misfit.vs)

        rho, vp, vs = misfit._medium(greens)

        expected = _medium_from_taup(TauPyModel('ak135'), 33000.)
        self.assertAlmostEqual(rho, expected[0])
        self.assertAlmostEqual(vp, expected[1])
        self.assertAlmostEqual(vs, expected[2])

        # SI units, and ak135 is faster and denser at 33 km than the
        # shallow values appropriate to induced microseismicity
        self.assertTrue(2000. < rho < 4000.)
        self.assertTrue(5000. < vp < 9000.)
        self.assertTrue(vp > vs)


    def test_medium_tracks_source_depth(self):
        # a depth search must not hold the medium fixed
        shallow, _ = _greens(n=4, depth_in_m=5000.)
        deep, _ = _greens(n=4, depth_in_m=100000.)
        misfit = PolarityPampSPampRatioMisfit()

        self.assertNotAlmostEqual(
            misfit._medium(shallow)[1], misfit._medium(deep)[1])


    def test_explicit_medium_overrides_model(self):
        # reproducing a published inversion requires pinning the medium
        greens, origin = _greens(n=4, depth_in_m=33000.)
        misfit = PolarityPampSPampRatioMisfit(
            rho=2300., vp=5400., vs=3300.)

        self.assertEqual(misfit._medium(greens), (2300., 5400., 3300.))


    def test_partial_medium_override(self):
        # each parameter is independent; the rest come from the model
        greens, origin = _greens(n=4, depth_in_m=33000.)
        misfit = PolarityPampSPampRatioMisfit(vp=5400.)

        rho, vp, vs = misfit._medium(greens)
        derived = _medium_from_taup(TauPyModel('ak135'), 33000.)

        self.assertEqual(vp, 5400.)
        self.assertAlmostEqual(rho, derived[0])
        self.assertAlmostEqual(vs, derived[2])


    def test_removed_parameters_rejected(self):
        # distance_mode and sp_obs_is_log10 were removed; silently
        # swallowing them would change results without warning
        for parameters in (
                {'distance_mode': 'ray_path'},
                {'sp_obs_is_log10': True}):

            with self.assertRaises(TypeError):
                PolarityPampSPampRatioMisfit(**parameters)



if __name__ == '__main__':
    unittest.main()
