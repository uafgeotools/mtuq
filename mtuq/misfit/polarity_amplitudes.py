
import numpy as np

import mtuq
from mtuq.util import AttribDict, Null, warn
from mtuq.misfit.polarity import (
    _get_azimuths, _radiation_coef_P, _takeoff_angles_taup)
from mtuq.misfit.waveform.level2 import _to_array, _type
from obspy.taup import TauPyModel


class PolarityPampSPampRatioMisfit(object):
    """ Polarity, P-amplitude and S/P-amplitude-ratio misfit function

    Evaluates the consistency of three body-wave observables with a trial
    moment tensor following Yang and Wang (2025, *Seismol. Res. Lett.*,
    96(2A), 1150-1162).


    .. rubric:: Observables

    Three quantities are compared. Each term can be enabled or disabled
    through its corresponding weight.

    1. **P-wave first-motion polarity**

       ``+1`` for positive/upward first motion, ``-1`` for negative/downward
       first motion, and ``0`` for unpicked or indeterminate observations.

    2. **P-wave amplitude**

       A signed scalar amplitude for each station. The amplitude magnitude
       is obtained from the maximum amplitude envelope within a short window
       beginning at the theoretical direct-P arrival. The sign of the vertical
       component at the sample corresponding to the envelope maximum determines
       the polarity and the sign of the P amplitude.

    3. **S/P amplitude ratio**

       A positive, linear S/P amplitude ratio derived from three-component,
       noise-corrected energies following Wang et al. (2020) and Yang and
       Wang (2025):

       .. code::

           rat = sqrt(
               [(S_R^2 + S_T^2 + S_Z^2)
                - (N_R^2 + N_T^2 + N_Z^2)]
               /
               [(P_R^2 + P_T^2 + P_Z^2)
                - (N_R^2 + N_T^2 + N_Z^2)]
           )

       where ``P``, ``S``, and ``N`` denote measurements from the P-wave,
       S-wave, and noise windows on the radial, transverse, and vertical
       components.

       The P and S windows begin at their corresponding theoretical direct
       arrivals, with window lengths based on a fraction of the S-P time.
       The noise window is chosen with the same duration before the event.

       This MTUQ implementation expects the observed S/P ratios as positive
       linear ratios and converts them internally to ``log10(S/P)`` when
       evaluating the misfit. Measurements recorded as logarithms are
       converted by passing ``spamp_ratio_is_log10=True`` to
       ``polarities_pamp_spamp_ratio_from_dict``. 

       Measurement of these observables is a preprocessing step outside this
       misfit class. They can be supplied using
       ``polarities_pamp_spamp_ratio_from_dict`` or equivalent arrays.


    .. rubric:: Predictions

    The predicted amplitudes are based on far-field body-wave radiation in a
    homogeneous isotropic whole-space approximation.

    The medium is one density and one pair of wave speeds for the whole
    propagation. By default these are read from the velocity model at the
    source depth, so that the medium and the takeoff angles come from the
    same model; see ``rho``, ``vp`` and ``vs`` below.

    Let ``R_P``, ``R_SV``, and ``R_SH`` denote the P, SV, and SH radiation
    coefficients and ``r`` the hypocentral distance. The predicted quantities
    are evaluated as

    .. code::

        amp_theo = R_P / (4 * pi * rho * vp**3 * r)

        log10_rat_theo = (
            log10(sqrt(R_SV**2 + R_SH**2) / abs(R_P))
            + 3 * log10(vp / vs)
        )

    ``R_P`` is evaluated using the direct-P takeoff angle. In this
    implementation, ``R_SV`` and ``R_SH`` are evaluated using the direct-S
    takeoff angle.

    With moments in N m, density in kg/m^3, velocities in m/s and distance
    in m, ``amp_theo`` has units of m s. Observed amplitudes must be calibrated
    to this coefficient; source-time and waveform-processing factors are not
    applied by this class.

    .. rubric:: Objective function

    Following NaNDC (Yang and Wang, 2025) equations 3-6, the default objective
    is the weighted sum of three normalized L1 residual terms:

    .. code::

        Phi = (
            lambda_pol   * Phi_pol
            + lambda_pamp  * Phi_pamp
            + lambda_sprat * Phi_sprat
        )

        Phi_pol = (
            sum(0.5 * tau * abs(pol_obs - pol_theo))
            / sum(tau * abs(pol_obs))
        )

        Phi_pamp = (
            sum(tau * abs(amp_obs - amp_theo))
            / sum(tau * abs(amp_obs))
        )

        Phi_sprat = (
            sum(tau * abs(log10(rat_obs) - log10_rat_theo))
            / sum(tau * abs(log10(rat_obs)))
        )

    ``tau`` represents station weights. Only finite, positive weights
    contribute; zero polarity and P amplitude mark unused observations.

    Each component is normalized by the corresponding observed-data scale,
    making the three terms dimensionless before weighting. If all usable S/P
    ratios equal one, normalization is undefined and ``ValueError`` is raised.

    When combining this objective with other MTUQ misfit functions whose
    numerical scales differ substantially, appropriate relative weighting or
    rescaling should be considered.

    .. rubric:: Usage

    The misfit returns one scalar value for each trial source and can therefore
    be used with the standard MTUQ ``grid_search`` workflow:

    .. code::

        misfit = PolarityPampSPampRatioMisfit(
            taup_model='ak135'
        )

        observations = polarities_pamp_spamp_ratio_from_dict(
            polarities_dict,
            pamp_dict,
            spamp_dict,
            stations,
        )

        results = grid_search(
            observations,
            greens,
            misfit,
            origin,
            sources,
        )

    """

    def __init__(self,
        method='taup',
        taup_model='ak135',
        lambda_pol=1.,
        lambda_pamp=1.,
        lambda_sprat=1.,
        rho=None,
        vp=None,
        vs=None,
        norm='L1',
        radiation_floor=1.e-6):

        """
        ``method`` (`str`):
        Source of takeoff angles.  Only `'taup'` is currently implemented.

        ``taup_model`` (`str`):
        Name of the TauP model used for takeoff angles

        ``lambda_pol``, ``lambda_pamp``, ``lambda_sprat`` (`float`):
        Weights on the polarity, P-amplitude and S/P-ratio terms.  A weight
        of zero disables the corresponding term entirely.

        ``rho``, ``vp``, ``vs`` (`float`):
        Density (kg/m^3) and P- and S-wave speeds (m/s) of the homogeneous
        source region.  If left as `None`, each is read from `taup_model`
        at the source depth, so that the medium and the takeoff angles come
        from the same velocity model.  Any of the three can be given
        explicitly to override the model, which is what reproducing a
        published inversion generally requires.

        ``norm`` (`str`):
        `'L1'` (default) and `'L2'` are accepted.

        ``radiation_floor`` (`float`):
        Positive, finite floor applied to both P and S radiation magnitudes
        inside the logarithms only. Expressed as a fraction of the Frobenius
        norm of the moment tensor, so nodal sources give finite residuals
        without coupling the ratio to moment magnitude.

        """
        if method != 'taup':
            raise TypeError("Bad parameter: method. Only 'taup' is "
                "currently implemented")

        if norm not in ('L1', 'L2'):
            raise ValueError("Bad parameter: norm. Expected 'L1' or 'L2'")

        for name, value in (('rho', rho), ('vp', vp), ('vs', vs)):
            if value is None:
                continue
            if not np.isfinite(value) or value <= 0.:
                raise ValueError('Bad parameter: %s must be a positive '
                    'number in SI units, or None to take the value from '
                    'the velocity model at the source depth' % name)

        self.method = method
        self.taup_model = taup_model
        self.lambda_pol = float(lambda_pol)
        self.lambda_pamp = float(lambda_pamp)
        self.lambda_sprat = float(lambda_sprat)
        self.rho = None if rho is None else float(rho)
        self.vp = None if vp is None else float(vp)
        self.vs = None if vs is None else float(vs)
        self.norm = norm
        self.radiation_floor = float(radiation_floor)

        for name, value in (('lambda_pol', self.lambda_pol),
                ('lambda_pamp', self.lambda_pamp),
                ('lambda_sprat', self.lambda_sprat)):
            if not np.isfinite(value) or value < 0.:
                raise ValueError('Bad parameter: %s must be finite and '
                    'nonnegative' % name)

        if not np.isfinite(self.radiation_floor) or self.radiation_floor <= 0.:
            raise ValueError('Bad parameter: radiation_floor must be finite '
                'and positive')

        self._taup = TauPyModel(self.taup_model)


    def __call__(self, data, greens, sources, progress_handle=Null(),
            set_attributes=False):

        pol_obs, amp_obs, rat_obs, tau_pol, tau_amp, tau_rat = \
            self.get_observed(data)

        if len(pol_obs) != len(greens):
            raise ValueError('Inconsistent dimensions: %d observations but '
                '%d Green\'s tensors. Observation arrays are ordered by '
                'station, so they must correspond one-to-one with the '
                'Green\'s tensors passed to the misfit function.'
                % (len(pol_obs), len(greens)))

        pol_theo, amp_theo, log_rat_theo = self._get_predicted(
            greens, sources, pamp=self.lambda_pamp != 0.,
            sprat=self.lambda_sprat != 0.)

        n_sources = pol_theo.shape[0]
        values = np.zeros(n_sources)

        if self.lambda_pol != 0.:
            values += self.lambda_pol*_misfit_term(
                pol_obs, pol_theo, tau_pol, self.norm, 'polarity', scale=0.5)

        if self.lambda_pamp != 0.:
            values += self.lambda_pamp*_misfit_term(
                amp_obs, amp_theo, tau_amp, self.norm, 'P amplitude')

        if self.lambda_sprat != 0.:
            mask = np.isfinite(rat_obs) & (tau_rat > 0.) & (rat_obs > 0.)
            log_rat_obs = np.full(len(rat_obs), np.nan)
            log_rat_obs[mask] = np.log10(rat_obs[mask])

            values += self.lambda_sprat*_misfit_term(
                log_rat_obs, log_rat_theo, tau_rat, self.norm,
                'S/P amplitude ratio', allow_zero=True)

        # returns a NumPy array of shape (len(sources), 1)
        return values.reshape(n_sources, 1)


    def get_observed(self, data):
        """ Unpacks observed polarities, P amplitudes and S/P ratios

        Accepts either a six-element bundle
        `(pol, amp, rat, tau_pol, tau_amp, tau_rat)` as returned by
        `polarities_pamp_spamp_ratio_from_dict`, or a three-element bundle
        `(pol, amp, rat)`, in which case unit weights are assumed.
        """

        if isinstance(data, mtuq.Dataset):
            raise NotImplementedError('Measuring P amplitudes and S/P ratios '
                'from a Dataset is not implemented. These observables require '
                'windowed measurements that MTUQ does not perform; supply '
                'them with polarities_pamp_spamp_ratio_from_dict instead.')

        if not isinstance(data, (tuple, list)) or len(data) not in (3, 6):
            raise TypeError('Expected a three- or six-element bundle of '
                'observation arrays')

        arrays = [np.atleast_1d(np.asarray(item, dtype=np.float64))
            for item in data]

        if any(array.ndim != 1 for array in arrays):
            raise ValueError('Observation and weight arrays must be '
                'one-dimensional')

        lengths = set(len(item) for item in arrays)
        if len(lengths) != 1:
            raise ValueError('Inconsistent dimensions: observation arrays '
                'must all have the same length')

        if len(arrays) == 3:
            unit = np.ones(len(arrays[0]))
            arrays += [unit.copy(), unit.copy(), unit.copy()]

        return tuple(arrays)


    def _medium(self, greens):
        """ Returns the (rho, vp, vs) used in the amplitude calculations

        Values supplied to the constructor take precedence.  Any left
        unspecified are read from the velocity model at the source depth,
        so that the medium is consistent with the takeoff angles.
        """

        rho, vp, vs = self.rho, self.vp, self.vs

        if None not in (rho, vp, vs):
            return rho, vp, vs

        depth_in_m = _origin_depth(greens)
        derived = _medium_from_taup(self._taup, depth_in_m)

        return (
            derived[0] if rho is None else rho,
            derived[1] if vp is None else vp,
            derived[2] if vs is None else vs)


    def get_predicted(self, greens, sources):
        """ Returns polarities, P amplitudes and log10(S/P) predictions

        All three predictions are returned, independently of the term weights.
        """
        return self._get_predicted(greens, sources)


    def _get_predicted(self, greens, sources, pamp=True, sprat=True):
        """ Evaluates only the amplitude predictions needed by the objective
        """

        if type(sources) == mtuq.MomentTensor:
            mt_array = sources.as_vector().reshape((1, 6))

        elif type(sources) == mtuq.Force:
            raise NotImplementedError

        elif _type(sources.dims) == 'MomentTensor':
            mt_array = _to_array(sources)

        elif _type(sources.dims) == 'Force':
            raise NotImplementedError

        else:
            raise TypeError

        if not np.all(np.isfinite(mt_array)):
            raise ValueError('Moment tensors must contain finite components')

        # geometry depends only on the origin and stations, so it is
        # evaluated once here rather than once per trial source
        azimuths = _get_azimuths(greens)
        takeoff_p = _takeoff_angles_taup(
            self._taup, greens, phase_list=['p', 'P'])
        radiation_p = _radiation_coef_P(mt_array, takeoff_p, azimuths)
        polarities = np.sign(radiation_p)
        amplitudes = log_ratios = None

        if pamp or sprat:
            rho, vp, vs = self._medium(greens)

        if pamp:
            distances = _hypocentral_distances(greens)
            amplitudes = radiation_p/(
                4.*np.pi*rho*vp**3*distances[None, :])

        if sprat:
            takeoff_s = _takeoff_angles_taup(
                self._taup, greens, phase_list=['s', 'S'])
            radiation_s = _radiation_coef_S(mt_array, takeoff_s, azimuths)

            # Six independent entries store the off-diagonal values once;
            # the Frobenius norm of the symmetric matrix counts them twice.
            source_norm = np.sqrt(
                np.sum(mt_array[:, :3]**2, axis=1) +
                2.*np.sum(mt_array[:, 3:]**2, axis=1))
            if np.any(~np.isfinite(source_norm) | (source_norm == 0.)):
                raise ValueError('S/P ratios require nonzero moment tensors '
                    'with finite Frobenius norms')

            # Floor both radiation magnitudes only inside the logarithms.
            # Scaling the floor with the source preserves magnitude invariance.
            floor = self.radiation_floor*source_norm[:, None]
            log_ratios = (
                np.log10(np.maximum(radiation_s, floor)) -
                np.log10(np.maximum(np.abs(radiation_p), floor)) +
                3.*np.log10(vp/vs))

        return polarities, amplitudes, log_ratios


    def collect_attributes(self, data, greens):
        """ Collects station attributes (used for beachball plots)
        """

        pol_obs, amp_obs, rat_obs = self.get_observed(data)[:3]

        takeoff_p = _takeoff_angles_taup(
            self._taup, greens, phase_list=['p', 'P'])
        if self.lambda_sprat != 0.:
            takeoff_s = _takeoff_angles_taup(
                self._taup, greens, phase_list=['s', 'S'])

        attrs_list = []
        for _i, greens_tensor in enumerate(greens):
            attrs = AttribDict(greens_tensor.station)
            attrs.update({
                'azimuth': greens_tensor.azimuth,
                'distance_in_m': greens_tensor.distance_in_m,
                'takeoff_angle': takeoff_p[_i],
                'polarity': pol_obs[_i],
                'p_amplitude': amp_obs[_i],
                'sp_ratio': rat_obs[_i],
                })
            if self.lambda_sprat != 0.:
                attrs.takeoff_angle_S = takeoff_s[_i]
            attrs_list.append(attrs)

        return attrs_list


    def description(self):
        _description = '\n'.join([
            f'    Misfit function type:\n    {type(self).__name__}\n',
            f'    Misfit function method:\n    {self.method}\n',
            f'    Weights (polarity, P amplitude, S/P ratio):\n'
            f'    {self.lambda_pol}, {self.lambda_pamp}, '
            f'{self.lambda_sprat}\n',
            ])
        return _description



#
# misfit evaluation
#

def _misfit_term(observed, predicted, weights, norm, label, scale=1.,
        allow_zero=False):

    valid = (np.isfinite(observed) & np.isfinite(weights) & (weights > 0.))
    if not allow_zero:
        valid &= (observed != 0.)

    if not valid.any():
        raise ValueError('No usable %s observations. Either supply data for '
            'at least one station or disable this term by setting its weight '
            'to zero.' % label)

    obs = observed[valid]
    syn = predicted[:, valid]
    tau = weights[valid]

    residual = scale*(syn - obs[None, :])

    if norm == 'L1':
        denominator = np.sum(tau*np.abs(obs))
        numerator = np.sum(tau[None, :]*np.abs(residual), axis=1)
    else:
        denominator = np.sum(tau*obs**2)
        numerator = np.sum(tau[None, :]*residual**2, axis=1)

    if denominator == 0.:
        raise ValueError('Degenerate %s observations: the normalizing sum is '
            'zero, so the term cannot be evaluated. This happens when every '
            'observation is zero.' % label)

    return numerator/denominator



#
# radiation patterns
#

def _radiation_coef_S(mt_array, takeoff, azimuth):
    """ Calculates the magnitude of far-field S radiation in USE coordinates

    Returns an array of shape `(len(mt_array), len(takeoff))`.  Angles are
    the direct-S takeoff angle measured from the downward vertical and the
    source-to-receiver azimuth, both in degrees.

    For a unit ray n, |n x (M n)| = sqrt(R_SV**2 + R_SH**2).  This avoids
    constructing a second coordinate convention or separate shear bases.
    """

    radiation = np.empty((len(mt_array), len(takeoff)))
    angles = zip(np.deg2rad(takeoff), np.deg2rad(azimuth))
    for _i, (theta, phi) in enumerate(angles):
        n = np.array([-np.cos(theta), -np.sin(theta)*np.cos(phi),
            np.sin(theta)*np.sin(phi)])
        mn = np.column_stack((
            mt_array[:, 0]*n[0] + mt_array[:, 3]*n[1] + mt_array[:, 4]*n[2],
            mt_array[:, 3]*n[0] + mt_array[:, 1]*n[1] + mt_array[:, 5]*n[2],
            mt_array[:, 4]*n[0] + mt_array[:, 5]*n[1] + mt_array[:, 2]*n[2]))
        radiation[:, _i] = np.linalg.norm(np.cross(n, mn), axis=1)

    return radiation


def _origin_depth(greens):
    """ Returns the source depth in m, checking that it is well defined
    """

    depths = set()
    for greens_tensor in greens:
        depths.add(float(greens_tensor.origin.depth_in_m))

    if len(depths) != 1:
        raise ValueError('Inconsistent dimensions: Green\'s tensors carry '
            'more than one origin depth')

    return depths.pop()


def _medium_from_taup(taup, depth_in_m):
    """ Reads density and wave speeds from the velocity model

    Returns `(rho, vp, vs)` in SI units, evaluated at `depth_in_m`.  The
    values are constant with respect to the receivers, so the whole-space
    approximation is preserved; only the choice of the constant is tied to
    the velocity model rather than left to the caller.
    """

    velocity_model = taup.model.s_mod.v_mod
    depth_in_km = depth_in_m/1000.

    try:
        rho = float(velocity_model.evaluate_below(depth_in_km, 'd')[0])
        vp = float(velocity_model.evaluate_below(depth_in_km, 'p')[0])
        vs = float(velocity_model.evaluate_below(depth_in_km, 's')[0])
    except Exception:
        raise ValueError('Could not read density and wave speeds from the '
            'velocity model at %.3f km depth. Supply rho, vp and vs '
            'explicitly instead.' % depth_in_km)

    if not all(np.isfinite(value) and value > 0. for value in (rho, vp, vs)):
        raise ValueError('The velocity model gives a nonpositive density or '
            'wave speed at %.3f km depth. Supply rho, vp and vs explicitly '
            'instead.' % depth_in_km)

    # velocity models are tabulated in km/s and g/cm^3
    return 1000.*rho, 1000.*vp, 1000.*vs


def _hypocentral_distances(greens):
    """ Calculates straight-line source-receiver distances

    The homogeneous whole-space approximation makes the geometrical
    spreading distance the straight line between hypocenter and receiver,
    not a ray-path length.  Receivers are assumed to lie at the surface.
    """

    distances = np.array([np.hypot(tensor.distance_in_m,
        tensor.origin.depth_in_m) for tensor in greens])

    if np.any(~np.isfinite(distances) | (distances <= 0.)):
        raise ValueError('Source-receiver distances must be finite and '
            'positive for the far-field approximation')

    return distances



#
# input preparation
#

def polarities_pamp_spamp_ratio_from_dict(dict_polarity, dict_pamp,
    dict_spamp_ratio, stations, dict_tau_polarity=None, dict_tau_pamp=None,
    dict_tau_spamp_ratio=None, spamp_ratio_is_log10=False):

    """ Converts dictionaries of observations to arrays ordered by station

    Each dictionary is keyed by station code, optionally prefixed by network
    (`'AK.BMR'` is tried before `'BMR'`).  Stations absent from a dictionary
    are set to zero, which marks them unused for that observable.

    ``spamp_ratio_is_log10`` (`bool`):
    Declares the units of `dict_spamp_ratio`.  `False` (the default) means
    linear ratios; `True` means their base-10 logarithms. A logarithmic zero
    is valid (S/P=1); omit the entry or use NaN for missing logarithmic data.

    Returns a six-element bundle suitable for passing to
    `PolarityPampSPampRatioMisfit` as its `data` argument.
    """

    n = len(stations)
    pol = np.zeros(n)
    amp = np.zeros(n)
    rat = np.zeros(n)
    tau_pol = np.ones(n)
    tau_amp = np.ones(n)
    tau_rat = np.ones(n)

    for _i, station in enumerate(stations):
        pol[_i] = _lookup(dict_polarity, station, 'polarities')
        amp[_i] = _lookup(dict_pamp, station, 'P amplitudes')
        rat[_i] = _lookup(dict_spamp_ratio, station, 'S/P ratios',
            default=np.nan)

        if dict_tau_polarity is not None:
            tau_pol[_i] = _lookup(
                dict_tau_polarity, station, 'polarity weights')
        if dict_tau_pamp is not None:
            tau_amp[_i] = _lookup(
                dict_tau_pamp, station, 'P amplitude weights')
        if dict_tau_spamp_ratio is not None:
            tau_rat[_i] = _lookup(
                dict_tau_spamp_ratio, station, 'S/P ratio weights')

    if spamp_ratio_is_log10:
        # Missing entries are NaN until conversion: a measured log10 ratio
        # of zero represents a valid linear ratio of one.
        mask = np.isfinite(rat)
        rat[mask] = 10.**rat[mask]

    elif np.any(rat < 0.):
        raise ValueError('Negative S/P amplitude ratio. Linear ratios are '
            'positive by construction, so a negative value usually means '
            'the measurements are logarithms. Pass '
            'spamp_ratio_is_log10=True if so.')

    rat[np.isnan(rat)] = 0.
    return pol, amp, rat, tau_pol, tau_amp, tau_rat


def _lookup(dictionary, station, label, default=0.):
    """ Retrieves a station value, trying 'NET.STA' before 'STA'
    """

    if dictionary is None:
        return default

    for key in ('%s.%s' % (station.network, station.station), station.station):
        if key in dictionary:
            return float(dictionary[key])

    warn('Station %s not found in dictionary of %s'
        % (station.station, label))
    return default
