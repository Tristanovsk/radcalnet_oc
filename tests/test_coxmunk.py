"""Sunglint reflectance (Cox and Munk statistics): normalisation, slope variance and wind direction."""

import numpy as np
import pytest

from radcalnet_oc import Sunglint


def fresnel_unpolarised(theta, m=1.334):
    """Fresnel reflectance of a flat water surface for unpolarised light at incidence theta (rad)."""
    ct, st = np.cos(theta), np.sin(theta)
    root = np.sqrt(m ** 2 - st ** 2)
    rs = (ct - root) / (ct + root)
    rp = (m ** 2 * ct - root) / (m ** 2 * ct + root)
    return (rs ** 2 + rp ** 2) / 2


def glint(sza, vza, azi, ws, **kwargs):
    return Sunglint(sza, vza, azi).sunglint(ws, **kwargs)[0]


@pytest.mark.parametrize('sza', [20., 40.])
def test_hemispherical_albedo_equals_fresnel(sza):
    """Energy conservation of Eq. glint: (1/pi) int R mu_v dOmega_v equals the Fresnel reflectance at the
    solar angle for a weakly rough surface (no shadowing), which checks the slope probability normalisation
    and the factor pi / (4 mu0 muv cos^4 theta_n)."""
    ws = 2.
    vza = np.radians(np.arange(0.25, 89.9, 0.5))
    azi = np.radians(np.arange(0.5, 360., 1.))
    R = np.array([[glint(sza, np.degrees(v), np.degrees(a), ws, stats='cm_iso', shadow=False) for a in azi]
                  for v in vza])
    albedo = (R * (np.cos(vza) * np.sin(vza))[:, None]).sum() * np.radians(0.5) * np.radians(1.) / np.pi
    assert albedo == pytest.approx(fresnel_unpolarised(np.radians(sza)), rel=0.03)


def test_isotropic_variance_is_total_mean_square_slope():
    """Isotropic statistics: sigma2 = 0.003 + 5.12e-3 W is the total mean square slope (upwind + crosswind),
    i.e. sigma2 / 2 per axis. Averaged over the wind directions, the directional statistics (same total
    variance, 0.003 + 5.08e-3 W) give the same glint off the specular direction; with a variance of sigma2
    per axis (the alternative formulation) the isotropic glint would be very different."""
    ws, sza, azi = 5., 30., 180.
    for vza in [20., 45.]:
        iso = glint(sza, vza, azi, ws, stats='cm_iso', shadow=False)
        directional = np.mean([glint(sza, vza, azi, ws, wazi=w, stats='cm_dir', shadow=False)
                               for w in np.arange(0., 360., 5.)])
        assert iso == pytest.approx(directional, rel=0.15), vza


def test_wind_direction_counterclockwise():
    """The wind direction turns in the same sense as the viewing azimuth: the glint is larger when the wind
    is aligned with the slope of the reflecting facets (upwind variance > crosswind variance) than when it
    is perpendicular, or mirrored (a wrong sense of rotation would align the wind with the mirror slope)."""
    sza, vza, azi, ws = 30., 40., 120., 10.
    zx, zy, _, _ = Sunglint(sza, vza, azi).sunglint(ws, wazi=0., stats='cm_iso', slope=True)
    psi = np.degrees(np.arctan2(zy, zx))  # azimuth of the facet slope
    aligned = glint(sza, vza, azi, ws, wazi=psi, stats='cm_dir', shadow=False)
    perpendicular = glint(sza, vza, azi, ws, wazi=psi + 90., stats='cm_dir', shadow=False)
    mirrored = glint(sza, vza, azi, ws, wazi=-psi, stats='cm_dir', shadow=False)
    assert aligned > 1.5 * perpendicular
    assert aligned > 1.2 * mirrored


def test_mirror_symmetry():
    """Mirroring the geometry and the wind about the solar plane leaves the glint unchanged."""
    for stats in ['cm_dir', 'bh2006']:
        a = glint(30., 35., 130., 7., wazi=40., stats=stats)
        b = glint(30., 35., -130., 7., wazi=-40., stats=stats)
        assert a == pytest.approx(b, rel=1e-10), stats


def test_shadowing_follows_the_wind_direction():
    """The shadowing uses the slope variance in the azimuth of the Sun and of the sensor relative to the
    wind: turning the whole scene (Sun, sensor and wind) about the vertical does not change the result,
    i.e. a wind along the solar plane with the sensor at azimuth 90 deg is equivalent, for the shadowing,
    to a wind at 90 deg with the sensor in the solar plane."""
    s = Sunglint(30., 50., 90.)
    ws = 10.
    s_up2, s_cr2 = 3.16e-3 * ws, 0.003 + 1.92e-3 * ws
    # wind along the solar plane: Sun upwind, sensor crosswind
    np.testing.assert_allclose(s.shadowing(s_up2, s_cr2, wazi=0.),
                               1 / (1 + s.Lambda(s_up2, s_cr2, 1., s.sza) + s.Lambda(s_up2, s_cr2, 0., s.vza)))
    # wind at 90 deg: Sun crosswind, sensor upwind
    np.testing.assert_allclose(s.shadowing(s_up2, s_cr2, wazi=90.),
                               1 / (1 + s.Lambda(s_up2, s_cr2, 0., s.sza) + s.Lambda(s_up2, s_cr2, 1., s.vza)),
                               atol=1e-12)


def test_heavy_tailed_limits():
    """The heavy-tailed isotropic statistics reduce to cm_iso: Student t for nu -> inf, mixture for f_tail = 0."""
    for vza, azi in [(10., 160.), (30., 120.), (45., 60.)]:
        iso = glint(35., vza, azi, 5., stats='cm_iso')
        assert glint(35., vza, azi, 5., stats='cm_iso_t', nu=1e7) == pytest.approx(iso, rel=1e-4)
        assert glint(35., vza, azi, 5., stats='cm_iso_mix', f_tail=0.) == pytest.approx(iso, rel=1e-12)


@pytest.mark.parametrize('kwargs', [{'stats': 'cm_iso_t', 'nu': 3.},
                                    {'stats': 'cm_iso_mix', 'f_tail': 0.05, 'sigma2_tail': 0.1}])
def test_heavy_tailed_slope_pdf(kwargs):
    """Slope pdf of the heavy-tailed statistics: normalised, Student t with the Cox and Munk mean square slope,
    and a heavier tail than the Gaussian far from the specular direction."""
    ws = 3.
    sigma2 = 3e-3 + 512e-5 * ws
    # slope pdf p(tan theta_n) through slope=True for geometries spanning the facet tilt
    t = np.linspace(0., 3., 3001)
    # direct evaluation of the pdf as in Sunglint (isotropic: function of tan^2 theta_n only)
    if kwargs['stats'] == 'cm_iso_t':
        nu = kwargs['nu']
        s2 = sigma2 * (nu - 2) / (2 * nu)
        p = 1. / (2 * np.pi * s2) * (1 + t ** 2 / (nu * s2)) ** (-(nu + 2) / 2)
    else:
        f, s2w = kwargs['f_tail'], kwargs['sigma2_tail']
        p = (1 - f) / (np.pi * sigma2) * np.exp(-t ** 2 / sigma2) + f / (np.pi * s2w) * np.exp(-t ** 2 / s2w)
    norm = np.trapezoid(2 * np.pi * t * p, t)
    assert norm == pytest.approx(1., abs=0.02)
    if kwargs['stats'] == 'cm_iso_t':
        # mean square slope (converges slowly for nu = 3: integrate analytically beyond t = 3)
        msq = np.trapezoid(2 * np.pi * t ** 3 * p, t)
        assert msq == pytest.approx(sigma2, rel=0.25)
    # check against Sunglint: ratio to cm_iso at the specular point and far from it
    s = Sunglint(30., 30., 180.)  # specular geometry (theta_n = 0)
    peak = s.sunglint(ws, stats=kwargs['stats'], shadow=False, slope=True,
                      **{k: v for k, v in kwargs.items() if k != 'stats'})[3]
    assert peak == pytest.approx(p[0], rel=1e-6)
    far = dict(sza=50., vza=10., azi=90.)  # facet tilt 26 deg (glint angle about 50 deg)
    tail = glint(far['sza'], far['vza'], far['azi'], ws, shadow=False, **kwargs)
    iso = glint(far['sza'], far['vza'], far['azi'], ws, stats='cm_iso', shadow=False)
    assert tail > 100 * iso


@pytest.mark.parametrize('kwargs', [{'stats': 'cm_iso_t', 'nu': 4.}, {'stats': 'cm_iso_mix', 'f_tail': 0.05}])
def test_heavy_tailed_hemispherical_albedo(kwargs):
    """Energy conservation of the heavy-tailed statistics (no shadowing), as for cm_iso."""
    sza, ws = 30., 2.
    vza = np.radians(np.arange(0.25, 89.9, 0.5))
    azi = np.radians(np.arange(0.5, 360., 1.))
    R = np.array([[glint(sza, np.degrees(v), np.degrees(a), ws, shadow=False, **kwargs) for a in azi]
                  for v in vza])
    albedo = (R * (np.cos(vza) * np.sin(vza))[:, None]).sum() * np.radians(0.5) * np.radians(1.) / np.pi
    assert albedo == pytest.approx(fresnel_unpolarised(np.radians(sza)), rel=0.05)


def test_glint_angle():
    """Glint angle: 0 in the specular direction (vza = sza, azimuth 0 of the LUT), sza + vza in the
    backscattering plane (azimuth 180)."""
    from radcalnet_oc import glint_angle
    assert glint_angle(30., 30., 0.) == pytest.approx(0., abs=1e-6)
    assert glint_angle(30., 20., 180.) == pytest.approx(50.)
    assert glint_angle(40., 0., 77.) == pytest.approx(40.)


def test_glint_flags_bits():
    from radcalnet_oc import glint_flags, GLINT_FLAGS
    f = glint_flags([0.005, 0.02, 0.2, 0.7, 0.7], [45., 45., 45., 45., 20.])
    assert f.dtype == np.uint8
    np.testing.assert_array_equal(f, [0, 1, 3, 7, 7 | GLINT_FLAGS['glint_near_specular']])


@pytest.mark.parametrize('stats', ['cm_iso', 'cm_iso_t', 'cm_iso_mix'])
def test_core_wind_coefficient(stats):
    """sigma2_coef: default Cox and Munk; (s0, s1) gives the same glint as Cox and Munk at the effective wind
    (s0 + s1 W - 0.003) / 0.00512, shadowing included."""
    s = Sunglint(35., 20., 140.)
    ws = 4.
    default = s.sunglint(ws, stats=stats)[0]
    assert s.sunglint(ws, stats=stats, sigma2_coef=(3e-3, 512e-5))[0] == pytest.approx(default, rel=1e-12)
    s0, s1 = 0.003, 0.0035
    w_eff = (s0 + s1 * ws - 3e-3) / 512e-5
    assert s.sunglint(ws, stats=stats, sigma2_coef=(s0, s1))[0] == pytest.approx(s.sunglint(w_eff, stats=stats)[0],
                                                                                 rel=1e-12)
