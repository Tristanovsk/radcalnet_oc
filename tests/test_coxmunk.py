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
