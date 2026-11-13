**Coupling over 350-2500 nm.** The result of 684-990 nm holds over the whole range (678 absorbing channels in
19 windows, sets v2 + v3). The path reflectance sees a fraction of the geometric air mass: $\eta_{atm}$ = 0.41
(940 nm) to 0.87 (2150 nm) in median in the H$_2$O bands, 0.48 (O$_2$-A R branch) to 0.89 (CH$_4$ 2300 nm) for the
well-mixed gases, with an IQR of 0.2-0.3; $\eta_{atm}$ increases with wavelength beyond 1000 nm (less scattering
below the absorbers). Two exceptions: O$_3$ (stratospheric, above the scatterers, $\eta_{atm}$ = 1.00) and the
weak CO$_2$ 1600 nm band ($\eta_{atm}$ = 1.04), for which the decoupled model is already close to exact. The
water-leaving term and the irradiance see 1-5 % more than the direct beam ($\eta$ = 1.04-1.05 below 1000 nm,
1.01-1.03 beyond 1100 nm).

**Error of the decoupled model.** Channel level, path reflectance: 5-16 % of the signal in median (95th percentile
22-38 %) in the strong H$_2$O bands (720, 820, 900, 940, 1130, 1380, 1880, 2450 nm), 1.5-4.4 % in the O$_2$ and
CO$_2$ 2010 nm bands, below 1 % for O$_3$, CO$_2$ 1600, CH$_4$ 1660 and the 1000 nm H$_2$O wing; water-leaving
term and irradiance 0.1-0.6 % (95th percentile < 1.7 %). Band level (sensor bands of section 6), absorption-depth
error of the path reflectance: 45-48 % for OLCI Oa19 / Oa20 and MSI B9, 29-34 % for OCI 1038 and the cirrus bands
(MSI B10, OLI B9, OCI 1378), 21-29 % for Oa13, Oa15 and Oa21, 6-13 % in the SWIR (1240-2260 nm), 3-6 % for MSI B3
and B4 (O$_3$). Weighted by the absorption depth $D_b$, the band path reflectance is underestimated by 31-34 %
(Oa20, B9, cirrus bands), 15 % (Oa13, Oa19), 4.7 % (Oa15) and below 2 % elsewhere; the irradiance by 0.2-3 % of
the absorption, i.e. < 0.5 % of the signal.

**Photon path-length model.** The Gamma-distribution model (`ppld`, 8 coefficients per window and term, 57
models fitted in 5 min) reduces the channel error of the path reflectance to 0.4-3.2 % in median (95th percentile
1-9 %; rms of the fit 3.7-4.4 % in the saturated H$_2$O bands 940-2450 nm), and that of the water-leaving term and
of the irradiance to < 0.07 % (95th percentile < 0.25 %). Band level, decoupled → ppld, median [95th percentile],
% of the absorption: Oa13 20.9 → 2.2 [6.2], Oa15 21.7 → 4.8 [13.8], Oa19 47.8 → 7.9 [19.6], Oa20 44.8 → 6.2
[18.0], MSI B9 44.7 → 6.1 [17.9], cirrus bands 29-33 → 5.4-6.1 [19], OCI 1038 34.3 → 9.0, MSI B12 9.0 → 5.3,
OCI 2130 10.2 → 6.8, MSI B3 / B4 3.1 / 6.4 → 1.6 / 3.2; irradiance ≤ 3 → ≤ 0.5 %. The residual band TOA error of
the path reflectance is then 4.6-6.1 % in the 940 nm and cirrus bands, 1.0-2.6 % in Oa13 / Oa15 / Oa19, < 1 %
elsewhere. The gain is small (1-2 points) where the absorption is weak and spread over a window with a variable
spectral shape: Oa21 (28.8 → 11.5 %), VIIRS M8 (12.5 → 11.0 %), the 1600 nm bands (7 → 6 %), all with
$D_b$ < 5 %, i.e. < 0.6 % of the signal.

**Robustness.** Refitted without the water vapour column 4 g cm$^{-2}$ (resp. the solar zenith angle 70°) and
evaluated on it, the channel error of the path reflectance increases by at most 0.1 point (resp. 0.5 point,
H$_2$O 2450 nm) relative to the full fit on the same conditions: the model extrapolates in water vapour and
geometry. A single model per absorber class is **not** sufficient: it doubles the error in the H$_2$O bands of the
SWIR (1880, 2150, 2450 nm) and is worse than the decoupled model in the CO$_2$ 1600, CH$_4$ 1660 and H$_2$O 2150 nm
windows and for MSI B11 (14 vs 7 %), B12 (11.5 vs 9 %) and OCI 2130 (19 vs 10 %); one model per window is needed.

**Path reflectance in the O$_3$ Chappuis band.** For O$_3$ the fitted $\eta_{atm}$ (1.15) and $s^2$ (3.4) are at
the limit of the model: the channel median error is not improved (0.26 → 0.33 %, 95th percentile 1.6 → 1.1 %) and
the 10 nm bands of 470-520 nm are worse than decoupled (~20 vs ~10 % of an absorption of < 2 %), while the broad
MSI B3 / B4 bands are improved. The O$_3$ window should keep the decoupled model for the path reflectance.

**Absorption tables.** The coupling corrections are relative and transfer to the package only as far as the
absorption tables agree. The absorption depth of RadCalNet-OC (REPTRAN) agrees with the CKD of SOS-ABS within 10 %
in the O$_3$, O$_2$-A, H$_2$O 820 / 940 / 1000, O$_2$ 1270, CO$_2$, CH$_4$ 2300 and H$_2$O 2450 nm windows, but is
25-40 % larger in the H$_2$O 720, 900, 1880, 2150 nm and CH$_4$ 1660 nm windows, 70 % larger in O$_2$-B (REPTRAN
O$_2$; `o2_model='lbl'` agreed within 2 % in the 684-990 nm study) and **one third smaller in the H$_2$O 1380 nm
window** (cirrus bands). These differences exceed the coupling residual in several windows.

**Implementation.** The candidate table (`scripts/sos_abs_coupling/gas_coupling_ppld_full.csv`, 19 windows,
57 rows) has the format of `gas_coupling_ppld.csv`; it doubles the cost of the coupled transmittance (0.9 → 1.9 s
for three terms and 20 geometries). Running it in the package exposed a bug fixed in
`GaseousTransmittance.get_coupled_transmittance`: contiguous windows whose common bound is a channel of the LUT
(1280.0 nm) gave a duplicated wavelength. The package couples only H$_2$O and the background gases
(`COUPLED_GASES`), whereas the fit is made on the total transmittance: in the O$_3$-VIS, CH$_4$ 1660 and CH$_4$
2300 nm windows the coefficients would be applied to the wrong absorbers.

**Recommendations.**

* adopt one `ppld` model per window beyond 990 nm (H$_2$O 1000, 1130, 1380, 1880, 2150, 2450 nm; O$_2$ 1270;
  CO$_2$ 2010 nm), where the gain is large and robust;
* keep the decoupled model for O$_3$ (at least for the path reflectance) and for the CO$_2$ 1600 nm window
  ($\eta \approx 1$, error < 1 %);
* for the CH$_4$ windows, either add CH$_4$ to `COUPLED_GASES` or drop them from the table;
* before adoption, check the transfer end to end (band-level path reflectance of the package against the coupled
  simulations, as for 684-990 nm), first in the H$_2$O 1380 and 1880 nm windows where the tables differ by 30-40 %.

**Limits.** Same as the 684-990 nm study (maritime aerosols, US62 profiles, 1013 hPa, scalar radiative transfer,
Cox-Munk 5 m s$^{-1}$, H$_2$O 0.5-4 g cm$^{-2}$, solar angles 20-70°, AOT(550) 0.05-0.4); the gaps between the
684-990 nm windows still use the decoupled model; the 95th percentile of the band residual remains 18-20 % of the
absorption in the saturated H$_2$O bands (940, 1380 nm).
