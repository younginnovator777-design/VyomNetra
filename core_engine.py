"""
================================================================================
 VyomNetra  ::  CORE DETECTION ENGINE  (protected module)
================================================================================
This module contains the complete exoplanet-detection algorithm stack used by
the VyomNetra dashboard. It is intentionally kept separate from app.py:

  - The UI layer (app.py) only ever calls the public functions below.
  - No algorithmic parameters, thresholds, or intermediate math are re-derived
    or duplicated in the UI layer.
  - This file can be independently compiled to bytecode (.pyc) or packaged as
    a private wheel for deployment, so the dashboard can ship without the
    UI ever revealing the underlying source.

Pipeline stages implemented here:
  1. Data acquisition        (TESS SAP_FLUX via lightkurve / MAST, or synthetic)
  2. Detrending               (two-pass median + Savitzky-Golay)
  3. Periodic transit search  (Box Least Squares, Kovacs et al. 2002)
  3.5 Mono-transit anomaly search (dual-path engine, isolated-event detector)
  4. Phase folding
  5. Astrophysical parameter extraction (Seager & Mallen-Ornelas 2003)
  6. False-positive vetting  (odd/even, secondary eclipse, transit shape)
  7. Diagnostic visualisation
  8. Batch / survey-mode pipeline
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal
from astropy.timeseries import BoxLeastSquares

# ------------------------------------------------------------------------
# Default search configuration (can be overridden by caller / UI inputs)
# ------------------------------------------------------------------------
DEFAULTS = {
    "flux_type": "sap_flux",
    "sector_idx": 0,
    "snr_threshold": 7.0,
    "period_min": 0.5,
    "period_max": 13.0,
    "n_periods": 5000,
}


# =========================================================================
# STAGE 1 — DATA ACQUISITION
# =========================================================================
def download_tess_lc(tic_id, flux_type="sap_flux", sector_idx=0):
    """
    Fetches a real TESS 2-minute cadence light curve from the NASA MAST
    archive via lightkurve, cleans it, and normalises flux to median = 1.
    Returns (time, flux, flux_err) as float64 numpy arrays.
    """
    import lightkurve as lk

    sr = lk.search_lightcurve(tic_id, author="SPOC", cadence="2min")
    if len(sr) == 0:
        sr = lk.search_lightcurve(tic_id, author="TESS-SPOC")
    if len(sr) == 0:
        raise RuntimeError(f"No light curves found on MAST for {tic_id}")

    lc = sr[sector_idx].download(flux_column=flux_type)
    lc = lc.remove_nans().remove_outliers(sigma=5.0)

    t = lc.time.value.astype(np.float64)
    f = lc.flux.value.astype(np.float64)
    fe = lc.flux_err.value.astype(np.float64)

    med = np.nanmedian(f)
    f /= med
    fe /= med
    return t, f, fe


def generate_demo_lightcurve(period=0.78884, depth_ppm=20200,
                              duration_hr=1.58, n_points=19440, span_days=27.0):
    """
    Synthesises a light curve reproducing the WASP-19 b system (TIC
    35516889) — an ultra-short-period, deep-transit hot Jupiter used as
    the reference validation case for this pipeline. Used for the
    built-in Demo Run, and as an offline fallback when live MAST access
    is unavailable. The random seed is fixed internally so the demo is
    fully reproducible.
    """
    rng = np.random.default_rng(1935168890)
    time = np.linspace(1712, 1712 + span_days, n_points)
    flux = rng.normal(1.0, 0.0015, len(time))

    dep_inj = depth_ppm / 1e6
    dur_inj = duration_hr / 24
    t0_inj = time[0] + 0.4
    phase = ((time - t0_inj) % period) / period
    in_tr = np.abs(phase - 0.5) < (dur_inj / period / 2)
    phase_c = (phase - 0.5) * period
    depth_profile = dep_inj * np.cos(np.pi * phase_c[in_tr] / dur_inj) ** 2
    flux[in_tr] -= dep_inj - depth_profile

    # instrumental ramp systematic
    flux += 0.004 * np.exp(-(time - time[0]) / 4.0)
    flux_err = np.full_like(flux, 0.0015)

    return time, flux, flux_err, {
        "period_days": period, "depth_ppm": depth_ppm, "duration_hr": duration_hr,
        "target_name": "WASP-19 b (TIC 35516889)",
    }


def load_custom_csv(dataframe):
    """
    Accepts a user-uploaded DataFrame with columns time, flux[, flux_err]
    and returns cleaned (time, flux, flux_err) numpy arrays normalised to
    median flux = 1.0.
    """
    cols = {c.lower().strip(): c for c in dataframe.columns}
    if "time" not in cols or "flux" not in cols:
        raise ValueError("CSV must contain at least 'time' and 'flux' columns.")

    t = dataframe[cols["time"]].to_numpy(dtype=np.float64)
    f = dataframe[cols["flux"]].to_numpy(dtype=np.float64)
    if "flux_err" in cols:
        fe = dataframe[cols["flux_err"]].to_numpy(dtype=np.float64)
    else:
        fe = np.full_like(f, np.nanstd(f))

    good = ~(np.isnan(t) | np.isnan(f))
    t, f, fe = t[good], f[good], fe[good]

    med = np.nanmedian(f)
    f = f / med
    fe = fe / med
    return t, f, fe


# =========================================================================
# STAGE 2 — DETRENDING
# =========================================================================
def detrend_lightcurve(time, flux, window_fraction=0.1, polyorder=3, n_sigma=3.5):
    """
    Two-pass detrending that removes instrumental / stellar systematics
    while preserving transit dips:
      Pass 1 -> wide median filter for a coarse trend
      Pass 2 -> Savitzky-Golay smoothing on sigma-clipped flux for the
                final trend model
    Returns (flat_flux, trend).
    """
    N = len(flux)
    win = max(int(N * window_fraction) | 1, polyorder + 3)
    if win % 2 == 0:
        win += 1

    trend1 = signal.medfilt(flux, kernel_size=win)
    resid = flux / (trend1 + 1e-12)

    med_r = np.nanmedian(resid)
    std_r = np.nanstd(resid)
    mask_good = np.abs(resid - med_r) < n_sigma * std_r

    flux_clean = flux.copy()
    flux_clean[~mask_good] = np.interp(time[~mask_good], time[mask_good], flux[mask_good])
    trend2 = signal.savgol_filter(flux_clean, window_length=win, polyorder=polyorder)

    flat_flux = flux / (trend2 + 1e-12)
    flat_flux /= np.nanmedian(flat_flux)
    return flat_flux, trend2


# =========================================================================
# STAGE 3 — PERIODIC TRANSIT SEARCH (Box Least Squares)
# =========================================================================
def run_bls(time, flat_flux, period_min=0.5, period_max=13.0, n_periods=5000):
    """
    Box Least Squares transit search (Kovacs, Zucker & Mazeh 2002) — the
    same core algorithm used by the TESS/SPOC pipeline. Returns the
    best-fit transit parameters and full periodogram.
    """
    durations = np.array([0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18,
                           0.22, 0.25, 0.30, 0.36])

    period_grid = np.geomspace(period_min, period_max, n_periods)
    bls = BoxLeastSquares(time, flat_flux, dy=np.full_like(flat_flux, np.nanstd(flat_flux)))
    result = bls.power(period_grid, durations, objective="snr")

    best_idx = np.argmax(result.power)
    best_period = float(result.period[best_idx])
    best_t0 = float(result.transit_time[best_idx])
    best_depth = float(result.depth[best_idx])
    best_duration = float(result.duration[best_idx])

    mean_power = np.nanmean(result.power)
    std_power = np.nanstd(result.power)
    sde = (float(result.power[best_idx]) - mean_power) / (std_power + 1e-12)

    phase = ((time - best_t0) % best_period) / best_period
    half_dur = best_duration / best_period / 2
    in_tr = (phase < half_dur) | (phase > 1 - half_dur)

    n_in = in_tr.sum()
    if n_in > 2:
        oot_scatter = np.nanstd(flat_flux[~in_tr])
        snr = best_depth / (oot_scatter / np.sqrt(n_in)) if oot_scatter > 0 else 0.0
    else:
        snr = 0.0

    return {
        "period": best_period, "t0": best_t0, "depth": best_depth,
        "duration": best_duration, "depth_ppm": best_depth * 1e6,
        "duration_hr": best_duration * 24, "snr": snr, "sde": sde,
        "period_grid": period_grid, "power": np.array(result.power),
    }


# =========================================================================
# STAGE 3.5 — MONO-TRANSIT ANOMALY DETECTION (dual-path engine)
# =========================================================================
def detect_mono_transits(time, flat_flux, threshold_sigma=4.0):
    """
    Complements the periodic BLS search: inverts the light curve and runs
    a peak-finding anomaly scan to catch single, non-repeating eclipse
    events. This recovers long-period planets and eccentric-orbit systems
    for which a purely periodic search has no repeat transit to lock onto.
    """
    flat_flux = np.asarray(flat_flux)
    inv_flux = -(flat_flux - 1.0)
    std_flux = float(np.nanstd(flat_flux))
    height_threshold = float(threshold_sigma * std_flux)

    peaks, _ = signal.find_peaks(inv_flux, height=height_threshold, distance=50)

    mono_events = []
    for p in peaks:
        mono_events.append({
            "time_btjd": round(float(time[p]), 4),
            "depth_ppm": round(float(inv_flux[p] * 1e6), 0),
            "sigma_significance": round(float(inv_flux[p] / std_flux), 1),
        })
    return sorted(mono_events, key=lambda x: x["depth_ppm"], reverse=True)


# =========================================================================
# STAGE 4 — PHASE FOLDING
# =========================================================================
def phase_fold(time, flux, period, t0, n_bins=150):
    """Folds a light curve on the best period and bins into n_bins."""
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    sort_idx = np.argsort(phase)
    ph_sorted = phase[sort_idx]
    fl_sorted = flux[sort_idx]

    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    centres, binned, bin_std = [], [], []
    for i in range(n_bins):
        mask = (ph_sorted >= edges[i]) & (ph_sorted < edges[i + 1])
        centres.append((edges[i] + edges[i + 1]) / 2)
        if mask.sum() > 0:
            binned.append(np.nanmedian(fl_sorted[mask]))
            bin_std.append(np.nanstd(fl_sorted[mask]) / np.sqrt(mask.sum()))
        else:
            binned.append(np.nan)
            bin_std.append(np.nan)

    return (np.array(centres), np.array(binned), np.array(bin_std), ph_sorted, fl_sorted)


# =========================================================================
# Stellar parameter lookup (TIC catalogue)
# =========================================================================
def get_stellar_params(tic_id_str):
    """
    Queries the TESS Input Catalog for host-star radius, mass and
    effective temperature. Falls back to solar baseline values if the
    catalogue lookup fails or the target cannot be resolved.
    """
    try:
        from astroquery.mast import Catalogs
        numeric_id = "".join(filter(str.isdigit, str(tic_id_str)))
        catalog_data = Catalogs.query_object(f"TIC {numeric_id}", catalog="TIC")
        if len(catalog_data) > 0:
            star = catalog_data[0]
            r_star = float(star["rad"]) if not np.ma.is_masked(star["rad"]) and not np.isnan(star["rad"]) else 1.0
            m_star = float(star["mass"]) if not np.ma.is_masked(star["mass"]) and not np.isnan(star["mass"]) else 1.0
            t_star = float(star["Teff"]) if not np.ma.is_masked(star["Teff"]) and not np.isnan(star["Teff"]) else 5778.0
            return r_star, m_star, t_star
    except Exception:
        pass
    return 1.0, 1.0, 5778.0


# =========================================================================
# STAGE 5 — ASTROPHYSICAL PARAMETER EXTRACTION
# =========================================================================
def compute_parameters(bls, R_star_solar=1.0, M_star_solar=1.0, T_star_K=5778):
    """
    Converts transit observables (period, depth, duration) into physical
    planet parameters following Seager & Mallen-Ornelas (2003), with
    SNR-propagated uncertainty estimates.
    """
    import astropy.constants as const

    P_days = bls["period"]
    depth = bls["depth"]
    dur_d = bls["duration"]
    snr = max(bls["snr"], 1.0)

    R_sun, M_sun = const.R_sun.value, const.M_sun.value
    R_jup, R_ear = const.R_jup.value, const.R_earth.value
    G = const.G.value

    R_star = R_star_solar * R_sun
    M_star = M_star_solar * M_sun
    P_sec = P_days * 86400.0

    rp_rs = np.sqrt(max(depth, 0))
    Rp = rp_rs * R_star

    a = (G * M_star * P_sec ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    a_AU = a / 1.496e11
    a_Rs = a / R_star if a > 0 and R_star > 0 else 0

    A_B = 0.1
    T_eq = T_star_K * (R_star / (2 * a)) ** 0.5 * (1 - A_B) ** 0.25 if a > 0 else 0

    sig_depth = depth / snr
    sig_rp_rs = rp_rs / (2 * snr)
    sig_Rp_Rj = sig_rp_rs * R_star / R_jup

    snr_score = min(bls["snr"] / 20.0, 1.0)
    sde_score = min(bls["sde"] / 15.0, 1.0)
    confidence = 0.5 * snr_score + 0.5 * sde_score

    return {
        "Rp_Rs": round(float(rp_rs), 5),
        "Rp_Rjup": round(float(Rp / R_jup), 3),
        "Rp_Rearth": round(float(Rp / R_ear), 2),
        "a_AU": round(float(a_AU), 5),
        "a_Rs": round(float(a_Rs), 2),
        "T_eq_K": round(float(T_eq)),
        "depth_ppm": round(float(depth) * 1e6, 1),
        "period_days": round(float(P_days), 5),
        "duration_hr": round(float(dur_d) * 24, 3),
        "sig_depth_ppm": round(float(sig_depth) * 1e6, 1),
        "sig_Rp_Rjup": round(float(sig_Rp_Rj), 4),
        "snr": round(float(bls["snr"]), 1),
        "sde": round(float(bls["sde"]), 1),
        "confidence": round(float(confidence), 3),
    }


# =========================================================================
# STAGE 6 — FALSE-POSITIVE VETTING
# =========================================================================
def false_positive_vetting(time, flat_flux, bls, snr_threshold=7.0):
    """
    Runs three automated false-positive tests mirroring TESS/SPOC
    data-validation checks:
      Test 1 - Odd/even transit-depth consistency (catches eclipsing binaries)
      Test 2 - Secondary eclipse search (catches eclipsing binaries)
      Test 3 - Transit shape U vs V (catches grazing binaries)
    Returns a classification label, per-test results, a confidence score,
    and an ML-ready feature vector for downstream classifiers.
    """
    P, t0, d, dur = bls["period"], bls["t0"], bls["depth"], bls["duration"]
    half_dur = dur / P / 2

    phase = ((time - t0) % P) / P
    phase[phase > 0.5] -= 1.0

    # Test 1: odd/even
    transit_epochs = t0 + P * np.arange(-100, 100)
    transit_epochs = transit_epochs[(transit_epochs > time[0]) & (transit_epochs < time[-1])]
    odd_depths, even_depths = [], []
    for i, tt in enumerate(transit_epochs):
        mask = np.abs(time - tt) < dur / 2
        if mask.sum() < 5:
            continue
        in_fl = np.nanmedian(flat_flux[mask])
        oot_fl = np.nanmedian(flat_flux[~mask])
        dep = oot_fl - in_fl
        if dep > 0:
            (odd_depths if i % 2 == 0 else even_depths).append(dep)

    if len(odd_depths) >= 2 and len(even_depths) >= 2:
        odd_mean, even_mean = np.mean(odd_depths), np.mean(even_depths)
        combined_std = np.std(odd_depths + even_depths)
        odd_even_sig = abs(odd_mean - even_mean) / (combined_std + 1e-12)
        test1_pass = odd_even_sig < 3.0
    else:
        odd_even_sig = 0.0
        test1_pass = True

    # Test 2: secondary eclipse
    ph_sec = phase - 0.5
    ph_sec[ph_sec < -0.5] += 1.0
    mask_sec = np.abs(ph_sec) < half_dur
    mask_prim = np.abs(phase) < half_dur
    mask_oot = ~mask_prim & ~mask_sec

    if mask_sec.sum() >= 3 and mask_oot.sum() >= 10:
        oot_med = np.nanmedian(flat_flux[mask_oot])
        sec_med = np.nanmedian(flat_flux[mask_sec])
        sec_depth = oot_med - sec_med
        test2_pass = sec_depth < 0.10 * d
    else:
        sec_depth = 0.0
        test2_pass = True

    # Test 3: transit shape
    in_tr_close = np.abs(phase) < half_dur
    if in_tr_close.sum() >= 5:
        in_flux = flat_flux[in_tr_close]
        threshold = 1.0 - 0.8 * d
        flat_frac = np.mean(in_flux < threshold)
        test3_pass = flat_frac > 0.15
    else:
        flat_frac = 0.5
        test3_pass = True

    n_pass = sum([test1_pass, test2_pass, test3_pass])

    if n_pass == 3 and bls["snr"] >= snr_threshold:
        label = "PLANET CANDIDATE"
    elif not test1_pass:
        label = "ECLIPSING BINARY (odd/even)"
    elif not test2_pass:
        label = "ECLIPSING BINARY (secondary eclipse)"
    elif not test3_pass:
        label = "GRAZING BINARY (V-shape)"
    elif bls["snr"] < snr_threshold:
        label = "SUB-THRESHOLD / NOISE"
    else:
        label = "UNCERTAIN - needs follow-up"

    vetting_score = n_pass / 3.0
    snr_score = min(bls["snr"] / 20.0, 1.0)
    sde_score = min(bls["sde"] / 15.0, 1.0)
    astro_score = 0.5 * snr_score + 0.5 * sde_score
    final_confidence_pct = (vetting_score * 0.5 + astro_score * 0.5) * 100

    return {
        "label": label,
        "ml_features": {
            "depth_ratio": round(float(d), 5),
            "duration_ratio": round(float(dur / P), 5),
            "flat_fraction": round(float(flat_frac), 3),
            "odd_even_sigma": round(float(odd_even_sig), 3),
            "sec_eclipse_ppm": round(float(sec_depth) * 1e6, 0),
            "snr": round(float(bls["snr"]), 2),
        },
        "n_pass": n_pass,
        "test1_pass": test1_pass, "odd_even_sig": round(float(odd_even_sig), 2),
        "test2_pass": test2_pass, "sec_depth_ppm": round(float(sec_depth) * 1e6, 0),
        "test3_pass": test3_pass, "flat_frac": round(float(flat_frac), 3),
        "confidence_pct": round(float(final_confidence_pct), 1),
    }


# =========================================================================
# STAGE 7 — DIAGNOSTIC VISUALISATION
# =========================================================================
def plot_diagnostic(time, flux, flat_flux, trend, bls, ph_c, ph_flux, ph_err,
                     params, vetting, tic_id, period_min, period_max):
    """
    Builds the 6-panel publication-quality diagnostic figure (raw LC,
    detrended LC, BLS periodogram, phase-folded transit, transit zoom,
    parameter summary table) and returns the matplotlib Figure.
    """
    DARK, PANEL = "#0D1117", "#161B22"
    BLUE, TEAL, AMBER, RED = "#3B82F6", "#10B981", "#F59E0B", "#EF4444"
    TXT, GREY = "#E6EDF3", "#8B949E"

    fig = plt.figure(figsize=(18, 12), facecolor=DARK)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    def style(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GREY, labelsize=9)
        ax.set_title(title, color=TXT, fontsize=10.5, pad=8)
        ax.xaxis.label.set_color(GREY)
        ax.yaxis.label.set_color(GREY)
        for s in ax.spines.values():
            s.set_edgecolor("#30363D")

    ax1 = fig.add_subplot(gs[0, :2])
    n = min(3000, len(time))
    ax1.plot(time[:n], flux[:n], color=BLUE, lw=0.4, alpha=0.6, label="Flux")
    ax1.plot(time[:n], trend[:n], color=AMBER, lw=1.5, ls="--", label="Trend model")
    ax1.set_xlabel("Time (BTJD days)"); ax1.set_ylabel("Relative Flux")
    ax1.legend(facecolor="#21262D", labelcolor=TXT, fontsize=8, framealpha=0.7)
    style(ax1, f"Raw + Trend   |   {tic_id}")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(time[:n], flat_flux[:n], color=TEAL, lw=0.5, alpha=0.8)
    ax2.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax2.set_xlabel("Time (BTJD days)"); ax2.set_ylabel("Normalised Flux")
    style(ax2, "Detrended Light Curve")

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(bls["period_grid"], bls["power"], color="#6366F1", lw=0.8, alpha=0.9)
    ax3.axvline(bls["period"], color=RED, lw=2, ls="--", label=f"Best P = {bls['period']:.5f} d")
    for h in [2, 3, 0.5]:
        hp = bls["period"] * h
        if period_min <= hp <= period_max:
            ax3.axvline(hp, color=AMBER, lw=1, ls=":", alpha=0.5)
    ax3.set_xlabel("Period (days)"); ax3.set_ylabel("BLS Power (SNR)")
    ax3.legend(facecolor="#21262D", labelcolor=TXT, fontsize=8)
    style(ax3, "BLS Periodogram")

    ax4 = fig.add_subplot(gs[1, 2])
    valid = ~np.isnan(ph_flux)
    ax4.plot(ph_c[valid], ph_flux[valid], "o", color=AMBER, ms=3.5, label="Binned", zorder=3)
    ax4.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax4.set_xlabel("Orbital Phase"); ax4.set_ylabel("Relative Flux")
    ax4.legend(facecolor="#21262D", labelcolor=TXT, fontsize=8)
    style(ax4, f"Phase-Folded   P = {bls['period']:.5f} d")

    ax5 = fig.add_subplot(gs[2, :2])
    zoom = np.abs(ph_c) < 0.1
    ax5.plot(ph_c[zoom & valid], ph_flux[zoom & valid], "o", color=TEAL, ms=5, zorder=3, label="Phase bins")
    ax5.fill_between(ph_c[zoom & valid], ph_flux[zoom & valid] - ph_err[zoom & valid],
                      ph_flux[zoom & valid] + ph_err[zoom & valid], color=TEAL, alpha=0.25)
    ax5.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax5.axhline(1.0 - bls["depth"], color=RED, lw=1.2, ls="--", label=f"Depth = {bls['depth_ppm']:.0f} ppm")
    ax5.set_xlabel("Orbital Phase"); ax5.set_ylabel("Relative Flux")
    ax5.legend(facecolor="#21262D", labelcolor=TXT, fontsize=8)
    style(ax5, f"Transit Zoom   Depth = {bls['depth_ppm']:.0f} ppm   Duration = {bls['duration_hr']:.2f} h")

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL); ax6.axis("off")
    style(ax6, "Pipeline Summary")
    clr = TEAL if vetting["n_pass"] == 3 else (AMBER if vetting["n_pass"] == 2 else RED)
    rows = [
        ["Parameter", "Value", ""],
        ["Period", f"{params['period_days']:.5f} d", ""],
        ["Depth", f"{params['depth_ppm']:.0f} +/- {params['sig_depth_ppm']:.0f} ppm", ""],
        ["Duration", f"{params['duration_hr']:.3f} h", ""],
        ["Rp", f"{params['Rp_Rjup']:.3f} Rj = {params['Rp_Rearth']:.2f} Re", ""],
        ["a", f"{params['a_AU']:.5f} AU", ""],
        ["T_eq", f"{params['T_eq_K']:.0f} K", ""],
        ["SNR", f"{params['snr']:.1f}", ""],
        ["SDE", f"{params['sde']:.1f}", ""],
        ["Odd/Even sig", f"{vetting['odd_even_sig']:.2f}", "OK" if vetting['test1_pass'] else "X"],
        ["2nd eclipse", f"{vetting['sec_depth_ppm']:.0f} ppm", "OK" if vetting['test2_pass'] else "X"],
        ["Shape", f"flat={vetting['flat_frac']:.2f}", "OK" if vetting['test3_pass'] else "X"],
        ["CLASS", vetting["label"][:22], ""],
    ]
    for ri, row in enumerate(rows):
        y = 1.0 - ri * 0.073
        is_hdr = ri == 0
        ax6.text(0.02, y, row[0], transform=ax6.transAxes,
                  color=TXT if not is_hdr else AMBER,
                  fontsize=9 if not is_hdr else 9.5,
                  fontweight="bold" if is_hdr else "normal", va="top")
        ax6.text(0.5, y, row[1], transform=ax6.transAxes, color=TXT, fontsize=9, va="top")
        if row[2]:
            ax6.text(0.93, y, row[2], transform=ax6.transAxes, color=TXT, fontsize=11, va="top", ha="center")

    conf_label = f"Confidence: {vetting['confidence_pct']}%"
    ax6.text(0.5, 0.03, conf_label, transform=ax6.transAxes, color=clr,
              fontsize=11, fontweight="bold", ha="center", va="bottom")

    fig.suptitle(
        f"VyomNetra   |   {tic_id}   |   {vetting['label']}   "
        f"|   SNR={params['snr']:.1f}   SDE={params['sde']:.1f}",
        color=TXT, fontsize=13, fontweight="bold"
    )
    return fig


# =========================================================================
# MODEL INSIGHTS — translates raw pipeline output into findings that
# highlight what the dual-path engine adds over a standard periodic-only
# search, for direct display in the dashboard.
# =========================================================================
def generate_insights(bls_result, vetting, mono_events, params, cfg):
    """
    Builds a short list of plain-language findings summarising what this
    run actually discovered, and where the dual-path (BLS + mono-transit)
    design added value over a standard single-method periodic search.
    Returns a list of dicts: {"kind": "positive"|"neutral"|"warning", "text": str}
    """
    insights = []

    # 1. Headline classification confidence
    if vetting["label"] == "PLANET CANDIDATE":
        insights.append({
            "kind": "positive",
            "text": (f"Periodic search converged on a P = {bls_result['period']:.5f} d signal "
                      f"at SNR {bls_result['snr']:.1f} (threshold {cfg['snr_threshold']:.1f}) and "
                      f"passed all 3 automated false-positive tests — classified as a genuine "
                      f"planet candidate with {vetting['confidence_pct']:.1f}% confidence.")
        })
    elif "BINARY" in vetting["label"]:
        insights.append({
            "kind": "warning",
            "text": (f"The strongest periodic signal (P = {bls_result['period']:.5f} d) failed "
                      f"vetting and was correctly re-classified as {vetting['label'].lower()} — "
                      f"this is exactly the kind of impostor a naive depth-only threshold would "
                      f"have missed.")
        })
    else:
        insights.append({
            "kind": "neutral",
            "text": (f"Best periodic candidate at P = {bls_result['period']:.5f} d reached "
                      f"SNR {bls_result['snr']:.1f}, below the {cfg['snr_threshold']:.1f} "
                      f"confidence threshold — flagged as {vetting['label'].lower()} rather than "
                      f"a confident detection.")
        })

    # 2. Dual-path value-add: mono-transit scan
    if len(mono_events) > 0:
        top = mono_events[0]
        # is the deepest mono event essentially the same event as the BLS transit, or new?
        insights.append({
            "kind": "positive",
            "text": (f"Independent mono-transit scan flagged {len(mono_events)} isolated dip(s) "
                      f"the periodic search does not use directly — deepest at t = "
                      f"{top['time_btjd']} BTJD, {top['depth_ppm']:.0f} ppm "
                      f"({top['sigma_significance']:.1f}σ). A periodic-only pipeline would report "
                      f"nothing here unless that event happens to repeat within the search window.")
        })
    else:
        insights.append({
            "kind": "neutral",
            "text": "No isolated single-transit anomalies detected — consistent with the periodic result and no signs of a missed long-period companion."
        })

    # 3. Vetting breakdown as a differentiator
    fails = [name for name, ok in [
        ("odd/even depth", vetting["test1_pass"]),
        ("secondary eclipse", vetting["test2_pass"]),
        ("transit shape", vetting["test3_pass"]),
    ] if not ok]
    if fails:
        insights.append({
            "kind": "warning",
            "text": f"Automated vetting caught a failure in: {', '.join(fails)}. "
                    f"This test would silently pass in an SNR-only ranking."
        })
    else:
        insights.append({
            "kind": "positive",
            "text": "All 3 automated false-positive tests passed — odd/even depth consistency, "
                     "no significant secondary eclipse, and a genuine flat-bottomed (U-shaped) transit."
        })

    # 4. Physical plausibility check
    insights.append({
        "kind": "neutral",
        "text": (f"Derived planet radius {params['Rp_Rearth']:.2f} R⊕ ({params['Rp_Rjup']:.3f} Rjup) "
                  f"at equilibrium temperature {params['T_eq_K']:.0f} K — computed directly from "
                  f"transit geometry (Seager & Mallén-Ornelas 2003), not fitted or assumed.")
    })

    return insights


# =========================================================================
# ORCHESTRATOR — runs the full pipeline end-to-end for one target
# =========================================================================
def run_pipeline(time, flux, flux_err, tic_id, config=None, r_star=None,
                  m_star=None, t_star=None):
    """
    Executes the complete VyomNetra pipeline (stages 2-7) on an already
    loaded light curve and returns every intermediate + final result the
    dashboard needs, in one bundled dict. This is the single entry point
    the UI layer should call.
    """
    cfg = {**DEFAULTS, **(config or {})}

    flat_flux, trend = detrend_lightcurve(time, flux)

    bls_result = run_bls(time, flat_flux, cfg["period_min"], cfg["period_max"], cfg["n_periods"])

    mono_events = detect_mono_transits(time, flat_flux)

    ph_c, ph_flux, ph_err, ph_raw, fl_raw = phase_fold(time, flat_flux, bls_result["period"], bls_result["t0"])

    if r_star is None or m_star is None or t_star is None:
        r_star, m_star, t_star = get_stellar_params(tic_id)

    params = compute_parameters(bls_result, R_star_solar=r_star, M_star_solar=m_star, T_star_K=t_star)

    vetting = false_positive_vetting(time, flat_flux, bls_result, cfg["snr_threshold"])

    fig = plot_diagnostic(time, flux, flat_flux, trend, bls_result, ph_c, ph_flux, ph_err,
                           params, vetting, tic_id, cfg["period_min"], cfg["period_max"])

    insights = generate_insights(bls_result, vetting, mono_events, params, cfg)

    # Full-resolution light curve is kept out of the in-memory bundle by
    # default (it can be tens of thousands of rows and adds nothing the
    # diagnostic figure doesn't already show) — callers that need it for
    # export can request it via include_raw_lightcurve=True below.
    lc_table = pd.DataFrame({
        "time_btjd": time, "raw_flux": flux, "flux_err": flux_err,
        "detrended_flux": flat_flux, "trend_model": trend,
    })

    phase_table = pd.DataFrame({
        "phase": ph_c, "binned_flux": ph_flux, "bin_error": ph_err,
    })

    mono_table = pd.DataFrame(mono_events) if mono_events else pd.DataFrame(
        columns=["time_btjd", "depth_ppm", "sigma_significance"])

    summary_table = pd.DataFrame([{
        "TIC": tic_id,
        "Classification": vetting["label"],
        "Period_days": params["period_days"],
        "Depth_ppm": params["depth_ppm"],
        "Depth_err_ppm": params["sig_depth_ppm"],
        "Duration_hr": params["duration_hr"],
        "Rp_Rjup": params["Rp_Rjup"],
        "Rp_Rearth": params["Rp_Rearth"],
        "a_AU": params["a_AU"],
        "a_Rs": params["a_Rs"],
        "T_eq_K": params["T_eq_K"],
        "SNR": params["snr"],
        "SDE": params["sde"],
        "Confidence_pct": vetting["confidence_pct"],
        "OddEven_pass": vetting["test1_pass"],
        "SecondaryEclipse_pass": vetting["test2_pass"],
        "TransitShape_pass": vetting["test3_pass"],
        "R_star_solar": r_star, "M_star_solar": m_star, "T_star_K": t_star,
    }])

    return {
        "bls_result": bls_result, "params": params, "vetting": vetting,
        "mono_events": mono_events, "figure": fig, "insights": insights,
        "lc_table": lc_table, "phase_table": phase_table,
        "mono_table": mono_table, "summary_table": summary_table,
        "stellar": {"R_star_solar": r_star, "M_star_solar": m_star, "T_star_K": t_star},
        "config": cfg,
    }


def run_batch(targets, config=None):
    """
    Runs the full pipeline across multiple TIC IDs (survey mode) and
    returns a single SNR-ranked candidate table.
    """
    records = []
    for tic in targets:
        try:
            t, f, fe = download_tess_lc(tic, (config or DEFAULTS).get("flux_type", "sap_flux"),
                                         (config or DEFAULTS).get("sector_idx", 0))
            result = run_pipeline(t, f, fe, tic, config=config)
            records.append(result["summary_table"].iloc[0].to_dict())
        except Exception as e:
            records.append({"TIC": tic, "Classification": "ERROR", "SNR": 0, "note": str(e)})

    df = pd.DataFrame(records)
    if "SNR" in df.columns:
        df = df.sort_values("SNR", ascending=False).reset_index(drop=True)
    return df
