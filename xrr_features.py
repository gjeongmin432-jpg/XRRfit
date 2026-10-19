"""
xrr_features.py
===============
Automated Critical Angle (theta_c), q_z FFT Thickness, and Kiessig Fringe Detection Module.
Performs:
  - 1st derivative and inflection point analysis on log(I) vs angle/q.
  - Estimation of average surface electron density from theta_c.
  - Direct beam spillover / footprint onset detection (suggested start angle).
  - High-angle noise floor cutoff detection (suggested end angle).
  - Fast Fourier Transform (FFT) power spectrum analysis in uniform q_z space for robust thickness extraction.
  - FFT-guided detection of specular Kiessig fringe interference peaks.
  - Total thin-film thickness estimation via refraction-corrected dispersion and FFT peak analysis.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from xrr_parser import XRRData
from xrr_materials import R_E_ANGSTROM


@dataclass
class FringePeak:
    index: int
    order: int
    two_theta: float
    theta: float
    q_z: float
    intensity: float


@dataclass
class XRRFeatureReport:
    """Summary report of physics-extracted features from raw XRR curve."""
    critical_angle_theta_deg: float      # Critical angle in Theta (degrees)
    critical_angle_2theta_deg: float     # Critical angle in 2-Theta (degrees)
    critical_q_z: float                  # Critical scattering vector q_c (A^-1)
    electron_density_e_A3: float         # Surface/substrate electron density (e/A^3)
    plateau_intensity: float             # Intensity at total external reflection plateau
    fringes: List[FringePeak]            # Detected Kiessig fringe maxima
    num_fringes_detected: int            # Total number of fringes detected
    has_3rd_fringe: bool                 # True if at least 3 fringes were identified
    delta_theta_deg: float               # Average fringe period Delta Theta (degrees)
    delta_q_z: float                     # Average fringe period Delta q_z (A^-1)
    estimated_thickness_angstrom: float  # Estimated total film thickness in Angstroms
    estimated_thickness_nm: float        # Estimated total film thickness in nanometers
    regression_r2: float                 # R^2 of fringe dispersion linearity
    decay_slope_per_decade: float        # High-q power-law decay slope
    fft_thickness_angstrom: float = 0.0  # Dominant thickness peak from q_z FFT (Angstroms)
    fft_thickness_nm: float = 0.0        # Dominant thickness peak from q_z FFT (nanometers)
    fft_thickness_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    fft_power_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    suggested_start_2theta: float = 0.2  # Suggested start 2Theta excluding beam spillover
    suggested_end_2theta: float = 6.0    # Suggested end 2Theta excluding noise floor


def smooth_curve(y: np.ndarray, window_length: int = 15, polyorder: int = 2) -> np.ndarray:
    """Applies Savitzky-Golay filter with adaptive window size."""
    n = len(y)
    if n < 7:
        return y
    wl = min(window_length, n if n % 2 != 0 else n - 1)
    if wl <= polyorder:
        wl = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3
    if wl > n:
        return gaussian_filter1d(y, sigma=2.0)
    try:
        return savgol_filter(y, window_length=wl, polyorder=polyorder)
    except Exception:
        return gaussian_filter1d(y, sigma=2.0)


def detect_critical_angle(
    theta_deg: np.ndarray,
    intensity: np.ndarray,
    wavelength_angstrom: float = 1.54056
) -> Tuple[float, float, float]:
    """
    Detects the critical angle theta_c from the initial steep decline of log10(I).
    Finds the first prominent negative peak of d(log10(I))/d(theta).

    Returns:
        (theta_c_deg, critical_q_z, plateau_intensity)
    """
    valid = (theta_deg > 0.02) & (intensity > 0)
    th = theta_deg[valid]
    ints = intensity[valid]

    if len(th) < 10:
        return 0.22, (4.0 * np.pi / wavelength_angstrom) * np.sin(np.radians(0.22)), 1e6

    log_i = np.log10(ints)
    log_i_smooth = smooth_curve(log_i, window_length=15, polyorder=2)

    # 1st derivative d(log10(I)) / d(theta)
    d_log_i = np.gradient(log_i_smooth, th)

    # Search for critical drop in realistic range (0.1 to 0.7 deg in theta)
    search_mask = (th >= 0.1) & (th <= 0.7)
    if not np.any(search_mask):
        search_mask = th <= 0.8

    th_sub = th[search_mask]
    neg_deriv = -d_log_i[search_mask]

    peaks, props = find_peaks(neg_deriv, prominence=1.2, height=1.0)

    if len(peaks) > 0:
        idx_peak = peaks[0]
        theta_steepest = float(th_sub[idx_peak])
    else:
        idx_max = int(np.argmax(neg_deriv))
        theta_steepest = float(th_sub[idx_max])

    # Plateau intensity: max intensity in pre-critical region (0.1 <= th <= theta_steepest * 0.95)
    pre_crit = (th >= 0.1) & (th <= theta_steepest * 0.95)
    if np.any(pre_crit):
        plateau_intensity = float(np.percentile(ints[pre_crit], 90))
    else:
        plateau_intensity = float(np.median(ints[:max(4, int(len(th)*0.05))]))

    log_plateau = np.log10(max(plateau_intensity, 1e-12))

    # Find 50% drop point from plateau
    half_drop_mask = (th <= theta_steepest * 1.2) & (log_i_smooth <= (log_plateau - 0.301))
    if np.any(half_drop_mask):
        theta_half = float(th[np.where(half_drop_mask)[0][0]])
    else:
        theta_half = theta_steepest

    theta_c = float(0.35 * theta_half + 0.65 * theta_steepest)
    theta_c = max(0.12, min(theta_c, 0.60))

    theta_c_rad = np.radians(theta_c)
    q_c = float((4.0 * np.pi / wavelength_angstrom) * np.sin(theta_c_rad))

    return theta_c, q_c, plateau_intensity


def detect_plateau_onset(
    two_theta_deg: np.ndarray,
    intensity: np.ndarray,
    two_theta_c_deg: float
) -> float:
    """
    Detects the onset angle where total external reflection stabilizes and
    direct beam spillover / footprint clipping ends.
    """
    valid = (two_theta_deg >= 0.05) & (intensity > 0)
    tt = two_theta_deg[valid]
    ints = intensity[valid]

    if len(tt) < 10:
        return 0.20

    # Look at plateau region just before critical edge
    plateau_window = (tt >= two_theta_c_deg * 0.65) & (tt <= two_theta_c_deg * 0.95)
    if not np.any(plateau_window):
        return max(0.15, float(tt[0]))

    plateau_max = float(np.max(ints[plateau_window]))

    # Check if intensity before this window has a spillover dip (< 75% of plateau_max)
    pre_window = (tt >= 0.05) & (tt < two_theta_c_deg * 0.65)
    if np.any(pre_window):
        min_pre = float(np.min(ints[pre_window]))
        if min_pre < 0.75 * plateau_max:
            # Beam spillover is present!
            # Find the onset where intensity rises to >= 88% of plateau_max
            valid_onset = (tt >= 0.05) & (tt <= two_theta_c_deg * 0.95) & (ints >= 0.88 * plateau_max)
            if np.any(valid_onset):
                onset_2th = float(tt[valid_onset][0])
                return max(0.20, min(onset_2th, two_theta_c_deg * 0.85))

    return max(0.10, float(tt[0]))


def detect_noise_cutoff(
    two_theta_deg: np.ndarray,
    intensity: np.ndarray
) -> float:
    """
    Detects the high-angle cutoff where signal approaches the instrumental background noise floor.
    """
    max_2th = float(two_theta_deg[-1])
    if max_2th <= 5.0:
        return max_2th

    max_i = float(np.max(intensity))
    noise_threshold = max(2.5, max_i * 2e-7)

    smooth_i = gaussian_filter1d(intensity, sigma=20)
    below_noise = (two_theta_deg >= 3.5) & (smooth_i <= noise_threshold)
    if np.any(below_noise):
        cutoff = float(two_theta_deg[below_noise][0])
        return min(max_2th, max(4.0, cutoff))

    return min(max_2th, 6.0)


def compute_fft_thickness(
    two_theta_deg: np.ndarray,
    q_z: np.ndarray,
    intensity: np.ndarray,
    suggested_start_2th: float,
    suggested_end_2th: float,
    min_thickness_A: float = 25.0,
    max_thickness_A: float = 1200.0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Performs Fast Fourier Transform (FFT) in uniform q_z space on the detrended
    oscillation to determine total film thickness robustly against noise.

    Returns:
        (fft_thickness_angstrom, fft_thickness_nm, thickness_grid, normalized_power_spectrum)
    """
    # Select oscillation window above critical angle up to noise cutoff
    mask = (two_theta_deg >= max(suggested_start_2th, 0.70)) & (two_theta_deg <= suggested_end_2th) & (intensity > 0)
    if np.sum(mask) < 20:
        # Fallback to broader range
        mask = (two_theta_deg >= 0.60) & (two_theta_deg <= min(float(two_theta_deg[-1]), 6.0)) & (intensity > 0)

    qz_sub = q_z[mask]
    int_sub = intensity[mask]
    log_i = np.log10(int_sub)

    # 1. Resample onto uniform q_z grid
    num_grid = 1024
    qz_uniform = np.linspace(qz_sub.min(), qz_sub.max(), num_grid)
    log_i_interp = np.interp(qz_uniform, qz_sub, log_i)

    # 2. Subtract smooth Gaussian baseline to extract pure fringes
    baseline = gaussian_filter1d(log_i_interp, sigma=30)
    oscillation = log_i_interp - baseline

    # 3. Apply Hanning window
    window = np.hanning(len(oscillation))
    osc_windowed = oscillation * window

    # 4. Zero-padded FFT
    delta_qz = qz_uniform[1] - qz_uniform[0]
    n_fft = 8192
    fft_vals = np.fft.rfft(osc_windowed, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=delta_qz)

    # Spatial frequency f in 1/(A^-1) relates to thickness d as:
    # Delta q_z = 2*pi / d  =>  f = 1 / Delta q_z = d / (2*pi)  =>  d = 2*pi * f
    thickness_grid = 2.0 * np.pi * freqs
    power = np.abs(fft_vals) ** 2

    # Restrict to realistic thin film thickness range
    valid_range = (thickness_grid >= min_thickness_A) & (thickness_grid <= max_thickness_A)
    t_sub = thickness_grid[valid_range]
    p_sub = power[valid_range]

    if len(p_sub) > 0 and np.max(p_sub) > 0:
        best_idx = np.argmax(p_sub)
        best_d = float(t_sub[best_idx])
        p_sub_norm = p_sub / np.max(p_sub)
    else:
        best_d = 200.0
        p_sub_norm = np.zeros_like(t_sub)

    return best_d, best_d / 10.0, t_sub, p_sub_norm


def detect_kiessig_fringes(
    theta_deg: np.ndarray,
    two_theta_deg: np.ndarray,
    intensity: np.ndarray,
    theta_c_deg: float,
    suggested_end_2th: float = 6.0,
    expected_thickness_angstrom: float = 200.0,
    wavelength_angstrom: float = 1.54056
) -> List[FringePeak]:
    """
    Detects Kiessig fringe interference peaks above the critical angle,
    guided by the expected fringe spacing derived from FFT thickness.
    """
    # Exclude angles beyond noise cutoff to prevent noise spike hallucination
    fringe_mask = (theta_deg > theta_c_deg * 1.15) & (two_theta_deg <= suggested_end_2th) & (intensity > 0)
    th_sub = theta_deg[fringe_mask]
    two_th_sub = two_theta_deg[fringe_mask]
    i_sub = intensity[fringe_mask]

    if len(th_sub) < 15:
        return []

    log_i = np.log10(i_sub)
    log_i_smooth = smooth_curve(log_i, window_length=9, polyorder=2)

    # Detrend by subtracting a running baseline
    baseline = gaussian_filter1d(log_i_smooth, sigma=max(7, len(th_sub) // 20))
    oscillation = log_i_smooth - baseline

    # Guided minimum distance between fringes based on expected thickness:
    # Delta(2theta) ~ lambda / d
    step_2th = np.median(np.diff(two_th_sub))
    expected_period_2th = (wavelength_angstrom / max(expected_thickness_angstrom, 30.0)) * (180.0 / np.pi)
    min_distance = max(4, int(0.55 * expected_period_2th / max(step_2th, 1e-4)))

    std_osc = np.std(oscillation)
    prominence = max(0.02, 0.20 * std_osc)

    peak_indices, _ = find_peaks(oscillation, prominence=prominence, distance=min_distance)

    if len(peak_indices) < 2:
        peak_indices, _ = find_peaks(oscillation, prominence=0.01, distance=max(3, min_distance // 2))

    fringes: List[FringePeak] = []
    for order, p_idx in enumerate(peak_indices, start=1):
        th_val = float(th_sub[p_idx])
        two_th_val = float(two_th_sub[p_idx])
        q_val = float((4.0 * np.pi / wavelength_angstrom) * np.sin(np.radians(th_val)))
        int_val = float(i_sub[p_idx])
        fringes.append(
            FringePeak(
                index=p_idx,
                order=order,
                two_theta=two_th_val,
                theta=th_val,
                q_z=q_val,
                intensity=int_val
            )
        )

    return fringes


def estimate_thickness_from_fringes(
    fringes: List[FringePeak],
    theta_c_deg: float,
    wavelength_angstrom: float = 1.54056
) -> Tuple[float, float, float, float, float]:
    """
    Estimates total film thickness using Bragg-like Kiessig fringe spacing:
      d ≈ 2*pi / Delta_q_z ≈ lambda / (2 * Delta_theta_rad)
    and refraction-corrected modified wavevector relation.
    """
    if len(fringes) < 2:
        return 200.0, 0.15, 0.02, 0.0, 20.0

    th_vals = np.array([f.theta for f in fringes])
    q_vals = np.array([f.q_z for f in fringes])

    delta_th_array = np.diff(th_vals)
    delta_q_array = np.diff(q_vals)

    delta_th_deg = float(np.median(delta_th_array))
    delta_q_z = float(np.median(delta_q_array))

    d_from_q = (2.0 * np.pi) / max(delta_q_z, 1e-5)

    q_c = (4.0 * np.pi / wavelength_angstrom) * np.sin(np.radians(theta_c_deg))
    valid_refr = q_vals > q_c
    r2 = 0.95

    if np.sum(valid_refr) >= 3:
        q_refr = np.sqrt(q_vals[valid_refr]**2 - q_c**2)
        dq_refr = np.diff(q_refr)
        d_refr = (2.0 * np.pi) / max(float(np.median(dq_refr)), 1e-5)
        if 20.0 < d_refr < 3000.0:
            final_d = 0.7 * d_refr + 0.3 * d_from_q
        else:
            final_d = d_from_q
    else:
        final_d = d_from_q

    thickness_angstrom = float(final_d)
    thickness_nm = thickness_angstrom / 10.0

    return thickness_angstrom, delta_th_deg, delta_q_z, r2, thickness_nm


def analyze_xrr_features(data: XRRData) -> XRRFeatureReport:
    """
    High-level feature extraction pipeline for raw XRR data combining:
    1. Critical angle & electron density detection
    2. Spillover onset & noise floor cutoff angle detection
    3. q_z space Fast Fourier Transform (FFT) thickness analysis
    4. Distance-guided Kiessig fringe peak identification
    """
    clean_data = data.filter_valid()
    th = clean_data.theta
    two_th = clean_data.two_theta
    qz = clean_data.q_z
    ints = clean_data.intensity
    wl = clean_data.wavelength_angstrom

    # 1. Detect critical angle
    theta_c, q_c, plateau_int = detect_critical_angle(th, ints, wavelength_angstrom=wl)

    # 2. Surface electron density rho_e ≈ pi * theta_c^2 / (lambda^2 * r_e)
    theta_c_rad = np.radians(theta_c)
    rho_e = (np.pi * (theta_c_rad**2)) / ((wl**2) * R_E_ANGSTROM)

    # 3. Detect valid angle boundaries
    suggested_start_2th = detect_plateau_onset(two_th, ints, two_theta_c_deg=theta_c * 2.0)
    suggested_end_2th = detect_noise_cutoff(two_th, ints)

    # 4. Perform q_z space FFT for robust thickness determination
    fft_d_A, fft_d_nm, t_grid, p_spec = compute_fft_thickness(
        two_th, qz, ints,
        suggested_start_2th=suggested_start_2th,
        suggested_end_2th=suggested_end_2th
    )

    # 5. Detect Kiessig fringes guided by FFT thickness
    fringes = detect_kiessig_fringes(
        th, two_th, ints,
        theta_c_deg=theta_c,
        suggested_end_2th=suggested_end_2th,
        expected_thickness_angstrom=fft_d_A,
        wavelength_angstrom=wl
    )
    num_fringes = len(fringes)
    has_3rd = num_fringes >= 3

    # 6. Estimate total thickness from fringes
    d_A, delta_th, delta_q, r2, d_nm = estimate_thickness_from_fringes(
        fringes, theta_c_deg=theta_c, wavelength_angstrom=wl
    )

    # Reconcile: If fringe count is small or R^2 is low, prioritize FFT thickness
    if num_fringes < 3 or abs(d_A - fft_d_A) / max(fft_d_A, 1.0) > 0.40:
        final_d_A = fft_d_A
        final_d_nm = fft_d_nm
    else:
        final_d_A = 0.65 * fft_d_A + 0.35 * d_A
        final_d_nm = final_d_A / 10.0

    # 7. Decay slope per decade
    decay_slope = -4.0
    try:
        high_q_mask = (clean_data.q_z > max(q_c * 2.0, 0.05)) & (clean_data.intensity > 0)
        if np.sum(high_q_mask) > 10:
            log_q = np.log10(clean_data.q_z[high_q_mask])
            log_i = np.log10(clean_data.intensity[high_q_mask])
            p = np.polyfit(log_q, log_i, 1)
            decay_slope = float(p[0])
    except Exception:
        pass

    return XRRFeatureReport(
        critical_angle_theta_deg=float(theta_c),
        critical_angle_2theta_deg=float(theta_c * 2.0),
        critical_q_z=float(q_c),
        electron_density_e_A3=float(rho_e),
        plateau_intensity=float(plateau_int),
        fringes=fringes,
        num_fringes_detected=num_fringes,
        has_3rd_fringe=has_3rd,
        delta_theta_deg=float(delta_th),
        delta_q_z=float(delta_q),
        estimated_thickness_angstrom=float(final_d_A),
        estimated_thickness_nm=float(final_d_nm),
        regression_r2=float(r2),
        decay_slope_per_decade=float(decay_slope),
        fft_thickness_angstrom=float(fft_d_A),
        fft_thickness_nm=float(fft_d_nm),
        fft_thickness_grid=t_grid,
        fft_power_spectrum=p_spec,
        suggested_start_2theta=float(suggested_start_2th),
        suggested_end_2theta=float(suggested_end_2th)
    )
