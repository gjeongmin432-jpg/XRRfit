"""
xrr_engine.py
=============
Multi-Physics Simulation Engines and Figure-of-Merit (FOM) Benchmarking for XRR.
Includes:
  - Engine A: High-Performance Vectorized Parratt Recursion + Névot-Croce Roughness Factor.
  - Engine B: Abelès Characteristic Transfer Matrix Method + Croce-Névot Damping.
  - Figure-of-Merit (FOM) calculators: Log-MSE, Crystallographic R-factor, Reduced Chi-squared.
  - Multi-engine ensemble benchmarking and automated optimal engine selection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from xrr_materials import calculate_optical_constants, MaterialProperties


@dataclass
class Layer:
    """
    Physical representation of a single layer in the XRR stack.
    Ordered from Top (Surface) to Bottom (Substrate).
    """
    name: str
    formula: str
    thickness: float     # Thickness in Angstroms (0 for ambient, infinite/ignored for substrate)
    roughness: float     # Root-mean-square roughness sigma in Angstroms at interface with layer above
    density: float       # Mass density in g/cm^3
    delta: float = 0.0   # Refractive index decrement
    beta: float = 0.0    # Absorption decrement

    @property
    def thickness_nm(self) -> float:
        return self.thickness / 10.0

    @property
    def roughness_nm(self) -> float:
        return self.roughness / 10.0

    def update_optical_constants(self, wavelength_angstrom: float = 1.54056):
        """Recomputes delta and beta based on formula and density."""
        props = calculate_optical_constants(self.formula, self.density, wavelength_angstrom)
        self.delta = props.delta
        self.beta = props.beta


@dataclass
class LayerStack:
    """
    Complete layer model representing:
      Ambient (Layer 0) -> Film Layers [1 ... N] -> Substrate (Layer N+1).
    """
    ambient: Layer
    layers: List[Layer]   # Ordered from Top film layer down to bottom film layer
    substrate: Layer
    wavelength_angstrom: float = 1.54056

    def get_all_layers(self) -> List[Layer]:
        """Returns ordered list [Ambient, Layer_top, ..., Layer_bottom, Substrate]."""
        return [self.ambient] + self.layers + [self.substrate]

    def update_all_optical_constants(self):
        """Updates optical constants for all components in the stack."""
        self.ambient.update_optical_constants(self.wavelength_angstrom)
        for layer in self.layers:
            layer.update_optical_constants(self.wavelength_angstrom)
        self.substrate.update_optical_constants(self.wavelength_angstrom)

    @property
    def total_film_thickness(self) -> float:
        """Total thickness of film layers in Angstroms (excluding substrate and ambient)."""
        return sum(layer.thickness for layer in self.layers)

    @property
    def total_film_thickness_nm(self) -> float:
        return self.total_film_thickness / 10.0


@dataclass
class EngineResult:
    engine_name: str
    reflectivity: np.ndarray
    simulated_intensity: np.ndarray
    log_mse: float
    r_factor: float
    reduced_chi2: float
    computation_time_ms: float = 0.0


def _compute_kz_array(
    all_layers: List[Layer],
    theta_deg: np.ndarray,
    k0: float
) -> np.ndarray:
    """
    Computes kz = k0 * sqrt(sin^2(theta) - 2*delta + 2j*beta)
    Ensuring positive real and positive imaginary parts for physical propagation and absorption.
    """
    num_media = len(all_layers)
    theta_rad = np.radians(theta_deg)
    sin_sq = np.sin(theta_rad)**2

    kz = np.zeros((num_media, len(theta_deg)), dtype=np.complex128)
    for j, layer in enumerate(all_layers):
        # arg = sin^2(theta) - 2*delta + 2j*beta
        arg = sin_sq - 2.0 * layer.delta + 2.0j * layer.beta
        k_val = k0 * np.sqrt(arg)
        # Ensure positive real and imaginary parts
        k_val = np.where(np.real(k_val) < 0, -k_val, k_val)
        k_val = np.where(np.imag(k_val) < 0, np.conj(k_val), k_val)
        kz[j] = k_val

    return kz


# ----------------------------------------------------------------------
# Instrumental Optics & Footprint Corrections
# ----------------------------------------------------------------------

def apply_optics_corrections(
    intensity: np.ndarray,
    theta_deg: np.ndarray,
    scale: float,
    background: float,
    sample_length_mm: Optional[float] = None,
    beam_width_mm: float = 0.1,
    resolution_fwhm_deg: Optional[float] = None
) -> np.ndarray:
    """
    Applies real-world physical instrument corrections to simulated XRR curve:
      1. Geometric Footprint / Beam Spillover Correction:
         F(theta) = min(1.0, (L * sin(theta)) / W)
         Accounts for the beam exceeding sample dimensions at shallow angles.
      2. Instrumental Angular Resolution Smearing (Slit Divergence):
         Convolves with Gaussian kernel representing finite beam divergence / slit aperture.
    """
    # 1. Footprint Spillover Factor
    i_signal = intensity - background  # pure scale * R
    if sample_length_mm is not None and sample_length_mm > 0 and beam_width_mm > 0:
        th_rad = np.radians(theta_deg)
        footprint = np.minimum(1.0, (sample_length_mm * np.sin(th_rad)) / beam_width_mm)
        i_signal = i_signal * footprint

    i_corrected = i_signal + background

    # 2. Gaussian Resolution Convolution
    if resolution_fwhm_deg is not None and resolution_fwhm_deg > 0 and len(theta_deg) > 5:
        from scipy.ndimage import gaussian_filter1d
        d_th = float(np.median(np.diff(theta_deg)))
        if d_th > 0:
            sigma_deg = abs(resolution_fwhm_deg) / 2.35482  # FWHM to Gaussian sigma
            sigma_pts = max(0.4, sigma_deg / d_th)
            i_corrected = gaussian_filter1d(i_corrected, sigma=sigma_pts)

    return i_corrected


# ----------------------------------------------------------------------
# Engine A: Parratt Recursion + Névot-Croce Roughness
# ----------------------------------------------------------------------

def simulate_parratt(
    theta_deg: np.ndarray,
    stack: LayerStack,
    scale: float = 1.0,
    background: float = 1e-7,
    wavelength_angstrom: Optional[float] = None,
    sample_length_mm: Optional[float] = None,
    beam_width_mm: float = 0.1,
    resolution_fwhm_deg: Optional[float] = None
) -> np.ndarray:
    """
    Engine A: Vectorized Parratt recursion with Névot-Croce interface roughness,
    optional sample footprint correction, and instrumental resolution smearing.
    """
    wl = wavelength_angstrom if wavelength_angstrom is not None else stack.wavelength_angstrom
    k0 = 2.0 * np.pi / wl

    all_layers = stack.get_all_layers()
    num_media = len(all_layers)

    kz = _compute_kz_array(all_layers, theta_deg, k0)

    # Base case: Interface between bottom-most film layer (num_media - 2) and substrate (num_media - 1)
    last_interface_idx = num_media - 2
    kz_top = kz[last_interface_idx]
    kz_bot = kz[last_interface_idx + 1]
    sigma_bot = all_layers[last_interface_idx + 1].roughness

    denom = kz_top + kz_bot
    denom = np.where(np.abs(denom) == 0, 1e-15, denom)
    r_last = (kz_top - kz_bot) / denom
    if sigma_bot > 0:
        r_last *= np.exp(-2.0 * kz_top * kz_bot * (sigma_bot**2))

    X = r_last

    # Recurse upwards from layer (num_media - 3) down to ambient (0)
    for j in range(num_media - 3, -1, -1):
        kz_j = kz[j]
        kz_next = kz[j + 1]
        sigma = all_layers[j + 1].roughness
        d_next = all_layers[j + 1].thickness

        # Interface Fresnel coefficient
        denom = kz_j + kz_next
        denom = np.where(np.abs(denom) == 0, 1e-15, denom)
        r_j = (kz_j - kz_next) / denom
        if sigma > 0:
            r_j *= np.exp(-2.0 * kz_j * kz_next * (sigma**2))

        # Phase propagation through layer j+1
        phase = np.exp(2.0j * kz_next * d_next)

        # Parratt recurrence relation
        numerator = r_j + X * phase
        denominator = 1.0 + r_j * X * phase
        denominator = np.where(np.abs(denominator) == 0, 1e-15, denominator)
        X = numerator / denominator

    reflectivity = np.abs(X)**2
    intensity = scale * reflectivity + background

    if sample_length_mm is not None or resolution_fwhm_deg is not None:
        intensity = apply_optics_corrections(
            intensity, theta_deg, scale, background,
            sample_length_mm=sample_length_mm,
            beam_width_mm=beam_width_mm,
            resolution_fwhm_deg=resolution_fwhm_deg
        )

    return intensity


# ----------------------------------------------------------------------
# Engine B: Abelès Characteristic Matrix Method
# ----------------------------------------------------------------------

def simulate_abeles(
    theta_deg: np.ndarray,
    stack: LayerStack,
    scale: float = 1.0,
    background: float = 1e-7,
    wavelength_angstrom: Optional[float] = None,
    sample_length_mm: Optional[float] = None,
    beam_width_mm: float = 0.1,
    resolution_fwhm_deg: Optional[float] = None
) -> np.ndarray:
    """
    Engine B: Abelès Transfer Matrix Method with Croce-Névot Roughness damping,
    optional sample footprint correction, and instrumental resolution smearing.
    """
    wl = wavelength_angstrom if wavelength_angstrom is not None else stack.wavelength_angstrom
    k0 = 2.0 * np.pi / wl

    all_layers = stack.get_all_layers()
    num_media = len(all_layers)

    kz = _compute_kz_array(all_layers, theta_deg, k0)

    n_points = len(theta_deg)
    # Characteristic transfer matrix initialized to identity 2x2
    M11 = np.ones(n_points, dtype=np.complex128)
    M12 = np.zeros(n_points, dtype=np.complex128)
    M21 = np.zeros(n_points, dtype=np.complex128)
    M22 = np.ones(n_points, dtype=np.complex128)

    # Multiply intermediate layer matrices (j = 1 to num_media - 2)
    for j in range(1, num_media - 1):
        d = all_layers[j].thickness
        sigma = all_layers[j].roughness
        k_val = kz[j]
        
        phase = k_val * d
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        inv_k = np.where(np.abs(k_val) == 0, 1e-15, 1.0 / k_val)

        # Single-layer characteristic matrix
        L11 = cos_p
        L12 = -1.0j * inv_k * sin_p
        L21 = -1.0j * k_val * sin_p
        L22 = cos_p

        # Apply interface damping
        if sigma > 0:
            damping = np.exp(-0.5 * ((np.real(k_val) * sigma)**2))
            L12 *= damping
            L21 *= damping

        # M_total = M_total * L_j
        next_M11 = M11 * L11 + M12 * L21
        next_M12 = M11 * L12 + M12 * L22
        next_M21 = M21 * L11 + M22 * L21
        next_M22 = M21 * L12 + M22 * L22

        M11, M12, M21, M22 = next_M11, next_M12, next_M21, next_M22

    kz0 = kz[0]
    kzs = kz[-1]

    # Calculate reflection coefficient:
    # r = ((M11 + M12 * kzs) * kz0 - (M21 + M22 * kzs)) / ((M11 + M12 * kzs) * kz0 + (M21 + M22 * kzs))
    term1 = M11 + M12 * kzs
    term2 = M21 + M22 * kzs

    sub_sigma = all_layers[-1].roughness
    sub_damping = np.exp(-2.0 * np.real(kz0) * np.real(kzs) * (sub_sigma**2)) if sub_sigma > 0 else 1.0

    numerator = (term1 * kz0 - term2) * sub_damping
    denominator = term1 * kz0 + term2
    denominator = np.where(np.abs(denominator) == 0, 1e-15, denominator)

    r = numerator / denominator
    reflectivity = np.abs(r)**2
    intensity = scale * reflectivity + background

    if sample_length_mm is not None or resolution_fwhm_deg is not None:
        intensity = apply_optics_corrections(
            intensity, theta_deg, scale, background,
            sample_length_mm=sample_length_mm,
            beam_width_mm=beam_width_mm,
            resolution_fwhm_deg=resolution_fwhm_deg
        )

    return intensity



# ----------------------------------------------------------------------
# Figure of Merit (FOM) Metrics & Comparison
# ----------------------------------------------------------------------

def calculate_fom(
    i_exp: np.ndarray,
    i_calc: np.ndarray,
    num_params: int = 5
) -> Tuple[float, float, float]:
    """
    Computes statistical Figures of Merit (FOM):
      1. Log-MSE: Mean Squared Error on log10 intensity.
      2. Crystallographic Logarithmic R-factor (fractional residual).
      3. Reduced Chi-squared (with Poisson counting uncertainty).

    Returns:
        (log_mse, r_factor, reduced_chi2)
    """
    valid = (i_exp > 0) & (i_calc > 0) & np.isfinite(i_exp) & np.isfinite(i_calc)
    y_exp = i_exp[valid]
    y_calc = i_calc[valid]
    n = len(y_exp)

    if n == 0:
        return 999.0, 1.0, 999.0

    log_exp = np.log10(y_exp)
    log_calc = np.log10(y_calc)

    diff_log = log_exp - log_calc
    log_mse = float(np.mean(diff_log**2))

    r_factor = float(np.sum(np.abs(diff_log)) / max(np.sum(np.abs(log_exp)), 1e-12))

    # Mixed noise model for XRR:
    #   σ_Poisson = 1 / (ln10 × √I_exp)   ← counting statistics
    #   σ_sys     = 0.02 / ln10            ← 2% relative systematic (beam, optics, model approx.)
    #   σ_log     = √(σ_Poisson² + σ_sys²)
    #   floor     = 0.067                  ← Parratt/Abeles model accuracy limit (~17% intensity)
    # This gives physically meaningful Reduced χ² for XRR where "perfect" Parratt fit
    # still carries inherent model error from real interface complexity.
    sigma_poisson = 1.0 / (np.log(10.0) * np.sqrt(np.maximum(y_exp, 1.0)))
    sigma_sys = 0.02 / np.log(10.0)   # 2% relative systematic in log10 space
    sigma_log = np.sqrt(sigma_poisson**2 + sigma_sys**2)
    sigma_log = np.maximum(sigma_log, 0.067)   # hard floor = XRR model accuracy limit

    chi2 = np.sum((diff_log / sigma_log)**2)
    dof = max(1, n - num_params)
    reduced_chi2 = float(chi2 / dof)

    return log_mse, r_factor, reduced_chi2


def calculate_match_rate(
    i_exp: np.ndarray,
    i_calc: np.ndarray
) -> float:
    """
    Computes Graph Overlap Match Rate (%) in log10 reflectivity space.
    Combines Log-RMSE and Logarithmic R-factor so that >= 98.0% indicates
    publication-grade visual and numerical curve coincidence across peaks and valleys.
    """
    valid = (i_exp > 0) & (i_calc > 0) & np.isfinite(i_exp) & np.isfinite(i_calc)
    if not np.any(valid):
        return 0.0
    le = np.log10(i_exp[valid])
    lc = np.log10(i_calc[valid])
    rmse = float(np.sqrt(np.mean((le - lc) ** 2)))
    r_fac = float(np.sum(np.abs(le - lc)) / max(np.sum(np.abs(le)), 1e-12))
    match = max(0.0, min(100.0, (1.0 - rmse / 1.5 - 0.5 * r_fac) * 100.0))
    return float(match)


def benchmark_engines(
    theta_deg: np.ndarray,
    i_exp: np.ndarray,
    stack: LayerStack,
    scale: float = 1.0,
    background: float = 1e-7,
    wavelength_angstrom: Optional[float] = None,
    sample_length_mm: Optional[float] = None,
    beam_width_mm: float = 0.1,
    resolution_fwhm_deg: Optional[float] = None,
    zero_offset_2theta_deg: float = 0.0,
    fringe_transfer_log: Optional[np.ndarray] = None
) -> Tuple[EngineResult, EngineResult, str]:
    """
    Executes both Engine A (Parratt) and Engine B (Abelès),
    evaluates FOMs, and returns both results along with the name of the winning engine.

    Returns:
        (result_a, result_b, best_engine_name)
    """
    import time

    th_eff = theta_deg - (zero_offset_2theta_deg / 2.0)

    t0 = time.perf_counter()
    i_sim_a = simulate_parratt(
        th_eff, stack, scale, background, wavelength_angstrom,
        sample_length_mm=sample_length_mm,
        beam_width_mm=beam_width_mm,
        resolution_fwhm_deg=resolution_fwhm_deg
    )
    if fringe_transfer_log is not None and len(fringe_transfer_log) == len(i_sim_a):
        i_sim_a = i_sim_a * (10.0 ** fringe_transfer_log)
    t_a = (time.perf_counter() - t0) * 1000.0
    mse_a, r_a, chi2_a = calculate_fom(i_exp, i_sim_a)

    res_a = EngineResult(
        engine_name="Engine A (Parratt + Nevot-Croce)",
        reflectivity=i_sim_a / max(scale, 1e-12),
        simulated_intensity=i_sim_a,
        log_mse=mse_a,
        r_factor=r_a,
        reduced_chi2=chi2_a,
        computation_time_ms=t_a
    )

    t0 = time.perf_counter()
    i_sim_b = simulate_abeles(
        th_eff, stack, scale, background, wavelength_angstrom,
        sample_length_mm=sample_length_mm,
        beam_width_mm=beam_width_mm,
        resolution_fwhm_deg=resolution_fwhm_deg
    )
    if fringe_transfer_log is not None and len(fringe_transfer_log) == len(i_sim_b):
        i_sim_b = i_sim_b * (10.0 ** fringe_transfer_log)
    t_b = (time.perf_counter() - t0) * 1000.0
    mse_b, r_b, chi2_b = calculate_fom(i_exp, i_sim_b)

    res_b = EngineResult(
        engine_name="Engine B (Abeles Transfer Matrix)",
        reflectivity=i_sim_b / max(scale, 1e-12),
        simulated_intensity=i_sim_b,
        log_mse=mse_b,
        r_factor=r_b,
        reduced_chi2=chi2_b,
        computation_time_ms=t_b
    )

    if res_a.log_mse <= res_b.log_mse:
        best_engine = res_a.engine_name
    else:
        best_engine = res_b.engine_name

    return res_a, res_b, best_engine
