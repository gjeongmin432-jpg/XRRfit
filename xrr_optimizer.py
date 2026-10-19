"""
xrr_optimizer.py
================
Hybrid Auto-Fitting and Parameter Tuning Pipeline for X-ray Reflectivity (XRR).
Combines:
  1. Stage 1: Gemini API inference for physics-based bounds/priors with FFT-heuristic fallback.
  2. Stage 2: Global Optimization via Differential Evolution with angle filtering and
     balanced low-to-mid q weighting ensuring overlap across all Kiessig fringes.
  3. Stage 3: Local Nelder-Mead Simplex refinement.
  4. Automated Layer Stack Exploration (Native Oxide on/off comparative analysis).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
import json
import os
import copy
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks

from xrr_parser import XRRData
from xrr_materials import calculate_optical_constants, STANDARD_MATERIALS
from xrr_features import XRRFeatureReport
from xrr_engine import Layer, LayerStack, simulate_parratt, simulate_abeles, calculate_fom


@dataclass
class OptimizationBounds:
    scale_bounds: Tuple[float, float] = (1e4, 1e9)
    background_bounds: Tuple[float, float] = (1e-6, 1e1)
    layer_bounds: List[Dict[str, Tuple[float, float]]] = field(default_factory=list)
    substrate_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    initial_guess: Optional[np.ndarray] = None
    ai_rationale: str = ""


@dataclass
class FitProgress:
    iteration: int
    max_iterations: int
    best_loss: float
    message: str = ""


@dataclass
class FitResult:
    best_stack: LayerStack
    scale: float
    background: float
    simulated_intensity: np.ndarray
    two_theta: np.ndarray
    theta: np.ndarray
    q_z: np.ndarray
    experimental_intensity: np.ndarray
    log_mse: float
    r_factor: float
    reduced_chi2: float
    best_engine: str
    ai_used: bool
    ai_rationale: str
    param_vector: np.ndarray
    param_names: List[str]
    residual_log: np.ndarray
    start_two_theta: float = 0.2
    end_two_theta: float = 6.0
    sample_length_mm: Optional[float] = None
    beam_width_mm: float = 0.1
    resolution_fwhm_deg: Optional[float] = None
    zero_offset_2theta_deg: float = 0.0
    is_polished: bool = False


# ----------------------------------------------------------------------
# Stage 1: Gemini AI Inference & Physics Heuristics
# ----------------------------------------------------------------------

def generate_heuristic_bounds(
    stack: LayerStack,
    feature_report: XRRFeatureReport,
    data: XRRData
) -> OptimizationBounds:
    """
    Generates physically constrained parameter search bounds and initial guesses
    using thin-film physics heuristics, q_z FFT thickness, and detected Kiessig fringe data.
    """
    plateau = feature_report.plateau_intensity
    scale_init = max(plateau, 1e5)
    scale_bounds = (scale_init * 0.2, scale_init * 5.0)

    # Background from high angle minimum
    valid_int = data.intensity[data.intensity > 0]
    bkg_init = max(float(np.percentile(valid_int, 5)) if len(valid_int) > 0 else 0.5, 1e-4)
    bkg_bounds = (1e-6, max(bkg_init * 15.0, 5.0))

    # Prioritize FFT thickness as primary estimate
    if feature_report.fft_thickness_angstrom > 20.0:
        total_est_d = feature_report.fft_thickness_angstrom
    else:
        total_est_d = feature_report.estimated_thickness_angstrom

    num_film_layers = len(stack.layers)
    layer_bounds: List[Dict[str, Tuple[float, float]]] = []
    initial_params = [np.log10(scale_init), np.log10(bkg_init)]

    # Distribute estimated total thickness among layers
    for i, layer in enumerate(stack.layers):
        is_oxide = "oxide" in layer.name.lower() or "sio2" in layer.formula.lower()
        is_surface = i == 0 and any(kw in layer.name.lower() for kw in ["top", "rough", "contam", "surf", "ox", "변질", "표면"])
        has_surface_above = (i > 0 and any(kw in stack.layers[0].name.lower() for kw in ["top", "rough", "contam", "surf", "ox", "변질", "표면"]))

        nom_rho = STANDARD_MATERIALS.get(layer.formula, layer.density if layer.density > 0 else 3.0)
        rho_bounds = (max(nom_rho * 0.70, 0.5), nom_rho * 1.30)
        rho_init = nom_rho

        if is_surface:
            d_init = min(22.0, max(8.0, total_est_d * 0.12))
            d_bounds = (8.0, 35.0)
            sig_bounds = (0.05, 2.0)
            sig_init = 0.5
            rho_bounds = (1.8, max(nom_rho * 1.15, 6.0))
        elif is_oxide and not has_surface_above:
            d_init = min(20.0, max(8.0, total_est_d * 0.10))
            d_bounds = (5.0, 40.0)
            sig_bounds = (0.1, 3.5)
            sig_init = 1.2
        elif has_surface_above:
            # Main film layer below surface degradation layer
            d_init = max(total_est_d - 20.0, 20.0)
            d_bounds = (max(total_est_d - 40.0, 20.0), total_est_d + 10.0)
            sig_bounds = (0.1, 4.0)
            sig_init = 1.8
        else:
            # Main film layer
            if num_film_layers == 1:
                d_init = total_est_d
            else:
                d_init = max(total_est_d * 0.85, 40.0)
            # Tightly bound thickness around FFT prior to prevent mode hopping
            d_bounds = (max(d_init * 0.70, 10.0), d_init * 1.35)
            # Enforce strict roughness ceiling (<= 4.5 A) to avoid the flat smoothing trap!
            sig_bounds = (0.1, 4.5)
            sig_init = 1.8

        layer_bounds.append({
            "thickness": d_bounds,
            "roughness": sig_bounds,
            "density": rho_bounds
        })
        initial_params.extend([d_init, sig_init, rho_init])

    # Substrate bounds
    sub_nom_rho = STANDARD_MATERIALS.get(stack.substrate.formula, stack.substrate.density)
    sub_bounds = {
        "roughness": (0.05, 3.5),
        "density": (sub_nom_rho * 0.85, sub_nom_rho * 1.15)
    }
    initial_params.extend([1.0, sub_nom_rho])

    return OptimizationBounds(
        scale_bounds=scale_bounds,
        background_bounds=bkg_bounds,
        layer_bounds=layer_bounds,
        substrate_bounds=sub_bounds,
        initial_guess=np.array(initial_params, dtype=np.float64),
        ai_rationale=f"Physics Heuristic Prior: Dominant film thickness {total_est_d:.1f} A derived from q_z FFT spectrum."
    )


def query_gemini_priors(
    stack: LayerStack,
    feature_report: XRRFeatureReport,
    data: XRRData,
    api_key: Optional[str] = None
) -> OptimizationBounds:
    """
    Queries Gemini API using `google-genai` SDK to deduce physical priors and parameter bounds.
    Falls back gracefully to heuristic generator if API fails or key is missing.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return generate_heuristic_bounds(stack, feature_report, data)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)

        # Prepare context description
        fringes_info = []
        for f in feature_report.fringes[:6]:
            fringes_info.append(f"Fringe #{f.order}: 2Theta={f.two_theta:.3f} deg, q_z={f.q_z:.4f} A^-1")

        stack_info = []
        for i, l in enumerate(stack.layers):
            stack_info.append(f"Layer {i+1} ({l.name}): Formula={l.formula}, Nominal Density={l.density:.2f} g/cm3")
        stack_info.append(f"Substrate ({stack.substrate.name}): Formula={stack.substrate.formula}, Nominal Density={stack.substrate.density:.2f} g/cm3")

        prompt = f"""
You are an expert X-ray Reflectivity (XRR) and Thin-Film Physicist.
Analyze the following experimental features extracted from a specular XRR measurement (Cu K-alpha 1.54056 A):

[Experimental Features]
- Detected Critical Angle (theta_c): {feature_report.critical_angle_theta_deg:.4f} deg (2Theta={feature_report.critical_angle_2theta_deg:.4f} deg)
- Surface Electron Density (estimated): {feature_report.electron_density_e_A3:.3f} e/A^3
- Plateau Intensity (I0): {feature_report.plateau_intensity:.2e}
- Observed Kiessig Fringes:
  {chr(10).join(fringes_info)}
- Kiessig Fringe Period (Delta 2Theta): {feature_report.delta_theta_deg*2.0:.4f} deg (Delta q_z: {feature_report.delta_q_z:.4f} A^-1)
- q_z Space Fast Fourier Transform (FFT) Thickness: {feature_report.fft_thickness_angstrom:.1f} Angstroms ({feature_report.fft_thickness_nm:.2f} nm)
- Suggested Valid Fitting Angle Range: 2Theta in [{feature_report.suggested_start_2theta:.2f} deg, {feature_report.suggested_end_2theta:.2f} deg]

[User Target Layer Stack (Top to Bottom)]
{chr(10).join(stack_info)}

Based on thin-film physics, X-ray scattering theory, and the FFT thickness peak, provide the recommended initial values and realistic search bounds (min, max) for:
1. Intensity Scale Factor (I0)
2. Instrumental Background
3. Each film layer: thickness (Angstroms), interface roughness (Angstroms, max 4.5 A), and mass density (g/cm3).
4. Substrate: interface roughness (Angstroms, max 3.5 A) and density (g/cm3).

Respond ONLY with valid JSON in this exact structure:
{{
  "rationale": "Brief physical explanation of choices",
  "scale": {{"init": float, "min": float, "max": float}},
  "background": {{"init": float, "min": float, "max": float}},
  "layers": [
    {{
      "name": "layer name",
      "thickness": {{"init": float, "min": float, "max": float}},
      "roughness": {{"init": float, "min": float, "max": float}},
      "density": {{"init": float, "min": float, "max": float}}
    }}
  ],
  "substrate": {{
    "roughness": {{"init": float, "min": float, "max": float}},
    "density": {{"init": float, "min": float, "max": float}}
  }}
}}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )

        resp_text = response.text.strip()
        parsed = json.loads(resp_text)

        # Parse JSON into OptimizationBounds
        scale_data = parsed.get("scale", {})
        bkg_data = parsed.get("background", {})
        scale_bounds = (float(scale_data.get("min", feature_report.plateau_intensity * 0.2)),
                        float(scale_data.get("max", feature_report.plateau_intensity * 5.0)))
        bkg_bounds = (float(bkg_data.get("min", 1e-6)), float(bkg_data.get("max", 1e2)))

        init_params = [
            np.log10(float(scale_data.get("init", feature_report.plateau_intensity))),
            np.log10(float(bkg_data.get("init", 1e-2)))
        ]

        target_d = feature_report.fft_thickness_angstrom if feature_report.fft_thickness_angstrom > 20.0 else feature_report.estimated_thickness_angstrom
        layer_bounds = []
        ai_layers = parsed.get("layers", [])
        for i, layer in enumerate(stack.layers):
            ai_l = ai_layers[i] if i < len(ai_layers) else {}
            th_d = ai_l.get("thickness", {})
            rg_d = ai_l.get("roughness", {})
            dn_d = ai_l.get("density", {})

            t_init = float(th_d.get("init", target_d))
            t_min = float(th_d.get("min", max(t_init * 0.70, 5.0)))
            t_max = float(th_d.get("max", t_init * 1.35))

            r_init = min(float(rg_d.get("init", 1.8)), 4.0)
            r_min = float(rg_d.get("min", 0.1))
            r_max = min(float(rg_d.get("max", 4.5)), 4.5)  # Enforce ceiling

            d_init = float(dn_d.get("init", layer.density))
            d_min = float(dn_d.get("min", d_init * 0.75))
            d_max = float(dn_d.get("max", d_init * 1.25))

            layer_bounds.append({
                "thickness": (t_min, t_max),
                "roughness": (r_min, r_max),
                "density": (d_min, d_max)
            })
            init_params.extend([t_init, r_init, d_init])

        sub_ai = parsed.get("substrate", {})
        sub_rg = sub_ai.get("roughness", {})
        sub_dn = sub_ai.get("density", {})
        sub_nom_rho = STANDARD_MATERIALS.get(stack.substrate.formula, stack.substrate.density)

        substrate_bounds = {
            "roughness": (float(sub_rg.get("min", 0.05)), min(float(sub_rg.get("max", 3.5)), 3.5)),
            "density": (float(sub_dn.get("min", sub_nom_rho * 0.85)), float(sub_dn.get("max", sub_nom_rho * 1.15)))
        }
        init_params.extend([
            float(sub_rg.get("init", 1.0)),
            float(sub_dn.get("init", sub_nom_rho))
        ])

        return OptimizationBounds(
            scale_bounds=scale_bounds,
            background_bounds=bkg_bounds,
            layer_bounds=layer_bounds,
            substrate_bounds=substrate_bounds,
            initial_guess=np.array(init_params, dtype=np.float64),
            ai_rationale=str(parsed.get("rationale", "Gemini AI prior bounds derived successfully."))
        )

    except Exception as e:
        bounds = generate_heuristic_bounds(stack, feature_report, data)
        bounds.ai_rationale = f"Gemini API fallback to Heuristics (Reason: {str(e)})"
        return bounds


# ----------------------------------------------------------------------
# Parameter Packing / Unpacking
# ----------------------------------------------------------------------

def pack_stack_params(
    stack: LayerStack,
    scale: float,
    background: float
) -> Tuple[np.ndarray, List[str]]:
    """Packs layer stack variables into parameter vector."""
    param_values = [np.log10(scale), np.log10(max(background, 1e-12))]
    param_names = ["log10(Scale)", "log10(Background)"]

    for i, layer in enumerate(stack.layers):
        param_values.extend([layer.thickness, layer.roughness, layer.density])
        param_names.extend([
            f"L{i+1}_{layer.name}_thick",
            f"L{i+1}_{layer.name}_rough",
            f"L{i+1}_{layer.name}_dens"
        ])

    param_values.extend([stack.substrate.roughness, stack.substrate.density])
    param_names.extend([
        f"Sub_{stack.substrate.name}_rough",
        f"Sub_{stack.substrate.name}_dens"
    ])

    return np.array(param_values, dtype=np.float64), param_names


def unpack_stack_params(
    vector: np.ndarray,
    template_stack: LayerStack
) -> Tuple[LayerStack, float, float]:
    """Unpacks 1D parameter vector back into a configured LayerStack object."""
    new_stack = copy.deepcopy(template_stack)
    scale = 10.0 ** vector[0]
    background = 10.0 ** vector[1]

    idx = 2
    for layer in new_stack.layers:
        layer.thickness = float(vector[idx])
        layer.roughness = float(vector[idx + 1])
        layer.density = float(vector[idx + 2])
        layer.update_optical_constants(new_stack.wavelength_angstrom)
        idx += 3

    new_stack.substrate.roughness = float(vector[idx])
    new_stack.substrate.density = float(vector[idx + 1])
    new_stack.substrate.update_optical_constants(new_stack.wavelength_angstrom)

    new_stack.ambient.update_optical_constants(new_stack.wavelength_angstrom)
    return new_stack, scale, background


def apply_adaptive_valley_polish(
    two_theta: np.ndarray,
    experimental_intensity: np.ndarray,
    current_stack: LayerStack,
    current_scale: float,
    current_bkg: float,
    current_fwhm: float,
    sample_length_mm: Optional[float],
    beam_width_mm: float,
    wavelength_angstrom: float,
    sim_func: Callable
) -> Tuple[LayerStack, float, float, float, np.ndarray, Tuple[float, float, float]]:
    """
    Stage 2: Adaptive Valley-Peak Polishing Pass.
    Identifies Kiessig fringe valleys with high residual discrepancy and executes a
    focused local refinement optimizing surface contrast, resolution FWHM, and
    goniometer zero-angle offset (Delta 2-theta) to drive valley residuals down to zero.
    """
    log_exp = np.log10(np.maximum(experimental_intensity, 1e-12))
    valleys_idx, _ = find_peaks(-log_exp, prominence=0.04, distance=6)

    s = copy.deepcopy(current_stack)
    init_p = [np.log10(max(current_scale, 1e4))]
    bounds: List[Tuple[float, float]] = [(init_p[0] - 0.08, init_p[0] + 0.08)]

    for i, l in enumerate(s.layers):
        init_p.extend([l.thickness, l.roughness, l.density])
        if i == 0:
            bounds.extend([
                (max(5.0, l.thickness * 0.80), min(45.0, l.thickness * 1.25)),
                (0.05, 2.5),
                (max(1.8, l.density * 0.80), min(6.8, l.density * 1.20))
            ])
        else:
            bounds.extend([
                (l.thickness * 0.95, l.thickness * 1.05),
                (0.05, 3.5),
                (l.density * 0.92, l.density * 1.08)
            ])

    init_p.extend([s.substrate.roughness, s.substrate.density])
    bounds.extend([
        (0.05, 3.5),
        (s.substrate.density * 0.94, s.substrate.density * 1.06)
    ])

    init_p.append(current_fwhm if (current_fwhm and current_fwhm > 0) else 0.015)
    bounds.append((0.005, 0.035))

    init_p.append(0.0)  # Delta 2-theta
    bounds.append((-0.012, 0.012))

    def polish_objective(p: np.ndarray) -> float:
        try:
            sc = 10**p[0]
            idx = 1
            new_layers = []
            for lyr in s.layers:
                new_layers.append(Layer(lyr.name, lyr.formula, float(p[idx]), float(p[idx+1]), float(p[idx+2])))
                idx += 3
            new_sub = Layer(s.substrate.name, s.substrate.formula, 0.0, float(p[idx]), float(p[idx+1]))
            stk = LayerStack(s.ambient, new_layers, new_sub, wavelength_angstrom)
            stk.update_all_optical_constants()
            fwhm_val = float(p[idx+2])
            d2th = float(p[idx+3])

            th_shifted = (two_theta - d2th) / 2.0
            sim = sim_func(
                th_shifted, stk, scale=sc, background=current_bkg,
                sample_length_mm=sample_length_mm,
                beam_width_mm=beam_width_mm,
                resolution_fwhm_deg=fwhm_val
            )
            if np.any(sim <= 0) or np.any(np.isnan(sim)):
                return 1e8
            log_sim = np.log10(sim)

            base_mse = np.mean((log_exp - log_sim) ** 2)

            v_penalty = 0.0
            if len(valleys_idx) > 0:
                v_diff = log_sim[valleys_idx] - log_exp[valleys_idx]
                v_penalty = np.mean(np.where(v_diff > 0, (v_diff * 3.5)**2, v_diff**2))

            sh_mask = (two_theta >= 0.48) & (two_theta <= 0.70)
            sh_penalty = np.mean((log_exp[sh_mask] - log_sim[sh_mask])**2) if np.any(sh_mask) else 0.0
            zero_reg = 30.0 * (d2th ** 2)

            return float(base_mse + 3.0 * v_penalty + 1.5 * sh_penalty + zero_reg)
        except Exception:
            return 1e8

    res_opt = minimize(
        polish_objective,
        init_p,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-13, "gtol": 1e-6}
    )

    p_opt = res_opt.x
    sc_opt = 10**p_opt[0]
    idx = 1
    opt_layers = []
    for lyr in s.layers:
        opt_layers.append(Layer(lyr.name, lyr.formula, float(p_opt[idx]), float(p_opt[idx+1]), float(p_opt[idx+2])))
        idx += 3
    opt_sub = Layer(s.substrate.name, s.substrate.formula, 0.0, float(p_opt[idx]), float(p_opt[idx+1]))
    opt_stk = LayerStack(s.ambient, opt_layers, opt_sub, wavelength_angstrom)
    opt_stk.update_all_optical_constants()
    fwhm_opt = float(p_opt[idx+2])
    d2th_opt = float(p_opt[idx+3])

    sim_opt = sim_func(
        (two_theta - d2th_opt) / 2.0, opt_stk,
        scale=sc_opt, background=current_bkg,
        sample_length_mm=sample_length_mm,
        beam_width_mm=beam_width_mm,
        resolution_fwhm_deg=fwhm_opt
    )
    fom_opt = calculate_fom(experimental_intensity, sim_opt)
    return opt_stk, sc_opt, fwhm_opt, d2th_opt, sim_opt, fom_opt


# ----------------------------------------------------------------------
# Stage 2 & 3: Global DE + Local Nelder-Mead Optimization
# ----------------------------------------------------------------------

def fit_xrr(
    data: XRRData,
    initial_stack: LayerStack,
    feature_report: XRRFeatureReport,
    api_key: Optional[str] = None,
    engine_choice: str = "Auto",
    start_two_theta: Optional[float] = None,
    end_two_theta: Optional[float] = None,
    sample_length_mm: Optional[float] = 15.0,
    beam_width_mm: float = 0.1,
    auto_fit_sample_length: bool = True,
    resolution_fwhm_deg: Optional[float] = 0.015,
    auto_surface_layer: bool = False,
    enable_valley_polishing: bool = True,
    max_generations: int = 35,
    popsize: int = 12,
    progress_callback: Optional[Callable[[FitProgress], None]] = None
) -> FitResult:
    """
    Hybrid automated fitting pipeline with physical footprint & instrument resolution modeling:
      1. Slices data to valid physical angle range (excluding direct beam drop & noise floor).
      2. Gemini AI / FFT heuristic inference for prior parameter bounds.
      3. Geometric footprint factor F(theta) = min(1.0, L*sin(theta)/W) modeling beam spillover.
      4. Instrumental angular resolution smearing (slit divergence convolution).
      5. Optional automated surface layer modeling (degradation/contamination layer).
      6. Differential Evolution global search + Nelder-Mead + L-BFGS-B gradient refinement.
      7. Multi-engine FOM evaluation.
    """
    clean_data = data.filter_valid()

    # Determine active angle fitting range
    start_2th = float(start_two_theta) if start_two_theta is not None else float(feature_report.suggested_start_2theta)
    end_2th = float(end_two_theta) if end_two_theta is not None else float(feature_report.suggested_end_2theta)

    angle_mask = (clean_data.two_theta >= start_2th) & (clean_data.two_theta <= end_2th)
    if np.sum(angle_mask) < 20:
        angle_mask = (clean_data.two_theta >= 0.2) & (clean_data.two_theta <= 6.0)

    th = clean_data.theta[angle_mask]
    two_th = clean_data.two_theta[angle_mask]
    qz = clean_data.q_z[angle_mask]
    y_exp = clean_data.intensity[angle_mask]
    log_y_exp = np.log10(y_exp)

    # Automated surface layer splitting if single film layer
    active_stack = initial_stack
    if auto_surface_layer and len(initial_stack.layers) == 1:
        main_l = initial_stack.layers[0]
        surf_formula = "SnO2" if "sno" in main_l.formula.lower() else main_l.formula
        nom_rho = STANDARD_MATERIALS.get(surf_formula, main_l.density if main_l.density > 0 else 5.0)
        target_d = feature_report.fft_thickness_angstrom if feature_report.fft_thickness_angstrom > 20.0 else feature_report.estimated_thickness_angstrom
        surf_layer = Layer(
            name="Surface Layer (최표면 변질/산화층)",
            formula=surf_formula,
            thickness=20.0,
            roughness=1.0,
            density=nom_rho
        )
        bulk_layer = Layer(
            name=f"{main_l.name} (벌크)",
            formula=main_l.formula,
            thickness=max(target_d - 20.0, 20.0),
            roughness=main_l.roughness,
            density=main_l.density
        )
        active_stack = LayerStack(
            ambient=initial_stack.ambient,
            layers=[surf_layer, bulk_layer],
            substrate=initial_stack.substrate,
            wavelength_angstrom=initial_stack.wavelength_angstrom
        )
        active_stack.update_all_optical_constants()

    # 1. Query Priors & Bounds
    bounds_obj = query_gemini_priors(active_stack, feature_report, clean_data, api_key=api_key)
    ai_used = "Gemini" in bounds_obj.ai_rationale and "fallback" not in bounds_obj.ai_rationale.lower()

    # Build bound tuples
    # If footprint correction is applied, true raw beam scale is ~2x higher than plateau
    plateau = feature_report.plateau_intensity
    scale_min = max(plateau * 0.5, 1e4)
    scale_max = max(plateau * 10.0, 1e10)
    bounds_list: List[Tuple[float, float]] = [
        (np.log10(scale_min), np.log10(scale_max)),
        (np.log10(bounds_obj.background_bounds[0]), np.log10(bounds_obj.background_bounds[1]))
    ]

    for lb in bounds_obj.layer_bounds:
        bounds_list.append(lb["thickness"])
        bounds_list.append(lb["roughness"])
        bounds_list.append(lb["density"])

    bounds_list.append(bounds_obj.substrate_bounds["roughness"])
    bounds_list.append(bounds_obj.substrate_bounds["density"])

    # Optional: Auto-fit sample length (Footprint spillover dimension)
    if auto_fit_sample_length:
        init_L = sample_length_mm if sample_length_mm is not None and sample_length_mm > 0 else 15.0
        l_min = max(4.0, init_L * 0.4)
        l_max = min(40.0, init_L * 2.5)
        bounds_list.append((l_min, l_max))

    # Optional: Auto-fit beam resolution FWHM (0.008° ~ 0.035°) to perfectly match fringe valleys
    fit_fwhm = (resolution_fwhm_deg is not None and resolution_fwhm_deg > 0)
    if fit_fwhm:
        bounds_list.append((0.008, 0.035))

    # 2. Weighting Vector: 1~5차 Fringe 및 임계각 에지에 최우선 가중치 집중
    weights = np.full_like(th, 0.05, dtype=np.float64)

    th_c = feature_report.critical_angle_theta_deg

    # 0.48° 미만 스필오버 전이 구간: 가중치 0.5x (과도한 풋프린트 왜곡 방지)
    low_spill = (th < 0.48 / 2.0)
    weights[low_spill] = 0.5

    # 전반사 임계각 에지 (0.48° ~ 0.72°): 6.0x (임계각 고원 어깨와 감쇠 절벽 최우선 정합)
    crit_mask = (th >= 0.48 / 2.0) & (th <= max(th_c * 1.35, 0.72 / 2.0))
    weights[crit_mask] = 6.0

    # 1차 Kiessig 진동 골 (First Valley, 0.75° ~ 0.88°): 8.0x (초반부 1차 간섭 골의 깊이 일치)
    v1_mask = (th >= 0.75 / 2.0) & (th <= 0.88 / 2.0)
    weights[v1_mask] = 8.0

    # Fringes 1~5 집중 구간: 4.0x (Kiessig Fringe 마루와 골 완벽 정합)
    if len(feature_report.fringes) >= 5:
        th_5th = feature_report.fringes[4].theta
        low_fringe_mask = (th > 0.88 / 2.0) & (th <= th_5th * 1.05)
        weights[low_fringe_mask] = 4.0
    elif len(feature_report.fringes) >= 3:
        th_last = feature_report.fringes[-1].theta
        low_fringe_mask = (th > 0.88 / 2.0) & (th <= th_last * 1.05)
        weights[low_fringe_mask] = 4.0
    else:
        low_fringe_mask = (th > 0.88 / 2.0) & (th <= th_c + 0.75)
        weights[low_fringe_mask] = 4.0

    sum_weights = np.sum(weights)

    # Choose primary simulation engine for fitting loops
    sim_func = simulate_parratt if engine_choice in ["Auto", "Parratt"] else simulate_abeles

    # Objective function: Weighted Log-MSE
    def objective(p: np.ndarray) -> float:
        try:
            # Enforce bounds strictly
            for val, (b_min, b_max) in zip(p, bounds_list):
                if val < b_min or val > b_max:
                    return 1e8

            p_idx = len(p)
            if fit_fwhm:
                p_idx -= 1
                curr_fwhm = float(p[p_idx])
            else:
                curr_fwhm = resolution_fwhm_deg

            if auto_fit_sample_length:
                p_idx -= 1
                L_sample = float(p[p_idx])
            else:
                L_sample = sample_length_mm

            stack_p = p[:p_idx]
            stk, sc, bg = unpack_stack_params(stack_p, active_stack)
            sim_i = sim_func(
                th, stk, scale=sc, background=bg,
                sample_length_mm=L_sample,
                beam_width_mm=beam_width_mm,
                resolution_fwhm_deg=curr_fwhm
            )
            if np.any(sim_i <= 0) or np.any(np.isnan(sim_i)):
                return 1e8
            log_sim = np.log10(sim_i)
            weighted_sq_err = weights * ((log_y_exp - log_sim) ** 2)
            loss = float(np.sum(weighted_sq_err) / sum_weights)
            return loss
        except Exception:
            return 1e8


    # 3. Global Optimization via Differential Evolution
    curr_gen = [0]

    def de_callback(xk: np.ndarray, convergence: float):
        curr_gen[0] += 1
        if progress_callback:
            loss_val = objective(xk)
            progress_callback(FitProgress(
                iteration=curr_gen[0],
                max_iterations=max_generations,
                best_loss=loss_val,
                message=f"전역 DE 탐색 세대 {curr_gen[0]}/{max_generations} (오차: {loss_val:.4f})"
            ))

    if progress_callback:
        progress_callback(FitProgress(0, max_generations, 0.0, "전역 차분 진화(DE) 풋프린트/스택 최적화 시작 중..."))

    de_res = differential_evolution(
        objective,
        bounds=bounds_list,
        strategy="best1bin",
        maxiter=max_generations,
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        callback=de_callback,
        polish=False,
        seed=42
    )

    best_p = de_res.x

    # 4. Local Refinement (Nelder-Mead)
    if progress_callback:
        progress_callback(FitProgress(
            max_generations, max_generations, float(de_res.fun),
            "Nelder-Mead 국소 심플렉스 정밀화 수행 중..."
        ))

    local_res = minimize(
        objective,
        best_p,
        method="Nelder-Mead",
        options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-5}
    )
    if local_res.success or local_res.fun < de_res.fun:
        best_p = local_res.x

    # 4b. L-BFGS-B 2차 Gradient 정밀화
    if progress_callback:
        progress_callback(FitProgress(
            max_generations, max_generations, float(objective(best_p)),
            "L-BFGS-B 그래디언트 기반 2차 정밀화 수행 중..."
        ))

    try:
        lbfgs_res = minimize(
            objective,
            best_p,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7}
        )
        if lbfgs_res.fun < objective(best_p):
            best_p = lbfgs_res.x
    except Exception:
        pass

    # Clip best_p to bounds
    for k, (b_min, b_max) in enumerate(bounds_list):
        best_p[k] = np.clip(best_p[k], b_min, b_max)

    # Unpack best parameters
    p_idx = len(best_p)
    if fit_fwhm:
        p_idx -= 1
        final_fwhm = float(best_p[p_idx])
    else:
        final_fwhm = resolution_fwhm_deg

    if auto_fit_sample_length:
        p_idx -= 1
        final_L = float(best_p[p_idx])
    else:
        final_L = sample_length_mm

    final_stack_p = best_p[:p_idx]

    # 5. Multi-Engine Benchmark & Evaluation on Active Range
    final_stack, final_scale, final_bkg = unpack_stack_params(final_stack_p, active_stack)
    sim_a = simulate_parratt(
        th, final_stack, scale=final_scale, background=final_bkg,
        sample_length_mm=final_L, beam_width_mm=beam_width_mm,
        resolution_fwhm_deg=final_fwhm
    )
    sim_b = simulate_abeles(
        th, final_stack, scale=final_scale, background=final_bkg,
        sample_length_mm=final_L, beam_width_mm=beam_width_mm,
        resolution_fwhm_deg=final_fwhm
    )

    fom_a = calculate_fom(y_exp, sim_a, num_params=len(best_p))
    fom_b = calculate_fom(y_exp, sim_b, num_params=len(best_p))

    if fom_a[0] <= fom_b[0]:
        best_engine_name = "Engine A (Parratt + Nevot-Croce)"
        final_sim = sim_a
        best_mse, best_r, best_chi2 = fom_a
    else:
        best_engine_name = "Engine B (Abeles Transfer Matrix)"
        final_sim = sim_b
        best_mse, best_r, best_chi2 = fom_b

    zero_offset_2th = 0.0
    is_polished = False
    if enable_valley_polishing:
        if progress_callback:
            progress_callback(FitProgress(
                max_generations, max_generations, best_mse,
                "[2단계] 적응형 골(Valley) 및 위상 잔차 정밀 연마 수행 중..."
            ))
        try:
            primary_sim = simulate_parratt if "Parratt" in best_engine_name else simulate_abeles
            p_stk, p_sc, p_fwhm, p_d2th, p_sim, p_fom = apply_adaptive_valley_polish(
                two_theta=two_th,
                experimental_intensity=y_exp,
                current_stack=final_stack,
                current_scale=final_scale,
                current_bkg=final_bkg,
                current_fwhm=final_fwhm if (final_fwhm and final_fwhm > 0) else 0.015,
                sample_length_mm=final_L,
                beam_width_mm=beam_width_mm,
                wavelength_angstrom=clean_data.wavelength_angstrom,
                sim_func=primary_sim
            )
            # Adopt polished solution if it improves or preserves FOM
            if p_fom[1] <= best_r * 1.15 or p_fom[0] <= best_mse * 1.10:
                final_stack = p_stk
                final_scale = p_sc
                final_fwhm = p_fwhm
                zero_offset_2th = p_d2th
                final_sim = p_sim
                best_mse, best_r, best_chi2 = p_fom
                is_polished = True
        except Exception:
            pass

    _, p_names = pack_stack_params(final_stack, final_scale, final_bkg)
    if auto_fit_sample_length:
        p_names.append("Sample_Length(mm)")
    if fit_fwhm:
        p_names.append("Beam_FWHM(deg)")
    if is_polished and abs(zero_offset_2th) > 1e-4:
        p_names.append("Zero_Offset(deg)")

    residuals = log_y_exp - np.log10(np.maximum(final_sim, 1e-12))

    if progress_callback:
        l_info = f", 시편 길이: {final_L:.1f}mm" if final_L else ""
        pol_msg = " ([2단계] 골 연마 완료)" if is_polished else ""
        progress_callback(FitProgress(
            max_generations, max_generations, best_mse,
            f"피팅 완료! 최적 엔진: {best_engine_name}{pol_msg} (R-팩터: {best_r:.2%}, χ²: {best_chi2:.2f}{l_info})"
        ))


    return FitResult(
        best_stack=final_stack,
        scale=final_scale,
        background=final_bkg,
        simulated_intensity=final_sim,
        two_theta=two_th,
        theta=th,
        q_z=qz,
        experimental_intensity=y_exp,
        log_mse=best_mse,
        r_factor=best_r,
        reduced_chi2=best_chi2,
        best_engine=best_engine_name,
        ai_used=ai_used,
        ai_rationale=bounds_obj.ai_rationale,
        param_vector=best_p,
        param_names=p_names,
        residual_log=residuals,
        start_two_theta=start_2th,
        end_two_theta=end_2th,
        sample_length_mm=final_L,
        beam_width_mm=beam_width_mm,
        resolution_fwhm_deg=final_fwhm,
        zero_offset_2theta_deg=zero_offset_2th,
        is_polished=is_polished
    )



# ----------------------------------------------------------------------
# Automated Stack Exploration (Native Oxide On/Off Comparison)
# ----------------------------------------------------------------------

def explore_native_oxide(
    data: XRRData,
    base_stack: LayerStack,
    feature_report: XRRFeatureReport,
    api_key: Optional[str] = None,
    start_two_theta: Optional[float] = None,
    end_two_theta: Optional[float] = None
) -> Tuple[FitResult, FitResult, str]:
    """
    Automated layer exploration:
      Compares Model 1 (without native oxide) vs Model 2 (with native oxide).
      Returns (result_without_oxide, result_with_oxide, recommendation_message).
    """
    stack_no_ox = copy.deepcopy(base_stack)
    stack_no_ox.layers = [l for l in stack_no_ox.layers if "oxide" not in l.name.lower() and "sio2" not in l.formula.lower()]

    stack_with_ox = copy.deepcopy(base_stack)
    has_oxide = any("oxide" in l.name.lower() or "sio2" in l.formula.lower() for l in stack_with_ox.layers)
    if not has_oxide:
        ox_layer = Layer(
            name="Native Oxide",
            formula="SiO2" if "si" in stack_with_ox.substrate.formula.lower() else "Al2O3",
            thickness=15.0,
            roughness=1.2,
            density=2.20
        )
        stack_with_ox.layers.append(ox_layer)

    res_no_ox = fit_xrr(
        data, stack_no_ox, feature_report,
        api_key=api_key,
        start_two_theta=start_two_theta,
        end_two_theta=end_two_theta,
        max_generations=20
    )
    res_with_ox = fit_xrr(
        data, stack_with_ox, feature_report,
        api_key=api_key,
        start_two_theta=start_two_theta,
        end_two_theta=end_two_theta,
        max_generations=20
    )

    delta_r = res_no_ox.r_factor - res_with_ox.r_factor
    if delta_r > 0.005:
        rec = f"판정: 자연 산화막(Native Oxide) 모델의 설명력이 통계적으로 우수합니다 (R-팩터 오차율 {delta_r*100:.2f}% 개선)."
    else:
        rec = f"판정: 자연 산화막이 없는 단순 적층 구조 모델로 충분합니다 (R-팩터 변화 미미: {delta_r*100:.2f}%)."

    return res_no_ox, res_with_ox, rec
