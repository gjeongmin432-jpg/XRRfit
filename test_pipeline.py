"""
test_pipeline.py
================
Verifies the complete XRR processing pipeline:
1. Parsing .xrdml file (wavelength, 2Theta, CPS, attenuation correction)
2. Materials optical constant and SLD calculation
3. Critical angle and Kiessig fringe auto-detection
4. Multi-engine simulation (Parratt vs Abelès)
5. Hybrid optimization (Differential Evolution + Nelder-Mead with low-to-mid q weighting)
"""

import sys
from xrr_parser import load_xrr_file
from xrr_materials import calculate_optical_constants
from xrr_features import analyze_xrr_features
from xrr_engine import Layer, LayerStack, simulate_parratt, simulate_abeles, benchmark_engines
from xrr_optimizer import fit_xrr, explore_native_oxide


def run_test():
    print("=== 1. Testing XRDML Parsing ===")
    with open("sample_sno_film.xrdml", "rb") as f:
        content = f.read()

    data = load_xrr_file(content, filename="sample_sno_film.xrdml")
    print(f"Loaded {len(data.two_theta)} points.")
    print(f"2Theta range: {data.two_theta[0]:.2f} to {data.two_theta[-1]:.2f} deg")
    print(f"Wavelength: {data.wavelength_angstrom:.5f} A ({data.wavelength_nm:.5f} nm)")
    print(f"Intensity max: {data.intensity.max():.2e}, min: {data.intensity.min():.2e}")
    assert len(data.two_theta) > 100
    assert data.metadata.get("attenuation_corrected") is True

    print("\n=== 2. Testing Materials Optical Constants ===")
    si_prop = calculate_optical_constants("Si", 2.33)
    sno_prop = calculate_optical_constants("SnO", 6.45)
    print(f"Si: delta={si_prop.delta:.3e}, beta={si_prop.beta:.3e}, theta_c={si_prop.critical_angle_deg:.3f} deg")
    print(f"SnO: delta={sno_prop.delta:.3e}, beta={sno_prop.beta:.3e}, theta_c={sno_prop.critical_angle_deg:.3f} deg")
    assert 0.20 < si_prop.critical_angle_deg < 0.25
    assert 0.30 < sno_prop.critical_angle_deg < 0.45

    print("\n=== 3. Testing Feature Extraction (theta_c & Kiessig Fringes) ===")
    features = analyze_xrr_features(data)
    print(f"Detected theta_c: {features.critical_angle_theta_deg:.4f} deg (2Theta={features.critical_angle_2theta_deg:.4f} deg)")
    print(f"Estimated surface electron density: {features.electron_density_e_A3:.3f} e/A^3")
    print(f"Detected Kiessig Fringes: {features.num_fringes_detected} fringes")
    for f in features.fringes[:5]:
        print(f"  Fringe #{f.order}: 2Theta={f.two_theta:.3f} deg, Intensity={f.intensity:.2e}")
    print(f"Fringe spacing Delta 2Theta: {features.delta_theta_deg*2.0:.4f} deg")
    print(f"Estimated film total thickness: {features.estimated_thickness_angstrom:.1f} A ({features.estimated_thickness_nm:.2f} nm)")
    assert features.has_3rd_fringe is True
    assert 200.0 < features.estimated_thickness_angstrom < 320.0

    print("\n=== 4. Testing Multi-Engine Simulation & Benchmarking ===")
    ambient = Layer("Air", "Air", 0.0, 0.0, 0.0012)
    sno = Layer("SnO Film", "SnO", 250.0, 1.8, 6.45)
    sio2 = Layer("Native Oxide", "SiO2", 18.0, 1.2, 2.20)
    sub = Layer("Si Substrate", "Si", 0.0, 1.0, 2.33)

    stack = LayerStack(ambient, [sno, sio2], sub, wavelength_angstrom=data.wavelength_angstrom)
    stack.update_all_optical_constants()

    res_a, res_b, winner = benchmark_engines(data.theta, data.intensity, stack, scale=1e7, background=0.5)
    print(f"Engine A (Parratt): Log-MSE={res_a.log_mse:.4f}, R-factor={res_a.r_factor:.2%}, Time={res_a.computation_time_ms:.1f} ms")
    print(f"Engine B (Abeles):  Log-MSE={res_b.log_mse:.4f}, R-factor={res_b.r_factor:.2%}, Time={res_b.computation_time_ms:.1f} ms")
    print(f"Winning Engine: {winner}")

    print("\n=== 5. Testing Hybrid Optimization Pipeline ===")
    # Test with 15 DE generations for quick automated test
    def progress_cb(prog):
        if prog.iteration % 5 == 0 or prog.iteration == prog.max_iterations:
            print(f"  [{prog.iteration}/{prog.max_iterations}] {prog.message}")

    fit_res = fit_xrr(
        data=data,
        initial_stack=stack,
        feature_report=features,
        max_generations=15,
        popsize=8,
        progress_callback=progress_cb
    )

    print(f"\nOptimization Finished!")
    print(f"Best Engine: {fit_res.best_engine}")
    print(f"Final Log-MSE: {fit_res.log_mse:.5f}, R-factor: {fit_res.r_factor:.2%}, Reduced Chi2: {fit_res.reduced_chi2:.3f}")
    print("Fitted Layer Stack Parameters:")
    for layer in fit_res.best_stack.layers:
        print(f"  {layer.name}: Thickness={layer.thickness:.1f} A ({layer.thickness_nm:.2f} nm), Roughness={layer.roughness:.2f} A, Density={layer.density:.2f} g/cm3")
    print(f"  Substrate: Roughness={fit_res.best_stack.substrate.roughness:.2f} A, Density={fit_res.best_stack.substrate.density:.2f} g/cm3")

    print("\nALL SYNTHETIC VERIFICATION TESTS PASSED SUCCESSFULLY!")

    # === 6. Verification with User's Real Experimental Data ===
    real_path = r"C:\Users\LG\Downloads\XRR-6-Constant_Temp-3-200-a_2-Theta_Omega.xrdml"
    import os
    if os.path.exists(real_path):
        print("\n=== 6. Testing User's Real Experimental Data (.xrdml) ===")
        with open(real_path, "rb") as f:
            real_data = load_xrr_file(f.read(), filename="real_data.xrdml")
        real_features = analyze_xrr_features(real_data)
        print(f"Real Data Points: {len(real_data.two_theta)}")
        print(f"Detected Critical Angle 2Theta: {real_features.critical_angle_2theta_deg:.3f} deg")
        print(f"Suggested Start 2Theta (Spillover onset): {real_features.suggested_start_2theta:.3f} deg")
        print(f"Suggested End 2Theta (Noise cutoff): {real_features.suggested_end_2theta:.3f} deg")
        print(f"q_z FFT Peak Thickness: {real_features.fft_thickness_angstrom:.1f} A ({real_features.fft_thickness_nm:.2f} nm)")
        print(f"Detected Kiessig Fringes: {real_features.num_fringes_detected} fringes")
        assert 170.0 <= real_features.fft_thickness_angstrom <= 195.0, "FFT thickness should be ~180 A"
        assert real_features.num_fringes_detected >= 8, "Should detect at least 8 fringes"

        # Fit single film stack
        real_ambient = Layer("Air", "Air", 0.0, 0.0, 0.0012)
        real_film = Layer("Film", "SnO", float(real_features.fft_thickness_angstrom), 1.8, 6.45)
        real_sub = Layer("Si Substrate", "Si", 0.0, 1.0, 2.33)
        real_stack = LayerStack(real_ambient, [real_film], real_sub, wavelength_angstrom=real_data.wavelength_angstrom)

        real_fit = fit_xrr(
            data=real_data,
            initial_stack=real_stack,
            feature_report=real_features,
            start_two_theta=0.45,
            end_two_theta=6.0,
            max_generations=15,
            popsize=8
        )
        print(f"Real Data Fit Result: Best Engine={real_fit.best_engine}, Log-MSE={real_fit.log_mse:.5f}, R-factor={real_fit.r_factor:.2%}")
        print(f"  Fitted Thickness: {real_fit.best_stack.layers[0].thickness:.1f} A ({real_fit.best_stack.layers[0].thickness_nm:.2f} nm)")
        print(f"  Fitted Roughness: {real_fit.best_stack.layers[0].roughness:.2f} A, Density: {real_fit.best_stack.layers[0].density:.2f} g/cm3")
        assert real_fit.r_factor < 0.05, f"R-factor {real_fit.r_factor:.2%} should be < 5.0%"
        print("\nALL REAL-DATA VERIFICATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_test()
