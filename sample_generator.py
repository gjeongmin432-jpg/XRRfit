"""
sample_generator.py
===================
Generates realistic synthetic Malvern Panalytical .xrdml raw data files and CSV data
for testing and demonstration of the automated XRR analysis system.
Includes:
  - Specular reflectivity simulation using exact Parratt physics
  - Realistic instrumental beam attenuator factors at low angles
  - Poisson photon counting statistics and background noise
  - Standard Panalytical XRDML XML format
"""

import os
import numpy as np
from xrr_materials import calculate_optical_constants
from xrr_engine import Layer, LayerStack, simulate_parratt


def generate_synthetic_xrr(
    filename_xrdml: str = "sample_sno_film.xrdml",
    filename_csv: str = "sample_sno_film.csv"
):
    """
    Creates a benchmark thin-film sample:
      Air -> Surface Contamination (1.2 nm) -> SnO Film (25.0 nm) -> SiO2 Native Oxide (1.8 nm) -> Si Substrate
    """
    # Incident 2-Theta from 0.2 deg to 4.5 deg (step 0.01 deg -> 431 points)
    two_theta = np.arange(0.20, 4.51, 0.01)
    theta = two_theta / 2.0
    wavelength_angstrom = 1.540598  # Cu K-alpha1 in Angstroms (0.1540598 nm)

    # 1. Define physical stack (realistic high-quality semiconductor thin film roughness)
    ambient = Layer(name="Air", formula="Air", thickness=0.0, roughness=0.0, density=0.0012)
    top_surf = Layer(name="Surface Layer", formula="C", thickness=12.0, roughness=2.0, density=1.80)
    sno_layer = Layer(name="SnO Film", formula="SnO", thickness=250.0, roughness=1.8, density=6.45)
    native_ox = Layer(name="Native Oxide", formula="SiO2", thickness=18.0, roughness=1.2, density=2.20)
    substrate = Layer(name="Si Substrate", formula="Si", thickness=0.0, roughness=1.0, density=2.33)

    stack = LayerStack(
        ambient=ambient,
        layers=[top_surf, sno_layer, native_ox],
        substrate=substrate,
        wavelength_angstrom=wavelength_angstrom
    )
    stack.update_all_optical_constants()

    # 2. Simulate true physical reflectivity
    true_scale = 1.0e7  # 1e7 counts/sec incident beam intensity
    true_bkg = 0.5      # 0.5 CPS dark counts
    counting_time_per_point = 0.5  # seconds

    true_refl = simulate_parratt(theta, stack, scale=1.0, background=0.0, wavelength_angstrom=wavelength_angstrom)

    # Calculate expected unattenuated CPS
    expected_cps = true_scale * true_refl + true_bkg

    # 3. Simulate beam attenuator
    # Panalytical instruments automatically insert an attenuator (e.g. 100x or 1000x)
    # when intensity exceeds detector saturation (e.g. at 2Theta <= 0.8 deg)
    attenuation_factors = np.ones_like(two_theta)
    low_angle_mask = two_theta <= 0.80
    attenuation_factors[low_angle_mask] = 100.0

    # Measured CPS on detector (attenuated)
    detector_cps = expected_cps / attenuation_factors

    # Detector counts with Gaussian approximation for large counts
    rng = np.random.default_rng(42)
    expected_counts = np.maximum(detector_cps * counting_time_per_point, 0.0)
    
    # Use normal approx when counts > 500
    noisy_counts = np.zeros_like(expected_counts)
    large_mask = expected_counts > 500
    noisy_counts[large_mask] = rng.normal(expected_counts[large_mask], np.sqrt(expected_counts[large_mask]))
    noisy_counts[~large_mask] = rng.poisson(expected_counts[~large_mask])
    noisy_counts = np.maximum(noisy_counts, 0.0)

    # Beam-corrected intensity (CPS)
    measured_cps = (noisy_counts / counting_time_per_point) * attenuation_factors

    # 4. Generate Panalytical XRDML XML format
    counts_str = " ".join(f"{int(round(c))}" for c in noisy_counts)
    times_str = " ".join(f"{counting_time_per_point:.1f}" for _ in noisy_counts)
    factors_str = " ".join(f"{int(f)}" for f in attenuation_factors)

    xrdml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<xrdMeasurements xmlns="http://www.panalytical.com/xrdml" status="Completed">
  <comment>
    <entry>Synthetically generated Malvern Panalytical XRR raw data for Si/SiO2/SnO thin film</entry>
  </comment>
  <sample status="Valid">
    <id>SnO-25nm-Si</id>
    <name>SnO 25nm / SiO2 1.8nm / Si Substrate</name>
  </sample>
  <wavelength type="K-Alpha 1">
    <kAlpha1>0.1540598</kAlpha1>
    <kAlpha2>0.1544426</kAlpha2>
    <ratioKAlpha2KAlpha1>0.5</ratioKAlpha2KAlpha1>
    <usedWavelength>0.1540598</usedWavelength>
  </wavelength>
  <xrdMeasurement measurementType="Scan">
    <scan scanAxis="2Theta" status="Completed">
      <dataPoints>
        <positions axis="2Theta" unit="deg">
          <startPosition>{two_theta[0]:.4f}</startPosition>
          <endPosition>{two_theta[-1]:.4f}</endPosition>
        </positions>
        <intensities unit="counts">{counts_str}</intensities>
        <countingTimes unit="seconds">{times_str}</countingTimes>
        <beamAttenuationFactors>{factors_str}</beamAttenuationFactors>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
"""

    with open(filename_xrdml, "w", encoding="utf-8") as f:
        f.write(xrdml_content)

    # Also save CSV version for easy ASCII testing
    with open(filename_csv, "w", encoding="utf-8") as f:
        f.write("# 2Theta(deg),Intensity(CPS),AttenuatedCounts,AttenuationFactor\n")
        for th, cps, cnt, att in zip(two_theta, measured_cps, noisy_counts, attenuation_factors):
            f.write(f"{th:.4f},{cps:.2f},{int(round(cnt))},{int(att)}\n")

    print(f"Generated {filename_xrdml} ({len(two_theta)} points)")
    print(f"Generated {filename_csv}")


if __name__ == "__main__":
    generate_synthetic_xrr()
