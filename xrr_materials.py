"""
xrr_materials.py
================
Thin-film materials database, chemical formula parser, and optical constants calculation.
Calculates Scattering Length Density (SLD), electron density, and refractive index
decrement parameters (delta, beta) for X-ray reflectivity at given wavelength (e.g. Cu K-alpha).
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import re
import numpy as np

# Classical electron radius in Angstroms: r_e = 2.8179403262e-5 A
R_E_ANGSTROM = 2.8179403262e-5

# Avogadro constant: mol^-1
N_A = 6.02214076e23

# Element database: Symbol -> (Atomic Number Z, Atomic Weight M (g/mol), f1_CuKa, f2_CuKa)
# f1 and f2 tabulated at Cu K-alpha1 (8.048 keV, lambda = 1.54056 A)
ELEMENT_DATA: Dict[str, Tuple[int, float, float, float]] = {
    "H":  (1,  1.008,   1.000, 0.000001),
    "He": (2,  4.0026,  2.000, 0.00001),
    "Li": (3,  6.94,    3.001, 0.0001),
    "Be": (4,  9.0122,  4.004, 0.0008),
    "B":  (5,  10.81,   5.018, 0.003),
    "C":  (6,  12.011,  6.035, 0.009),
    "N":  (7,  14.007,  7.065, 0.022),
    "O":  (8,  15.999,  8.113, 0.046),
    "F":  (9,  18.998,  9.186, 0.084),
    "Ne": (10, 20.180,  10.29, 0.141),
    "Na": (11, 22.990,  11.44, 0.223),
    "Mg": (12, 24.305,  12.63, 0.334),
    "Al": (13, 26.982,  13.88, 0.478),
    "Si": (14, 28.085,  14.47, 0.330),
    "P":  (15, 30.974,  15.65, 0.449),
    "S":  (16, 32.06,   16.73, 0.589),
    "Cl": (17, 35.45,   17.84, 0.757),
    "Ar": (18, 39.948,  18.98, 0.957),
    "K":  (19, 39.098,  20.17, 1.201),
    "Ca": (20, 40.078,  21.40, 1.488),
    "Sc": (21, 44.956,  22.68, 1.832),
    "Ti": (22, 47.867,  23.95, 2.235),
    "V":  (23, 50.942,  25.26, 2.709),
    "Cr": (24, 51.996,  26.54, 3.251),
    "Mn": (25, 54.938,  27.79, 3.864),
    "Fe": (26, 55.845,  28.98, 4.549),
    "Co": (27, 58.933,  25.04, 3.812), # Near edge correction
    "Ni": (28, 58.693,  25.10, 1.050),
    "Cu": (29, 63.546,  26.83, 0.589),
    "Zn": (30, 65.38,   28.53, 0.716),
    "Ga": (31, 69.723,  30.22, 0.887),
    "Ge": (32, 72.630,  31.87, 1.099),
    "As": (33, 74.922,  33.50, 1.348),
    "Se": (34, 78.971,  35.10, 1.638),
    "Br": (35, 79.904,  36.68, 1.970),
    "Kr": (36, 83.798,  38.23, 2.348),
    "Rb": (37, 85.468,  39.77, 2.776),
    "Sr": (38, 87.62,   41.27, 3.250),
    "Y":  (39, 88.906,  42.75, 3.774),
    "Zr": (40, 91.224,  44.18, 4.354),
    "Nb": (41, 92.906,  45.58, 4.987),
    "Mo": (42, 95.95,   46.94, 5.676),
    "Ru": (44, 101.07,  49.56, 7.218),
    "Rh": (45, 102.91,  50.81, 8.082),
    "Pd": (46, 106.42,  52.01, 9.014),
    "Ag": (47, 107.87,  53.18, 10.01),
    "Cd": (48, 112.41,  54.30, 11.08),
    "In": (49, 114.82,  55.39, 12.21),
    "Sn": (50, 118.71,  56.44, 13.40),
    "Sb": (51, 121.76,  57.44, 14.65),
    "Te": (52, 127.60,  58.41, 15.98),
    "I":  (53, 126.90,  59.34, 17.37),
    "Ba": (56, 137.33,  61.90, 21.84),
    "La": (57, 138.91,  62.70, 23.46),
    "Ce": (58, 140.12,  63.48, 25.13),
    "Nd": (60, 144.24,  64.98, 28.65),
    "Hf": (72, 178.49,  74.80, 5.250),
    "Ta": (73, 180.95,  75.50, 5.750),
    "W":  (74, 183.84,  76.20, 6.300),
    "Re": (75, 186.21,  76.90, 6.900),
    "Os": (76, 190.23,  77.60, 7.550),
    "Ir": (77, 192.22,  78.30, 8.250),
    "Pt": (78, 195.08,  79.00, 9.000),
    "Au": (79, 196.97,  79.70, 9.800),
    "Pb": (82, 207.2,   82.00, 12.50),
    "Bi": (83, 208.98,  82.80, 13.50)
}

# Standard thin-film and substrate reference materials: Formula -> Nominal Density (g/cm^3)
STANDARD_MATERIALS: Dict[str, float] = {
    "Si": 2.33,
    "SiO2": 2.20,
    "Si3N4": 3.10,
    "Ge": 5.323,
    "GaAs": 5.317,
    "Al2O3": 3.98,       # Sapphire or dense ALD
    "TiO2": 4.23,        # Rutile (anatase ~3.9)
    "SnO": 6.45,
    "SnO2": 6.95,
    "HfO2": 9.68,
    "ZrO2": 5.68,
    "Ta2O5": 8.20,
    "ZnO": 5.61,
    "In2O3": 7.18,
    "Au": 19.32,
    "Pt": 21.45,
    "Ni": 8.90,
    "Cu": 8.96,
    "Ti": 4.51,
    "Cr": 7.19,
    "W": 19.25,
    "Mo": 10.28,
    "Ru": 12.45,
    "Pd": 12.02,
    "Ag": 10.49,
    "C": 2.00,           # Amorphous carbon / diamond-like
    "Air": 0.0012,
    "Vacuum": 0.0
}


def parse_chemical_formula(formula: str) -> Dict[str, float]:
    """
    Parses a chemical formula string into element stoichiometric ratios.
    Supports formulas like 'SiO2', 'Al2O3', 'SnO', 'Hf0.5Zr0.5O2', 'Ti', 'InGaZnO4'.

    Args:
        formula: Chemical formula string.

    Returns:
        Dictionary of {ElementSymbol: stoichiometric_count}.
    """
    formula = formula.strip()
    if not formula:
        return {}

    # Regex matches Element Symbol followed by optional integer or decimal count
    pattern = r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)"
    matches = re.findall(pattern, formula)

    composition: Dict[str, float] = {}
    for element, count in matches:
        if not element:
            continue
        c = float(count) if count else 1.0
        composition[element] = composition.get(element, 0.0) + c

    return composition


@dataclass
class MaterialProperties:
    formula: str
    density_g_cm3: float           # Mass density in g/cm^3
    molar_mass_g_mol: float        # Formula molecular weight in g/mol
    electron_density_e_A3: float   # Electron density in electrons / Angstrom^3
    sld_real_A2: float             # Real Scattering Length Density in Angstrom^-2
    sld_imag_A2: float             # Imaginary Scattering Length Density in Angstrom^-2
    delta: float                   # Refractive index decrement delta (n = 1 - delta - i*beta)
    beta: float                    # Absorption decrement beta
    critical_angle_deg: float      # Theoretical critical angle theta_c in degrees


def calculate_optical_constants(
    formula: str,
    density_g_cm3: Optional[float] = None,
    wavelength_angstrom: float = 1.54056
) -> MaterialProperties:
    """
    Calculates optical constants (delta, beta, SLD, electron density, critical angle)
    for a given material and density at specified X-ray wavelength.

    Args:
        formula: Chemical formula or material key (e.g. 'Si', 'SiO2', 'SnO').
        density_g_cm3: Density in g/cm^3. If None, uses nominal library value.
        wavelength_angstrom: Wavelength in Angstroms (default: Cu Ka1 = 1.54056 A).

    Returns:
        MaterialProperties dataclass.
    """
    formula = formula.strip()
    
    # Handle vacuum / air
    if formula.lower() in ["vacuum", "air"]:
        return MaterialProperties(
            formula=formula,
            density_g_cm3=0.0,
            molar_mass_g_mol=0.0,
            electron_density_e_A3=0.0,
            sld_real_A2=0.0,
            sld_imag_A2=0.0,
            delta=0.0,
            beta=0.0,
            critical_angle_deg=0.0
        )

    # Use nominal density if not provided
    if density_g_cm3 is None or density_g_cm3 <= 0:
        density_g_cm3 = STANDARD_MATERIALS.get(formula, 3.0)

    composition = parse_chemical_formula(formula)
    if not composition:
        # Fallback default (Si-like)
        composition = {"Si": 1.0}

    total_molar_mass = 0.0
    total_f1 = 0.0
    total_f2 = 0.0
    total_electrons = 0.0

    for element, count in composition.items():
        if element in ELEMENT_DATA:
            z, m, f1, f2 = ELEMENT_DATA[element]
        else:
            # Fallback estimation for unknown elements
            z = 20
            m = 40.0
            f1 = float(z)
            f2 = 0.5
        total_molar_mass += count * m
        total_electrons += count * z
        total_f1 += count * f1
        total_f2 += count * f2

    if total_molar_mass <= 0:
        total_molar_mass = 28.085

    # Number density of formula units in Angstrom^-3:
    # N_density = (density in g/cm^3) * N_A / M * 10^-24 (since 1 cm^3 = 10^24 A^3)
    n_density_A3 = (density_g_cm3 * N_A / total_molar_mass) * 1e-24

    # Electron density in e/A^3
    rho_e = n_density_A3 * total_electrons

    # Real and Imaginary Scattering Length Density (SLD) in A^-2
    # SLD_real = r_e * sum(count * f1) * n_density
    sld_real = R_E_ANGSTROM * n_density_A3 * total_f1
    sld_imag = R_E_ANGSTROM * n_density_A3 * total_f2

    # Refractive index: n = 1 - delta - i * beta
    # delta = (lambda^2 / (2*pi)) * SLD_real
    # beta  = (lambda^2 / (2*pi)) * SLD_imag
    lambda_factor = (wavelength_angstrom**2) / (2.0 * np.pi)
    delta = lambda_factor * sld_real
    beta = lambda_factor * sld_imag

    # Theoretical critical angle theta_c in radians: theta_c ≈ sqrt(2 * delta)
    if delta > 0:
        theta_c_rad = np.sqrt(2.0 * delta)
        theta_c_deg = np.degrees(theta_c_rad)
    else:
        theta_c_deg = 0.0

    return MaterialProperties(
        formula=formula,
        density_g_cm3=density_g_cm3,
        molar_mass_g_mol=total_molar_mass,
        electron_density_e_A3=rho_e,
        sld_real_A2=sld_real,
        sld_imag_A2=sld_imag,
        delta=delta,
        beta=beta,
        critical_angle_deg=theta_c_deg
    )


def get_default_materials_list() -> List[str]:
    """Returns sorted list of standard material names available in library."""
    return sorted(list(STANDARD_MATERIALS.keys()))
