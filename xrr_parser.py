"""
xrr_parser.py
=============
Malvern Panalytical .xrdml (XML-based) and ASCII (CSV/TXT/DAT) Raw Data Parser.
Extracts 2-Theta / Omega scan axes, Counts / CPS, measurement counting times,
beam attenuation corrections, and X-ray wavelength (Cu K-alpha default: 1.54056 A).
Automatically converts 2-Theta to incident angle Theta and scattering vector q_z.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import io
import re


@dataclass
class XRRData:
    """Standardized XRR Experimental Dataset Container."""
    two_theta: np.ndarray      # 2-Theta angles in degrees
    theta: np.ndarray          # Incident angle Theta in degrees (two_theta / 2)
    intensity: np.ndarray      # Intensity (counts or CPS, corrected for attenuation)
    q_z: np.ndarray            # Scattering vector q_z = 4*pi*sin(theta)/lambda in Angstrom^-1
    wavelength_angstrom: float # X-ray wavelength in Angstroms (default 1.54056 A for Cu Ka1)
    intensity_raw: Optional[np.ndarray] = None
    counting_time: Optional[np.ndarray] = None
    attenuation_factors: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    filename: str = ""

    @property
    def wavelength_nm(self) -> float:
        """X-ray wavelength in nanometers."""
        return self.wavelength_angstrom / 10.0

    @property
    def q_z_nm(self) -> np.ndarray:
        """Scattering vector q_z in nm^-1."""
        return self.q_z * 10.0

    @property
    def incident_slit_mm(self) -> Optional[float]:
        """Automatically extracts incident beam slit width in mm from metadata if present."""
        def _search_dict(d: dict) -> Optional[float]:
            for k, v in d.items():
                if isinstance(v, dict):
                    res = _search_dict(v)
                    if res is not None:
                        return res
                k_low = str(k).lower()
                if "incident slit" in k_low or "beam slit" in k_low or "is slit" in k_low or "divergence slit" in k_low:
                    match = re.search(r'([0-9]+\.?[0-9]*)', str(v))
                    if match:
                        return float(match.group(1))
            return None
        return _search_dict(self.metadata)

    def get_valid_mask(self) -> np.ndarray:
        """Returns boolean mask where 2-Theta > 0 and Intensity > 0."""
        return (self.two_theta > 0) & (self.intensity > 0) & np.isfinite(self.intensity)

    def filter_valid(self) -> "XRRData":
        """Returns a new XRRData instance with non-positive/NaN values filtered out."""
        mask = self.get_valid_mask()
        return XRRData(
            two_theta=self.two_theta[mask],
            theta=self.theta[mask],
            intensity=self.intensity[mask],
            q_z=self.q_z[mask],
            wavelength_angstrom=self.wavelength_angstrom,
            intensity_raw=self.intensity_raw[mask] if self.intensity_raw is not None else None,
            counting_time=self.counting_time[mask] if self.counting_time is not None else None,
            attenuation_factors=self.attenuation_factors[mask] if self.attenuation_factors is not None else None,
            metadata=self.metadata.copy(),
            filename=self.filename
        )


def _strip_tag(tag: str) -> str:
    """Helper to remove XML namespaces from tag name."""
    return tag.split("}")[-1] if "}" in tag else tag


def parse_xrdml(content: str | bytes, filename: str = "raw_data.xrdml") -> XRRData:
    """
    Parses Malvern Panalytical .xrdml XML format.

    Extracts:
      - Wavelength (kAlpha1, kAlpha2, usedWavelength)
      - 2Theta or Omega scan positions
      - Intensities (counts / CPS)
      - Counting time per point
      - Beam attenuation factors (auto-applied to scale intensities)

    Args:
        content: XML string or bytes.
        filename: Optional source file name.

    Returns:
        XRRData object populated with physics-ready arrays.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    tree = ET.parse(io.BytesIO(content))
    root = tree.getroot()

    metadata: Dict[str, Any] = {}
    wavelength_angstrom = 1.54056  # Standard Cu K-alpha 1 default in Angstroms

    # 1. Extract Wavelength safely
    for elem in root.iter():
        tag = _strip_tag(elem.tag)
        
        # Check kAlpha1 tag directly
        if tag == "kAlpha1" and elem.text and elem.text.strip():
            try:
                val = float(elem.text.strip())
                unit = elem.attrib.get("unit", "").lower()
                if "nm" in unit or val < 1.0:
                    wavelength_angstrom = val * 10.0
                else:
                    wavelength_angstrom = val
                metadata["kAlpha1"] = wavelength_angstrom
            except ValueError:
                pass

        # Check usedWavelength tag (can be leaf float or parent container)
        elif tag == "usedWavelength":
            if elem.text and elem.text.strip():
                try:
                    val = float(elem.text.strip())
                    unit = elem.attrib.get("unit", "").lower()
                    if "nm" in unit or val < 1.0:
                        wavelength_angstrom = val * 10.0
                    else:
                        wavelength_angstrom = val
                    metadata["usedWavelength"] = wavelength_angstrom
                except ValueError:
                    pass
            # Check child elements inside usedWavelength
            for child in elem:
                if _strip_tag(child.tag) == "kAlpha1" and child.text and child.text.strip():
                    try:
                        val = float(child.text.strip())
                        unit = child.attrib.get("unit", "").lower()
                        if "nm" in unit or val < 1.0:
                            wavelength_angstrom = val * 10.0
                        else:
                            wavelength_angstrom = val
                        metadata["kAlpha1"] = wavelength_angstrom
                    except ValueError:
                        pass

    # 2. Extract Measurement Points & Intensities
    scan_points: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None
    counting_times: Optional[np.ndarray] = None
    common_counting_time: Optional[float] = None
    attenuation_factors: Optional[np.ndarray] = None
    axis_type = "2Theta"

    for data_points in root.iter():
        if _strip_tag(data_points.tag) != "dataPoints":
            continue

        for child in data_points:
            child_tag = _strip_tag(child.tag)

            # Check positions
            if child_tag == "positions":
                axis = child.attrib.get("axis", "2Theta")
                axis_type = axis
                start_pos_elem = None
                end_pos_elem = None
                list_pos_elem = None

                for p in child:
                    ptag = _strip_tag(p.tag)
                    if ptag == "startPosition" and p.text and p.text.strip():
                        try:
                            start_pos_elem = float(p.text.strip())
                        except ValueError:
                            pass
                    elif ptag == "endPosition" and p.text and p.text.strip():
                        try:
                            end_pos_elem = float(p.text.strip())
                        except ValueError:
                            pass
                    elif ptag == "listPositions" and p.text and p.text.strip():
                        list_pos_elem = p.text.strip()

                if list_pos_elem:
                    nums = [float(x) for x in list_pos_elem.replace(",", " ").split() if x.strip()]
                    scan_points = np.array(nums, dtype=np.float64)
                elif start_pos_elem is not None and end_pos_elem is not None:
                    metadata["startPosition"] = start_pos_elem
                    metadata["endPosition"] = end_pos_elem

            # Check intensities
            elif child_tag == "intensities":
                unit = child.attrib.get("unit", "counts")
                metadata["intensity_unit"] = unit
                if child.text and child.text.strip():
                    nums = [float(x) for x in child.text.replace(",", " ").split() if x.strip()]
                    intensities = np.array(nums, dtype=np.float64)

            # Check per-point counting times
            elif child_tag == "countingTimes":
                if child.text and child.text.strip():
                    nums = [float(x) for x in child.text.replace(",", " ").split() if x.strip()]
                    counting_times = np.array(nums, dtype=np.float64)

            # Check common counting time (single scalar value)
            elif child_tag == "commonCountingTime":
                if child.text and child.text.strip():
                    try:
                        common_counting_time = float(child.text.strip())
                        metadata["commonCountingTime"] = common_counting_time
                    except ValueError:
                        pass

            # Check beam attenuator factors
            elif child_tag == "beamAttenuationFactors":
                if child.text and child.text.strip():
                    nums = [float(x) for x in child.text.replace(",", " ").split() if x.strip()]
                    attenuation_factors = np.array(nums, dtype=np.float64)

    if intensities is None:
        raise ValueError("XRDML 파일 내에서 <intensities> 데이터 노드를 찾을 수 없습니다.")

    num_points = len(intensities)

    # Reconstruct positions if given as start and end
    if scan_points is None:
        if "startPosition" in metadata and "endPosition" in metadata:
            scan_points = np.linspace(metadata["startPosition"], metadata["endPosition"], num_points)
        else:
            raise ValueError("XRDML 파일에서 스캔 각도(positions) 데이터를 추출할 수 없습니다.")

    # Reconstruct counting times if commonCountingTime is given
    if counting_times is None and common_counting_time is not None:
        counting_times = np.full(num_points, common_counting_time, dtype=np.float64)

    # Harmonize 2-Theta vs Omega
    if "omega" in axis_type.lower() and "2theta" not in axis_type.lower():
        two_theta = scan_points * 2.0
    else:
        two_theta = scan_points

    intensity_raw = intensities.copy()

    # Apply beam attenuation correction (intensity = raw_intensity * attenuation_factor)
    if attenuation_factors is not None and len(attenuation_factors) == num_points:
        intensities = intensities * attenuation_factors
        metadata["attenuation_corrected"] = True
    else:
        metadata["attenuation_corrected"] = False

    # Normalize to CPS if counts and counting times are present
    if counting_times is not None and len(counting_times) == num_points:
        if metadata.get("intensity_unit", "").lower() == "counts":
            intensities = np.where(counting_times > 0, intensities / counting_times, intensities)
            metadata["converted_to_cps"] = True

    theta = two_theta / 2.0
    theta_rad = np.radians(theta)
    q_z = (4.0 * np.pi / wavelength_angstrom) * np.sin(theta_rad)

    return XRRData(
        two_theta=two_theta,
        theta=theta,
        intensity=intensities,
        q_z=q_z,
        wavelength_angstrom=wavelength_angstrom,
        intensity_raw=intensity_raw,
        counting_time=counting_times,
        attenuation_factors=attenuation_factors,
        metadata=metadata,
        filename=filename
    )


def parse_ascii(content: str, filename: str = "raw_data.txt", wavelength_angstrom: float = 1.54056) -> XRRData:
    """
    Parses generic CSV, TSV, Rigaku SmartLab TXT, Bruker RAW/XY, and whitespace-delimited ASCII files.
    Robustly handles arbitrary metadata headers, variable delimiter lines, and diverse instrument formats.

    Args:
        content: File text content.
        filename: Optional source file name.
        wavelength_angstrom: Default wavelength in Angstroms.

    Returns:
        XRRData object.
    """
    lines = content.strip().splitlines()
    if not lines:
        raise ValueError("파일 내용이 비어 있습니다.")

    metadata: Dict[str, Any] = {}

    def _is_float(val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False

    header_lines = []
    numeric_rows = []
    table_header_cols = None

    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith(("#", "//", "!", "%")):
            continue

        # Split line by comma, tab, or whitespace
        tokens = [t.strip() for t in re.split(r'[\s,\t]+', line_s) if t.strip()]

        # Check if line contains at least 2 tokens and first 2 tokens are numeric
        if len(tokens) >= 2 and _is_float(tokens[0]) and _is_float(tokens[1]):
            numeric_rows.append([float(t) for t in tokens if _is_float(t)])
        else:
            header_lines.append(line_s)
            # Check for metadata key-value pair
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2 and parts[0].strip():
                    metadata[parts[0].strip()] = parts[1].strip()
            elif ':' in line_s and not line_s.startswith("http"):
                parts = line_s.split(':', 1)
                metadata[parts[0].strip()] = parts[1].strip()
            elif '=' in line_s:
                parts = line_s.split('=', 1)
                metadata[parts[0].strip()] = parts[1].strip()

            # Check if this non-numeric line could be a table header row
            lower_tokens = [t.lower() for t in tokens]
            if any(k in lower_tokens for k in ["angle", "2theta", "theta", "omega", "qz", "q_z", "2th", "intensity", "cps", "counts"]):
                table_header_cols = lower_tokens

    if not numeric_rows:
        raise ValueError("텍스트 파일 내에서 유효한 수치 데이터(각도 및 반사율 강도) 행을 찾을 수 없습니다.")

    # Convert numeric rows to NumPy 2D array
    data_mat = np.array([r[:2] for r in numeric_rows], dtype=np.float64)
    x_vals = data_mat[:, 0]
    y_vals = data_mat[:, 1]

    # Check for wavelength in metadata
    wl = wavelength_angstrom
    for k, v in metadata.items():
        k_low = k.lower()
        if "wavelength" in k_low or "lambda" in k_low:
            match = re.search(r'([0-9]+\.?[0-9]*)', v)
            if match:
                val = float(match.group(1))
                wl = val * 10.0 if val < 1.0 else val
        elif "x-ray" in k_low or "target" in k_low:
            if "cu" in v.lower() or "45kv" in v.lower():
                wl = 1.54056

    # Determine axis type (2Theta, Theta, q_z)
    is_qz = False
    is_theta = False

    scan_mode = metadata.get("ScanningMode", "") or metadata.get("ScanAxis", "") or ""
    scan_mode_low = scan_mode.lower()

    if table_header_cols:
        col0 = table_header_cols[0]
        if any(k in col0 for k in ["qz", "q_z", "q (1/a)", "q(1/a)", "q [1/a]"]):
            is_qz = True
        elif any(k in col0 for k in ["theta", "omega", "th"]) and "2theta" not in col0:
            is_theta = True
    elif "omega" in scan_mode_low and "2theta" not in scan_mode_low:
        is_theta = True

    if is_qz:
        q_z = x_vals
        sin_th = np.clip(q_z * wl / (4.0 * np.pi), -1.0, 1.0)
        theta = np.degrees(np.arcsin(sin_th))
        two_theta = theta * 2.0
    elif is_theta:
        theta = x_vals
        two_theta = theta * 2.0
        q_z = (4.0 * np.pi / wl) * np.sin(np.radians(theta))
    else:
        # Default: 2Theta (e.g. 2Theta/Omega, 2Theta, Gonio)
        two_theta = x_vals
        theta = two_theta / 2.0
        q_z = (4.0 * np.pi / wl) * np.sin(np.radians(theta))

    return XRRData(
        two_theta=two_theta,
        theta=theta,
        intensity=y_vals,
        q_z=q_z,
        wavelength_angstrom=wl,
        filename=filename,
        metadata={"source_format": "ASCII/TXT", "metadata": metadata}
    )


def load_xrr_file(file_content: bytes | str, filename: str, wavelength_angstrom: float = 1.54056) -> XRRData:
    """
    Unified file loader for Panalytical .xrdml, .csv, .txt, and .dat files.
    """
    if filename.lower().endswith(".xrdml"):
        data = parse_xrdml(file_content, filename=filename)
    else:
        text = file_content.decode("utf-8", errors="replace") if isinstance(file_content, bytes) else file_content
        data = parse_ascii(text, filename=filename, wavelength_angstrom=wavelength_angstrom)

    return data.filter_valid()
