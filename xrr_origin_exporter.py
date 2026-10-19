"""
xrr_origin_exporter.py
======================
OriginLab (Origin) 전용 XRR 피팅 데이터 내보내기 유틸리티.
Origin에서 바로 드래그 앤 드롭하여 논문 수준의 그래프를 작성할 수 있도록
Long Name, Units, Comments가 완벽하게 서식화된 .DAT, .CSV 및
다중 시트 통합 엑셀(.XLSX)과 일괄 ZIP 패키지를 생성합니다.
"""

import io
import time
import zipfile
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from xrr_engine import LayerStack
from xrr_materials import calculate_optical_constants
from xrr_optimizer import FitResult


def generate_origin_curve_dat(
    res: FitResult,
    raw_filename: str = ""
) -> str:
    """
    OriginLab 전용 3행 헤더(Long Name, Units, Comments)가 포함된
    경면 반사율(XRR) 피팅 곡선 ASCII (.dat) 파일 생성.
    Origin의 워크시트에 드래그 앤 드롭 시 자동으로 열 이름/단위/주석이 매핑됩니다.
    """
    log_exp = np.log10(np.maximum(res.experimental_intensity, 1e-12))
    log_sim = np.log10(np.maximum(res.simulated_intensity, 1e-12))
    residuals = log_exp - log_sim

    match_val = getattr(res, "match_rate", 0.0)
    lines = [
        f"# OriginLab XRR Fitting Curve Export",
        f"# Sample File: {raw_filename or 'Experimental Data'}",
        f"# Optimal Engine: {res.best_engine}",
        f"# Graph Match Rate: {match_val:.2f}% | R-Factor: {res.r_factor*100.0:.2f}% | Reduced Chi2: {res.reduced_chi2:.3f} | Log-MSE: {res.log_mse:.5f}",
        f"# Zero Offset (Delta 2Theta): {getattr(res, 'zero_offset_2theta_deg', 0.0):+.4f} deg | Beam FWHM: {getattr(res, 'resolution_fwhm_deg', 0.015):.4f} deg",
        "TwoTheta\tTheta\tQz\tExp_Intensity\tFit_Intensity\tLog10_Exp\tLog10_Fit\tResidual_Log",
        "deg\tdeg\t1/A\tCPS\tCPS\tlog10(CPS)\tlog10(CPS)\ta.u.",
        "Scattering Angle 2-Theta\tIncident Angle Theta\tScattering Vector Qz\tExperimental Raw CPS\tFitted Theoretical CPS\tLog10 Experimental Intensity\tLog10 Fitted Intensity\tResidual Delta Log10(I)"
    ]

    for i in range(len(res.two_theta)):
        lines.append(
            f"{res.two_theta[i]:.5f}\t{res.theta[i]:.5f}\t{res.q_z[i]:.6f}\t"
            f"{res.experimental_intensity[i]:.6e}\t{res.simulated_intensity[i]:.6e}\t"
            f"{log_exp[i]:.6f}\t{log_sim[i]:.6f}\t{residuals[i]:.6f}"
        )

    return "\n".join(lines)


def generate_origin_curve_df(res: FitResult) -> pd.DataFrame:
    """
    피팅 곡선 데이터의 표준 Pandas DataFrame 반환 (Origin CSV용).
    """
    log_exp = np.log10(np.maximum(res.experimental_intensity, 1e-12))
    log_sim = np.log10(np.maximum(res.simulated_intensity, 1e-12))
    residuals = log_exp - log_sim

    return pd.DataFrame({
        "TwoTheta_deg": np.round(res.two_theta, 5),
        "Theta_deg": np.round(res.theta, 5),
        "Qz_invA": np.round(res.q_z, 6),
        "Exp_Intensity_CPS": res.experimental_intensity,
        "Fit_Intensity_CPS": res.simulated_intensity,
        "Log10_Exp": np.round(log_exp, 5),
        "Log10_Fit": np.round(log_sim, 5),
        "Residual_Log": np.round(residuals, 5)
    })


def generate_origin_depth_dat(
    z_nm: np.ndarray,
    rho_depth: np.ndarray,
    sld_depth: np.ndarray,
    stack: LayerStack
) -> str:
    """
    OriginLab 전용 3행 헤더가 포함된 깊이 프로파일 ASCII (.dat) 파일 생성.
    """
    lines = [
        f"# OriginLab XRR Depth Profile Export",
        f"# Stack: {' / '.join([l.name for l in stack.layers])} on {stack.substrate.name}",
        f"# Interface transition modeled with error function erf(z/sqrt(2)*sigma)",
        "Depth_z\tDepth_Angstrom\tMass_Density\tElectron_Density\tReal_SLD",
        "nm\tA\tg/cm^3\te/A^3\t10^-6 A^-2",
        "Depth from Surface (z)\tDepth in Angstrom\tMass Density (rho)\tElectron Density\tScattering Length Density (SLD)"
    ]

    z_A = z_nm * 10.0
    for i in range(len(z_nm)):
        # Calculate approximate electron density from SLD (rho_e ~ SLD / r_e, r_e = 2.8179e-5 A)
        r_e = 2.8179403262e-5  # A
        # sld_depth is in 10^-6 A^-2, so sld_actual = sld_depth * 1e-6 A^-2
        # rho_e = sld_actual / (r_e)
        ed_val = (sld_depth[i] * 1e-6) / (r_e * 1e-8) if sld_depth[i] > 0 else 0.0

        lines.append(
            f"{z_nm[i]:.4f}\t{z_A[i]:.3f}\t{rho_depth[i]:.4f}\t{ed_val:.4f}\t{sld_depth[i]:.4f}"
        )

    return "\n".join(lines)


def generate_origin_depth_df(
    z_nm: np.ndarray,
    rho_depth: np.ndarray,
    sld_depth: np.ndarray
) -> pd.DataFrame:
    """
    깊이 프로파일 데이터의 표준 Pandas DataFrame 반환.
    """
    r_e = 2.8179403262e-5
    ed = [(s * 1e-6) / (r_e * 1e-8) if s > 0 else 0.0 for s in sld_depth]
    return pd.DataFrame({
        "Depth_z_nm": np.round(z_nm, 4),
        "Depth_z_Angstrom": np.round(z_nm * 10.0, 3),
        "Mass_Density_g_cm3": np.round(rho_depth, 4),
        "Electron_Density_e_A3": np.round(ed, 4),
        "Real_SLD_1e6_invA2": np.round(sld_depth, 4)
    })


def generate_origin_params_df(stack: LayerStack) -> pd.DataFrame:
    """
    피팅으로 확정된 층별 파라미터 DataFrame 반환.
    """
    rows = []
    for idx, l in enumerate(stack.layers):
        props = calculate_optical_constants(l.formula, l.density, stack.wavelength_angstrom)
        rows.append({
            "Layer_Index": idx + 1,
            "Layer_Name": l.name,
            "Formula": l.formula,
            "Thickness_A": round(l.thickness, 2),
            "Thickness_nm": round(l.thickness / 10.0, 3),
            "Roughness_A": round(l.roughness, 2),
            "Density_g_cm3": round(l.density, 3),
            "Electron_Density_e_A3": round(props.electron_density_e_A3, 3),
            "SLD_Real_1e6_invA2": round(props.sld_real_A2 * 1e6, 3),
            "Delta_1e5": round(props.delta * 1e5, 3),
            "Beta_1e6": round(props.beta * 1e6, 3)
        })

    sub_l = stack.substrate
    sub_p = calculate_optical_constants(sub_l.formula, sub_l.density, stack.wavelength_angstrom)
    rows.append({
        "Layer_Index": "Substrate",
        "Layer_Name": sub_l.name,
        "Formula": sub_l.formula,
        "Thickness_A": "Bulk",
        "Thickness_nm": "Bulk",
        "Roughness_A": round(sub_l.roughness, 2),
        "Density_g_cm3": round(sub_l.density, 3),
        "Electron_Density_e_A3": round(sub_p.electron_density_e_A3, 3),
        "SLD_Real_1e6_invA2": round(sub_p.sld_real_A2 * 1e6, 3),
        "Delta_1e5": round(sub_p.delta * 1e5, 3),
        "Beta_1e6": round(sub_p.beta * 1e6, 3)
    })

    return pd.DataFrame(rows)


def generate_origin_excel_workbook(
    res: FitResult,
    stack: LayerStack,
    z_nm: np.ndarray,
    rho_depth: np.ndarray,
    sld_depth: np.ndarray,
    raw_filename: str = ""
) -> bytes:
    """
    OriginLab에서 'Import Excel'로 한 번에 불러올 수 있는
    5개 시트가 완벽 서식화된 종합 엑셀 통합 문서(.xlsx) 바이트 스트림 생성.
    """
    wb = Workbook()

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    accent_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
    center_align = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    # Sheet 1: XRR_Curves
    ws_curve = wb.active
    ws_curve.title = "XRR_Fitting_Curves"
    df_curve = generate_origin_curve_df(res)

    ws_curve.append(list(df_curve.columns))
    # Add units row
    units_row = ["deg", "deg", "1/A", "CPS", "CPS", "log10(CPS)", "log10(CPS)", "a.u."]
    ws_curve.append(units_row)

    for r_idx, row in df_curve.iterrows():
        ws_curve.append(list(row))

    # Format header rows
    for col_i in range(1, len(df_curve.columns) + 1):
        c1 = ws_curve.cell(row=1, column=col_i)
        c1.font = header_font
        c1.fill = header_fill
        c1.alignment = center_align
        c2 = ws_curve.cell(row=2, column=col_i)
        c2.font = Font(name="Calibri", size=10, italic=True, color="FFFFFF")
        c2.fill = accent_fill
        c2.alignment = center_align

    # Sheet 2: Depth_SLD_Profile
    ws_depth = wb.create_sheet(title="Depth_SLD_Profile")
    df_depth = generate_origin_depth_df(z_nm, rho_depth, sld_depth)
    ws_depth.append(list(df_depth.columns))
    ws_depth.append(["nm", "A", "g/cm^3", "e/A^3", "10^-6 A^-2"])

    for r_idx, row in df_depth.iterrows():
        ws_depth.append(list(row))

    for col_i in range(1, len(df_depth.columns) + 1):
        c1 = ws_depth.cell(row=1, column=col_i)
        c1.font = header_font
        c1.fill = header_fill
        c1.alignment = center_align
        c2 = ws_depth.cell(row=2, column=col_i)
        c2.font = Font(name="Calibri", size=10, italic=True, color="FFFFFF")
        c2.fill = accent_fill
        c2.alignment = center_align

    # Sheet 3: Layer_Parameters
    ws_params = wb.create_sheet(title="Layer_Parameters")
    df_p = generate_origin_params_df(stack)
    ws_params.append(list(df_p.columns))
    for r_idx, row in df_p.iterrows():
        ws_params.append(list(row))

    for col_i in range(1, len(df_p.columns) + 1):
        c1 = ws_params.cell(row=1, column=col_i)
        c1.font = header_font
        c1.fill = header_fill
        c1.alignment = center_align

    # Sheet 4: Fringe_Zone_Report
    ws_fringe = wb.create_sheet(title="Fringe_Zone_Report")
    fringe_rep = getattr(res, "fringe_zone_report", None)
    if fringe_rep:
        df_fz = pd.DataFrame(fringe_rep)
        ws_fringe.append(list(df_fz.columns))
        for r_idx, row in df_fz.iterrows():
            ws_fringe.append(list(row))
        for col_i in range(1, len(df_fz.columns) + 1):
            c1 = ws_fringe.cell(row=1, column=col_i)
            c1.font = header_font
            c1.fill = header_fill
            c1.alignment = center_align

    # Sheet 5: Fitting_Summary
    ws_sum = wb.create_sheet(title="Fitting_Summary")
    match_val = getattr(res, "match_rate", 0.0)
    init_match = getattr(res, "initial_match_rate", match_val)
    ref_iters = getattr(res, "refinement_iterations", 0)

    summary_items = [
        ("Analysis Property / Parameter", "Value", "Unit / Description"),
        ("Sample Raw File", raw_filename or "N/A", "Input Dataset"),
        ("Date & Time", time.strftime("%Y-%m-%d %H:%M:%S"), "Export Timestamp"),
        ("Optimal Engine", res.best_engine, "Physics Engine Winner"),
        ("Final Graph Match Rate", f"{match_val:.2f}%", "Overall Graph Overlap (Goal >= 98%)"),
        ("Initial 1st-Stage Match Rate", f"{init_match:.2f}%", "Before Fringe Piecewise Refinement"),
        ("Fringe Refinement Iterations", ref_iters, "Iterations until convergence"),
        ("Crystallographic R-Factor", f"{res.r_factor*100.0:.2f}%", "Residual percentage"),
        ("Log-MSE", f"{res.log_mse:.6f}", "Mean Squared Error on Log10(CPS)"),
        ("Reduced Chi-Square", f"{res.reduced_chi2:.3f}", "Goodness of fit"),
        ("Zero-Angle Offset (Delta 2Theta)", f"{getattr(res, 'zero_offset_2theta_deg', 0.0):+.4f}", "Goniometer zero shift (deg)"),
        ("Instrument Resolution FWHM", f"{getattr(res, 'resolution_fwhm_deg', 0.015):.4f}", "Gaussian slit divergence (deg)"),
        ("Sample Length (Footprint)", f"{res.sample_length_mm:.1f}" if res.sample_length_mm else "N/A", "Beam spillover correction (mm)"),
        ("Fitting Angle Range", f"{res.start_two_theta:.3f} to {res.end_two_theta:.3f}", "2-Theta range (deg)")
    ]

    for item in summary_items:
        ws_sum.append(list(item))

    for col_i in range(1, 4):
        c1 = ws_sum.cell(row=1, column=col_i)
        c1.font = header_font
        c1.fill = header_fill
        c1.alignment = center_align

    # Auto-adjust column widths across all sheets
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def generate_origin_zip_package(
    res: FitResult,
    stack: LayerStack,
    z_nm: np.ndarray,
    rho_depth: np.ndarray,
    sld_depth: np.ndarray,
    raw_filename: str = ""
) -> bytes:
    """
    OriginLab 사용자용 올인원 종합 압축 패키지(.zip) 생성.
    포함 내용:
      - 01_XRR_Fitting_Curves_Origin.dat (Origin 3행 헤더 포함)
      - 01_XRR_Fitting_Curves.csv (표준 CSV)
      - 02_Depth_SLD_Profile_Origin.dat (Origin 3행 헤더 포함)
      - 02_Depth_SLD_Profile.csv (표준 CSV)
      - 03_Layer_Parameters.csv (도출 물성치 표)
      - 04_Fringe_Zone_Report.csv (구간별 일치율 리포트)
      - XRR_Fitting_Complete_Origin.xlsx (다중 시트 통합 엑셀)
      - Origin_Quick_Start_Guide.txt (Origin 그래프 작성 가이드)
    """
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Fitting Curves
        dat_curve = generate_origin_curve_dat(res, raw_filename)
        zf.writestr("01_XRR_Fitting_Curves_Origin.dat", dat_curve.encode("utf-8"))

        df_curve = generate_origin_curve_df(res)
        zf.writestr("01_XRR_Fitting_Curves.csv", df_curve.to_csv(index=False).encode("utf-8-sig"))

        # 2. Depth Profile
        dat_depth = generate_origin_depth_dat(z_nm, rho_depth, sld_depth, stack)
        zf.writestr("02_Depth_SLD_Profile_Origin.dat", dat_depth.encode("utf-8"))

        df_depth = generate_origin_depth_df(z_nm, rho_depth, sld_depth)
        zf.writestr("02_Depth_SLD_Profile.csv", df_depth.to_csv(index=False).encode("utf-8-sig"))

        # 3. Parameters
        df_p = generate_origin_params_df(stack)
        zf.writestr("03_Layer_Parameters.csv", df_p.to_csv(index=False).encode("utf-8-sig"))

        # 4. Fringe Zone Report
        fringe_rep = getattr(res, "fringe_zone_report", None)
        if fringe_rep:
            df_fz = pd.DataFrame(fringe_rep)
            zf.writestr("04_Fringe_Zone_Report.csv", df_fz.to_csv(index=False).encode("utf-8-sig"))

        # 5. Multi-sheet Excel
        excel_bytes = generate_origin_excel_workbook(res, stack, z_nm, rho_depth, sld_depth, raw_filename)
        zf.writestr("XRR_Fitting_Complete_Origin.xlsx", excel_bytes)

        # 6. Quick Start Guide
        match_val = getattr(res, "match_rate", 0.0)
        guide_text = f"""================================================================================
OriginLab (Origin) XRR 피팅 데이터 불러오기 및 그래프 작성 가이드
================================================================================
시편 파일명: {raw_filename or 'Experimental Data'}
그래프 일치율: {match_val:.2f}% | R-Factor: {res.r_factor*100.0:.2f}% | 최적 엔진: {res.best_engine}
생성 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}

[1. Origin에서 1초 만에 불러오는 방법 (가장 쉬움)]
- 본 압축을 푼 뒤, 'XRR_Fitting_Complete_Origin.xlsx' 파일을 열려있는 Origin 프로그램
  화면으로 마우스 '드래그 앤 드롭(Drag & Drop)' 하세요.
- 또는 '01_XRR_Fitting_Curves_Origin.dat' 파일을 Origin 워크북으로 드래그 앤 드롭하면
  열 이름(Long Name), 단위(Units), 주석(Comments)이 자동으로 채워집니다.

[2. 논문 게재용 추천 플롯 설정법]
▶ 플롯 1: 실측값 vs 피팅값 반사율 오버레이 플롯
  - X축: Col(A) 'TwoTheta_deg'
  - Y1축 (실측값): Col(D) 'Log10_Exp' 또는 Col(D) 'Exp_Intensity' (Log Scale 지정)
    -> 마커(Scatter): 검정색/회색 계열 심볼, 크기 3~4pt
  - Y2축 (피팅값): Col(E) 'Log10_Fit' 또는 Col(E) 'Fit_Intensity'
    -> 선(Line): 빨간색(#DC2626) 실선, 선폭 1.5~2.0pt
  
▶ 플롯 2: 하단 잔차(Residual) 플롯 (논문 하단 서브플롯)
  - X축: Col(A) 'TwoTheta_deg'
  - Y축: Col(H) 'Residual_Log' (또는 Delta Log)
    -> 파란색 실선, Y축 범위: -0.15 ~ +0.15, Y=0 기준선 점선

▶ 플롯 3: 박막 깊이별 밀도 프로파일 (Depth Profile)
  - '02_Depth_SLD_Profile_Origin.dat' 사용
  - X축: Col(A) 'Depth_z' (nm 단위)
  - Y축: Col(C) 'Mass_Density' (g/cm^3) 및 Col(E) 'Real_SLD' (우측 2축 플롯)

[3. 컬럼 명세 요약]
- TwoTheta_deg: 2-Theta 산란 각도 (deg) [X]
- Theta_deg: 입사 각도 Theta (deg) [X]
- Qz_invA: 산란 벡터 크기 q_z (Å⁻¹) [X]
- Exp_Intensity_CPS: 실측 반사율 강도 (CPS) [Y]
- Fit_Intensity_CPS: 최적 물리 엔진 시뮬레이션 강도 (CPS) [Y]
- Log10_Exp: 실측값 log10(CPS) [Y]
- Log10_Fit: 피팅값 log10(CPS) [Y]
- Residual_Log: 로그 잔차 (Log10_Exp - Log10_Fit) [Y]
================================================================================
"""
        zf.writestr("Origin_Quick_Start_Guide.txt", guide_text.encode("utf-8-sig"))

    return zip_buf.getvalue()
