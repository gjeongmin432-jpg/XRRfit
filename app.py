"""
app.py
======
XRR 자동 피팅 스튜디오 (XRR Auto-Fitting Studio)
X-선 반사율(XRR) 자동 데이터 파싱, 임계각 및 q_z FFT 박막 두께 분석,
멀티 물리 엔진(Parratt + Nevot-Croce / Abeles) 피팅, 박막 물성치 도출 및 시각화 시스템.
기본 언어: 한국어 (Korean)
"""

import os
import sys
import io
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
import streamlit as st

# Matplotlib 한글 폰트 설정 (Windows, Linux, Mac, Streamlit Cloud 완벽 호환)
import matplotlib.font_manager as fm

def setup_matplotlib_korean_font():
    bundled_font = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".streamlit", "fonts", "NanumGothic.ttf")
    if os.path.exists(bundled_font):
        try:
            fm.fontManager.addfont(bundled_font)
            prop = fm.FontProperties(fname=bundled_font)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            pass

    linux_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
    ]
    for p in linux_paths:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                prop = fm.FontProperties(fname=p)
                plt.rcParams['font.family'] = prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                return
            except Exception:
                pass

    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in ['NanumGothic', 'Malgun Gothic', 'Noto Sans CJK KR', 'AppleGothic', 'Gulim']:
        if font_name in available:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            return

    plt.rcParams['axes.unicode_minus'] = False

setup_matplotlib_korean_font()

import importlib
import xrr_parser
import xrr_materials
import xrr_features
import xrr_engine
import xrr_optimizer

importlib.reload(xrr_parser)
importlib.reload(xrr_materials)
importlib.reload(xrr_features)
importlib.reload(xrr_engine)
importlib.reload(xrr_optimizer)

from xrr_parser import load_xrr_file, XRRData
from xrr_materials import (
    calculate_optical_constants,
    STANDARD_MATERIALS,
    get_default_materials_list
)
from xrr_features import (
    analyze_xrr_features,
    XRRFeatureReport
)
from xrr_engine import (
    Layer,
    LayerStack,
    simulate_parratt,
    simulate_abeles,
    benchmark_engines,
    calculate_fom
)
from xrr_optimizer import (
    fit_xrr,
    explore_native_oxide,
    FitResult,
    FitProgress
)

# ----------------------------------------------------------------------
# 페이지 설정 및 UI 스타일링
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="XRR 자동 피팅 스튜디오",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (과학용 데스크톱 스타일)
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.4rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #2563EB;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-winner {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-ai {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# 세션 상태 초기화
# ----------------------------------------------------------------------

if "xrr_data" not in st.session_state:
    st.session_state.xrr_data = None
if "feature_report" not in st.session_state:
    st.session_state.feature_report = None
if "fit_result" not in st.session_state:
    st.session_state.fit_result = None
if "layers_config" not in st.session_state:
    # 기본 박막 스택: 단일 박막 (두께 180 A) / Si 기판 (현실적 표면 거칠기)
    st.session_state.layers_config = [
        {"name": "주 박막 (Film)", "formula": "SnO", "thickness": 180.0, "roughness": 1.8, "density": 6.45}
    ]
if "substrate_config" not in st.session_state:
    st.session_state.substrate_config = {
        "name": "Si 기판", "formula": "Si", "roughness": 1.0, "density": 2.33
    }


# ----------------------------------------------------------------------
# 연속 깊이 프로파일 (Depth Profile) 계산 함수
# ----------------------------------------------------------------------

def calculate_depth_profile(stack: LayerStack, num_points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    계면 거칠기(sigma)에 따른 오차 함수(erf) 전이를 반영하여
    깊이 z에 따른 연속 질량 밀도 및 SLD 프로파일을 산출합니다.
    """
    total_d = sum(l.thickness for l in stack.layers)
    max_z = total_d + 60.0  # 기판 내부 깊이
    min_z = -30.0           # 진공/공기 영역

    z = np.linspace(min_z, max_z, num_points)
    density_profile = np.zeros_like(z)
    sld_profile = np.zeros_like(z)

    layers_info = []

    # 공기/진공 (Ambient)
    layers_info.append({
        "density": stack.ambient.density,
        "sld": calculate_optical_constants(stack.ambient.formula, stack.ambient.density).sld_real_A2 * 1e6,
        "roughness": 0.0
    })

    for l in stack.layers:
        props = calculate_optical_constants(l.formula, l.density, stack.wavelength_angstrom)
        layers_info.append({
            "density": l.density,
            "sld": props.sld_real_A2 * 1e6,
            "roughness": max(l.roughness, 0.5)
        })

    # 기판 (Substrate)
    sub_props = calculate_optical_constants(stack.substrate.formula, stack.substrate.density, stack.wavelength_angstrom)
    layers_info.append({
        "density": stack.substrate.density,
        "sld": sub_props.sld_real_A2 * 1e6,
        "roughness": max(stack.substrate.roughness, 0.5)
    })

    density_profile[:] = layers_info[0]["density"]
    sld_profile[:] = layers_info[0]["sld"]

    # 표면 (Ambient -> Layer 1)
    if len(stack.layers) > 0:
        sig0 = max(stack.layers[0].roughness, 0.5)
        step = 0.5 * (1.0 + erf(z / (np.sqrt(2.0) * sig0)))
        density_profile = density_profile + (layers_info[1]["density"] - layers_info[0]["density"]) * step
        sld_profile = sld_profile + (layers_info[1]["sld"] - layers_info[0]["sld"]) * step

    # 박막 층 간 전이
    curr_int_z = 0.0
    for i in range(1, len(stack.layers)):
        curr_int_z += stack.layers[i - 1].thickness
        sig = max(stack.layers[i].roughness, 0.5)
        step = 0.5 * (1.0 + erf((z - curr_int_z) / (np.sqrt(2.0) * sig)))
        density_profile = density_profile + (layers_info[i + 1]["density"] - layers_info[i]["density"]) * step
        sld_profile = sld_profile + (layers_info[i + 1]["sld"] - layers_info[i]["sld"]) * step

    # 최하부 박막 -> 기판 전이
    curr_int_z += stack.layers[-1].thickness if len(stack.layers) > 0 else 0.0
    sub_sig = max(stack.substrate.roughness, 0.5)
    step = 0.5 * (1.0 + erf((z - curr_int_z) / (np.sqrt(2.0) * sub_sig)))
    density_profile = density_profile + (layers_info[-1]["density"] - layers_info[-2]["density"]) * step
    sld_profile = sld_profile + (layers_info[-1]["sld"] - layers_info[-2]["sld"]) * step

    return z, density_profile, sld_profile


# ----------------------------------------------------------------------
# 사이드바 컨트롤 패널
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ XRR 데이터 및 시스템 제어")

    graph_lang = st.radio(
        "🌐 그래프 라벨 언어 (Plot Language)",
        ["한국어 (Korean)", "English (학술 논문용)"],
        horizontal=True,
        help="학술 논문 및 해외 저널 투고 시 'English'를 선택하면 그래프 제목, 축 라벨, 범례가 영문으로 표시됩니다."
    )
    is_eng = "English" in graph_lang

    # 1. 파일 입력
    st.markdown("#### 1. 원시 데이터 입력 (Data Input)")
    uploaded_file = st.file_uploader(
        "Panalytical .xrdml 또는 CSV/TXT 파일 업로드",
        type=["xrdml", "csv", "txt", "dat"],
        help="Malvern Panalytical사의 XML 기반 .xrdml 또는 2Theta/q vs Intensity 텍스트 파일 지원"
    )

    col_demo, col_clear = st.columns(2)
    with col_demo:
        if st.button("📥 데모 .xrdml 로드", use_container_width=True):
            if os.path.exists("sample_sno_film.xrdml"):
                with open("sample_sno_film.xrdml", "rb") as f:
                    content = f.read()
                data_obj = load_xrr_file(content, filename="sample_sno_film.xrdml")
                feat_obj = analyze_xrr_features(data_obj)
                st.session_state.xrr_data = data_obj
                st.session_state.feature_report = feat_obj
                st.session_state.fit_result = None
                if len(st.session_state.layers_config) > 0 and feat_obj.fft_thickness_angstrom > 20.0:
                    st.session_state.layers_config[0]["thickness"] = float(round(feat_obj.fft_thickness_angstrom, 1))
                st.success("데모 SnO/SiO2/Si (.xrdml) 로드 완료!")
            else:
                st.error("데모 파일이 없습니다. sample_generator.py를 먼저 실행하세요.")

    with col_clear:
        if st.button("🗑️ 초기화", use_container_width=True):
            st.session_state.xrr_data = None
            st.session_state.feature_report = None
            st.session_state.fit_result = None
            st.rerun()

    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        try:
            data_obj = load_xrr_file(content, filename=uploaded_file.name)
            feat_obj = analyze_xrr_features(data_obj)
            st.session_state.xrr_data = data_obj
            st.session_state.feature_report = feat_obj
            if len(st.session_state.layers_config) > 0 and feat_obj.fft_thickness_angstrom > 20.0:
                st.session_state.layers_config[0]["thickness"] = float(round(feat_obj.fft_thickness_angstrom, 1))
            st.success(f"로드 성공: {uploaded_file.name} ({len(data_obj.two_theta)}개 포인트, FFT 두께: {feat_obj.fft_thickness_nm:.1f} nm)")
        except Exception as e:
            st.error(f"파일 파싱 오류: {str(e)}")

    st.markdown("---")

    # 2. Gemini AI 연동
    st.markdown("#### 2. Gemini AI 지원 (선택 사항)")
    api_key_input = st.text_input(
        "Gemini API 키 (선택 사항)",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="측정된 XRR 곡선의 형상을 해석하여 물리적으로 타당한 초기 탐색 범위를 제안받습니다. 비워둘 경우 내장된 박막 물리 휴리스틱 알고리즘으로 자동 전환됩니다."
    )
    if api_key_input:
        st.markdown('<span class="badge-ai">🤖 Gemini API 연동 활성화</span>', unsafe_allow_html=True)
    else:
        st.caption("ℹ️ 로컬 박막 물리 및 FFT 휴리스틱 엔진 동작 중")

    st.markdown("---")

    # 3. 박막 적층 구조 설정
    st.markdown("#### 3. 박막 적층 구조 설정 (상부 → 하부)")
    
    # 기판 (Substrate)
    with st.expander("하부 기판 (Substrate)", expanded=True):
        sub_formula = st.selectbox(
            "기판 재료 선택",
            options=["Si", "Ge", "GaAs", "Al2O3", "Glass", "직접 입력(Custom)"],
            index=0
        )
        if sub_formula == "직접 입력(Custom)":
            sub_formula = st.text_input("기판 화학식", "Si")
        sub_density = st.number_input(
            "공칭 밀도 (g/cm³)",
            value=STANDARD_MATERIALS.get(sub_formula, 2.33),
            step=0.05
        )
        sub_rough = st.number_input("초기 계면 거칠기 (Å)", value=1.0, min_value=0.05, max_value=4.0, step=0.1)
        st.session_state.substrate_config = {
            "name": f"{sub_formula} 기판",
            "formula": sub_formula,
            "roughness": sub_rough,
            "density": sub_density
        }

    # 박막 레이어 (Film Layers)
    st.markdown("**박막 레이어 (Film Layers)**")
    materials_list = get_default_materials_list()

    num_layers = len(st.session_state.layers_config)
    for idx in range(num_layers):
        l = st.session_state.layers_config[idx]
        with st.expander(f"레이어 {idx+1}: {l['name']}", expanded=(idx == 0)):
            c1, c2 = st.columns(2)
            with c1:
                new_name = st.text_input("레이어 이름", value=l["name"], key=f"name_{idx}")
                mat_choice = st.selectbox(
                    "화학식(Formula)",
                    options=materials_list + ["직접 입력"],
                    index=materials_list.index(l["formula"]) if l["formula"] in materials_list else len(materials_list),
                    key=f"mat_{idx}"
                )
                if mat_choice == "직접 입력":
                    mat_choice = st.text_input("사용자 정의 화학식", value=l["formula"], key=f"cmat_{idx}")
            with c2:
                th_val = st.number_input("두께 (Å)", value=float(l["thickness"]), step=5.0, key=f"th_{idx}")
                rg_val = st.number_input("거칠기 (Å)", value=float(l["roughness"]), min_value=0.05, max_value=6.0, step=0.1, key=f"rg_{idx}")
                dn_val = st.number_input("밀도 (g/cm³)", value=float(l["density"]), step=0.1, key=f"dn_{idx}")

            st.session_state.layers_config[idx] = {
                "name": new_name,
                "formula": mat_choice,
                "thickness": th_val,
                "roughness": rg_val,
                "density": dn_val
            }

    c_add, c_del = st.columns(2)
    with c_add:
        if st.button("➕ 층 추가", use_container_width=True):
            st.session_state.layers_config.insert(0, {
                "name": f"레이어 {num_layers+1}", "formula": "TiO2", "thickness": 50.0, "roughness": 1.5, "density": 4.23
            })
            st.rerun()
    with c_del:
        if num_layers > 1 and st.button("➖ 최상층 삭제", use_container_width=True):
            st.session_state.layers_config.pop(0)
            st.rerun()

    auto_surface_layer = st.checkbox(
        "✨ 최표면 자연 변질/산화층(Surface Layer) 자동 분할 모델링",
        value=True,
        help="대기 노출로 인해 형성되는 최표면 1~2 nm 산화/변질층을 자동 모델링하여 1차 Fringe 피크의 높이와 저각 감쇠 곡선의 일치도를 논문 수준(R < 0.5%, χ² < 1.0)으로 끌어올립니다."
    )

    auto_explore_ox = st.checkbox(
        "자연 산화막(Native Oxide) 자동 탐색 비교",
        value=False,
        help="자연 산화막 포함/미포함 모델을 각각 피팅하여 오차율 개선 여부를 통계적으로 비교 판별합니다."
    )

    st.markdown("---")

    # 4. 시편 규격 및 X선 광학계 보정 (Footprint & Optics)
    st.markdown("#### 4. 시편 규격 및 광학계 보정 (Footprint & Optics)")
    detected_slit = st.session_state.xrr_data.incident_slit_mm if st.session_state.xrr_data else None
    default_slit = float(detected_slit) if detected_slit is not None else 0.100

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        sample_len_val = st.number_input(
            "시편 길이 (mm)",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="저각도 빔 스필오버(Footprint factor) 보정을 위한 시편의 길이입니다. 표준 1인치 웨이퍼 기준 25 mm입니다."
        )
    with col_opt2:
        beam_slit_val = st.number_input(
            "입사 빔 슬릿 폭 (mm)",
            min_value=0.01,
            max_value=2.0,
            value=default_slit,
            step=0.02,
            format="%.3f",
            help="X선 빔의 수직 입사 폭(Incident Slit)입니다. 파일 헤더에서 자동으로 추출됩니다."
        )

    # 5. 피팅 각도 범위 설정
    st.markdown("#### 5. 유효 피팅 각도 범위 설정 (2θ)")
    feat_rep: Optional[XRRFeatureReport] = st.session_state.feature_report
    def_start = float(round(feat_rep.suggested_start_2theta, 3)) if feat_rep else 0.35

    # 5차 Fringe 각도 계산 (논문 게재용 컷오프)
    f5_cutoff = None
    if feat_rep and len(feat_rep.fringes) >= 5:
        f5_pos = feat_rep.fringes[4].two_theta
        f5_cutoff = float(round(f5_pos + feat_rep.delta_theta_deg, 2))

    paper_mode = st.checkbox(
        "📄 논문 모드 (5차 Fringe까지만 집중 피팅)",
        value=True,
        help="논문 게재 표준에 맞추어 전반사 고원 정점(0.48°)부터 5번째 Kiessig Fringe 직후까지만 피팅을 진행하여 1~5차 구간 일치도를 극대화(R < 0.5%)합니다."
    )

    col_opt_chk1, col_opt_chk2 = st.columns(2)
    with col_opt_chk1:
        auto_fit_sample_len = st.checkbox(
            "시편 길이 자동 피팅",
            value=False if paper_mode else True,
            help="시편 길이를 최적화 파라미터로 포함합니다. 논문 모드(0.48° 이상)에서는 스필오버가 이미 해소되어 자동 비활성화됩니다."
        )
    with col_opt_chk2:
        enable_smearing = st.checkbox(
            "기기 분해능 스미어링",
            value=True,
            help="슬릿 발산각에 의한 가우시안 빔 스미어링을 적용하여 Kiessig 진동 골의 깊이를 실측치와 일치시킵니다."
        )

    if paper_mode:
        def_start = 0.48
        if f5_cutoff is not None:
            def_end = f5_cutoff
        else:
            def_end = 3.0
    else:
        def_start = float(round(feat_rep.suggested_start_2theta, 3)) if feat_rep else 0.35
        def_end = float(round(feat_rep.suggested_end_2theta, 2)) if feat_rep else 6.0

    col_ang1, col_ang2 = st.columns(2)
    with col_ang1:
        fit_start_2th = st.number_input(
            "피팅 시작 2θ (°)",
            min_value=0.05,
            max_value=3.0,
            value=def_start,
            step=0.02 if paper_mode else 0.05,
            key=f"start_2th_{paper_mode}",
            help="피팅 시작 각도입니다. 논문 모드에서는 0.48°부터 시작하여 스필오버 왜곡을 원천 배제하고 1~5차 Fringe 오버랩을 극대화합니다."
        )
    with col_ang2:
        fit_end_2th = st.number_input(
            "피팅 종료 2θ (°)",
            min_value=1.0,
            max_value=12.0,
            value=def_end,
            step=0.2 if paper_mode else 0.5,
            key=f"end_2th_{paper_mode}",
            help="피팅 데이터 포함 범위의 고각 컷오프입니다. 논문 모드 활성화 시 5번째 Fringe 직후로 자동 맞춰집니다."
        )



    st.markdown("---")

    # 6. 피팅 엔진 및 최적화 제어
    st.markdown("#### 6. 멀티 물리 엔진 및 최적화 설정")
    engine_sel = st.selectbox(
        "계산 엔진 선택",
        ["자동 (둘 다 계산 후 최저 오차 채택)", "엔진 A (Parratt + Nevot-Croce)", "엔진 B (Abeles Transfer Matrix)"]
    )
    max_gens = st.slider("전역 차분 진화(DE) 최대 세대 수", min_value=10, max_value=60, value=25, step=5)
    pop_size = st.slider("군집 크기 (Population Size)", min_value=8, max_value=20, value=10, step=2)

    run_fitting_btn = st.button("🚀 자동 피팅 시작", type="primary", use_container_width=True)


# ----------------------------------------------------------------------
# 메인 헤더
# ----------------------------------------------------------------------

st.markdown('<div class="main-title">🔬 XRR 자동 피팅 스튜디오</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">X-선 반사율 자동 분석, 임계각 검출, 멀티 물리 엔진(Parratt / Abelès) 시뮬레이션 및 박막 물성치 도출 시스템</div>', unsafe_allow_html=True)

if st.session_state.xrr_data is None:
    st.info("👋 환영합니다! 사이드바에서 Malvern Panalytical `.xrdml` (또는 CSV/TXT) 파일을 업로드하거나, **'📥 데모 .xrdml 로드'** 버튼을 클릭하여 바로 시작하세요.")
    st.stop()
    if not st.runtime.exists():
        sys.exit(0)

data: XRRData = st.session_state.xrr_data
features: XRRFeatureReport = st.session_state.feature_report

# LayerStack 객체 구성
ambient_layer = Layer("공기", "Air", 0.0, 0.0, 0.0012)
film_layers = [
    Layer(l["name"], l["formula"], float(l["thickness"]), float(l["roughness"]), float(l["density"]))
    for l in st.session_state.layers_config
]
sub_cfg = st.session_state.substrate_config
substrate_layer = Layer(sub_cfg["name"], sub_cfg["formula"], 0.0, float(sub_cfg["roughness"]), float(sub_cfg["density"]))

current_stack = LayerStack(
    ambient=ambient_layer,
    layers=film_layers,
    substrate=substrate_layer,
    wavelength_angstrom=data.wavelength_angstrom
)
current_stack.update_all_optical_constants()


# ----------------------------------------------------------------------
# 피팅 실행 제어
# ----------------------------------------------------------------------

if run_fitting_btn:
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_ui_progress(prog: FitProgress):
        pct = min(1.0, max(0.0, prog.iteration / max(prog.max_iterations, 1)))
        progress_bar.progress(pct)
        status_text.markdown(f"**최적화 진행 중**: {prog.message}")

    try:
        engine_mode = "Auto" if "자동" in engine_sel else ("Parratt" if "Parratt" in engine_sel else "Abeles")
        
        with st.spinner("차분 진화(DE) 전역 탐색 및 풋프린트/스택 정밀화 수행 중..."):
            fit_res = fit_xrr(
                data=data,
                initial_stack=current_stack,
                feature_report=features,
                api_key=api_key_input if api_key_input else None,
                engine_choice=engine_mode,
                start_two_theta=fit_start_2th,
                end_two_theta=fit_end_2th,
                sample_length_mm=float(sample_len_val),
                beam_width_mm=float(beam_slit_val),
                auto_fit_sample_length=auto_fit_sample_len,
                resolution_fwhm_deg=0.030 if enable_smearing else None,
                auto_surface_layer=auto_surface_layer,
                max_generations=max_gens,
                popsize=pop_size,
                progress_callback=update_ui_progress
            )
            st.session_state.fit_result = fit_res

        progress_bar.progress(1.0)
        status_text.success(f"피팅 완료! 최적 채택 엔진: {fit_res.best_engine} (R-팩터 오차율: {fit_res.r_factor:.2%})")

        if auto_explore_ox:
            with st.spinner("자연 산화막(Native Oxide) 자동 비교 분석 중..."):
                _, _, rec_msg = explore_native_oxide(
                    data, current_stack, features,
                    api_key=api_key_input if api_key_input else None,
                    start_two_theta=fit_start_2th,
                    end_two_theta=fit_end_2th
                )
                st.info(f"💡 산화막 탐색 판정: {rec_msg}")

        st.balloons()
    except Exception as e:
        st.error(f"피팅 실행 중 오류 발생: {str(e)}")


# ----------------------------------------------------------------------
# 탭 구성
# ----------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 1. 원시 데이터 및 특성 검출",
    "🎯 2. 멀티 엔진 피팅 및 오버랩 정합성",
    "📐 3. 도출된 박막 물성치 및 깊이 프로파일"
])


# ----------------------------------------------------------------------
# [탭 1] 원시 데이터 및 물리 특성 검출
# ----------------------------------------------------------------------

with tab1:
    st.markdown("### 📈 측정된 반사율 프로파일 및 자동 검출 물리 특성")

    # 핵심 메트릭 카드
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            label="전반사 임계각 (θ_c)",
            value=f"{features.critical_angle_theta_deg:.4f}°",
            help="전반사가 종료되고 반사율이 급격히 감쇠하기 시작하는 입사각입니다."
        )
    with m2:
        st.metric(
            label="임계각 2θ_c",
            value=f"{features.critical_angle_2theta_deg:.4f}°"
        )
    with m3:
        st.metric(
            label="표면 전자밀도",
            value=f"{features.electron_density_e_A3:.3f} e/Å³",
            help="θ_c로부터 직접 역산출된 표면의 평균 전자밀도입니다: rho_e = pi * theta_c^2 / (lambda^2 * r_e)."
        )
    with m4:
        st.metric(
            label="Kiessig 간섭 주기 (Δ2θ)",
            value=f"{features.delta_theta_deg * 2.0:.4f}°",
            help="검출된 Kiessig Fringe 간섭 무늬의 평균 진동 주기입니다."
        )
    with m5:
        st.metric(
            label="q_z FFT 분석 총 두께",
            value=f"{features.fft_thickness_nm:.2f} nm",
            delta=f"{features.fft_thickness_angstrom:.1f} Å",
            help="q_z-공간 고속 푸리에 변환(FFT) 파워 스펙트럼에서 도출된 잡음에 강인한 고유 박막 두께입니다."
        )

    # Matplotlib 그래프: 원시 곡선 + theta_c + Kiessig fringes + 피팅 구간 표시
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    # 상단: Log Intensity
    lbl_raw = "Measured XRR Reflectivity (CPS)" if is_eng else "실측 XRR 반사율 (CPS)"
    lbl_span = f"Valid Fitting Range [{fit_start_2th:.2f}° ~ {fit_end_2th:.2f}°]" if is_eng else f"유효 피팅 구간 [{fit_start_2th:.2f}° ~ {fit_end_2th:.2f}°]"
    lbl_crit = (
        rf"Critical Angle $\theta_c$ = {features.critical_angle_theta_deg:.3f}° ($2\theta$={features.critical_angle_2theta_deg:.3f}°)"
        if is_eng else
        rf"임계각 $\theta_c$ = {features.critical_angle_theta_deg:.3f}° ($2\theta$={features.critical_angle_2theta_deg:.3f}°)"
    )
    lbl_y1 = r"Reflectivity Intensity ($\log_{10}$ CPS)" if is_eng else r"반사율 강도 ($\log_{10}$ CPS)"
    lbl_title1 = "XRR Reflectivity Profile & Automated Feature Detection" if is_eng else "XRR 반사율 곡선 및 핵심 물리 특성 자동 검출"
    lbl_dlog = r"1st Derivative $\mathrm{d}(\log_{10} I)/\mathrm{d}(2\theta)$" if is_eng else r"1차 미분 $\mathrm{d}(\log_{10} I)/\mathrm{d}(2\theta)$"
    lbl_x = r"$2\theta$ Angle (deg)" if is_eng else r"$2\theta$ 각도 (도, deg)"
    lbl_dy = "Derivative" if is_eng else "미분값"

    ax1.plot(data.two_theta, data.intensity, color="#2563EB", lw=1.5, label=lbl_raw)
    ax1.set_yscale("log")
    ax1.set_ylabel(lbl_y1, fontsize=11, fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.4)

    # 피팅 권장 유효 구간 음영 표기
    ax1.axvspan(fit_start_2th, fit_end_2th, color="#ECFDF5", alpha=0.6, label=lbl_span)

    # 임계각 표기
    ax1.axvline(features.critical_angle_2theta_deg, color="#DC2626", ls="--", lw=1.8, label=lbl_crit)

    # Kiessig Fringe 피크 마커
    for f in features.fringes:
        ax1.plot(f.two_theta, f.intensity, "ro", markersize=5)
        if f.order <= 6:
            ax1.annotate(
                f"#{f.order}",
                (f.two_theta, f.intensity),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#B91C1C"
            )

    ax1.legend(loc="upper right", frameon=True)
    ax1.set_title(lbl_title1, fontsize=13, fontweight="bold")

    # 하단: 1차 미분 d(log I)/d(2Theta)
    log_i_smooth = np.log10(data.intensity)
    d_log = np.gradient(log_i_smooth, data.two_theta)
    ax2.plot(data.two_theta, d_log, color="#059669", lw=1.2, label=lbl_dlog)
    ax2.axvline(features.critical_angle_2theta_deg, color="#DC2626", ls="--", lw=1.5)
    ax2.set_xlabel(lbl_x, fontsize=11, fontweight="bold")
    ax2.set_ylabel(lbl_dy, fontsize=11, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # q_z FFT 박막 두께 파워 스펙트럼 그래프
    if len(features.fft_thickness_grid) > 0:
        fft_hdr = "🔬 Spatial Frequency (q_z) FFT Thickness Power Spectrum" if is_eng else "🔬 q_z 공간 고속 푸리에 변환 (FFT) 박막 두께 스펙트럼"
        st.markdown(f"#### {fft_hdr}")
        fig_fft, ax_fft = plt.subplots(figsize=(10, 3.8))
        lbl_fft_pwr = "FFT Power Spectrum" if is_eng else "FFT 파워 스펙트럼"
        lbl_fft_pk = f"Dominant Thickness Peak = {features.fft_thickness_nm:.2f} nm ({features.fft_thickness_angstrom:.1f} Å)" if is_eng else f"지배 박막 두께 피크 = {features.fft_thickness_nm:.2f} nm ({features.fft_thickness_angstrom:.1f} Å)"
        lbl_fft_x = "Film Thickness d (nm)" if is_eng else "박막 두께 d (nm)"
        lbl_fft_y = "Normalized Power (a.u.)" if is_eng else "정규화된 파워 (a.u.)"
        lbl_fft_title = "Kiessig Fringe Spatial Frequency (q_z) Fourier Transform" if is_eng else "Kiessig Fringe 간섭 무늬의 공간 주파수(q_z) 푸리에 변환 및 고유 박막 두께 검출"

        ax_fft.plot(features.fft_thickness_grid / 10.0, features.fft_power_spectrum, color="#7C3AED", lw=2.0, label=lbl_fft_pwr)
        ax_fft.axvline(features.fft_thickness_nm, color="#DC2626", ls="--", lw=1.8, label=lbl_fft_pk)
        ax_fft.plot(features.fft_thickness_nm, 1.0, "r*", markersize=12)
        ax_fft.set_xlabel(lbl_fft_x, fontsize=11, fontweight="bold")
        ax_fft.set_ylabel(lbl_fft_y, fontsize=11, fontweight="bold")
        ax_fft.set_title(lbl_fft_title, fontsize=12, fontweight="bold")
        ax_fft.set_xlim(2.0, min(80.0, max(features.fft_thickness_grid / 10.0)))
        ax_fft.grid(True, ls="--", alpha=0.4)
        ax_fft.legend(loc="upper right")
        plt.tight_layout()
        st.pyplot(fig_fft)
        plt.close(fig_fft)

    # Fringe 상세 테이블
    with st.expander(f"📋 검출된 Kiessig Fringe 상세 목록 ({features.num_fringes_detected}개 피크 식별됨)"):
        fringe_rows = []
        for f in features.fringes:
            fringe_rows.append({
                "간섭 차수 #": f.order,
                "2-Theta (도)": f"{f.two_theta:.3f}",
                "Theta (도)": f"{f.theta:.3f}",
                "산란 벡터 q_z (Å⁻¹)": f"{f.q_z:.4f}",
                "반사율 강도 (CPS)": f"{f.intensity:.2e}"
            })
        st.dataframe(pd.DataFrame(fringe_rows), use_container_width=True)


# ----------------------------------------------------------------------
# [탭 2] 멀티 엔진 피팅 및 오버랩 정합성
# ----------------------------------------------------------------------

with tab2:
    st.markdown("### 🎯 멀티 물리 엔진 반사율 곡선 피팅 및 오버랩 정합성 검증")

    if st.session_state.fit_result is None:
        st.warning("⚠️ 아직 피팅이 수행되지 않았습니다. 사이드바의 **'🚀 자동 피팅 시작'** 버튼을 클릭하여 멀티 엔진 최적화를 실행하세요!")
        
        # 피팅 전 초기 시뮬레이션 곡선 비교 (유효 피팅 구간 기준)
        st.markdown("#### 초기 적층 모델 시뮬레이션 미리보기 (최적화 전)")
        mask_prev = (data.two_theta >= fit_start_2th) & (data.two_theta <= fit_end_2th)
        th_prev = data.theta[mask_prev]
        two_th_prev = data.two_theta[mask_prev]
        int_prev = data.intensity[mask_prev]

        sim_init_a = simulate_parratt(th_prev, current_stack, scale=features.plateau_intensity, background=1.0)
        sim_init_b = simulate_abeles(th_prev, current_stack, scale=features.plateau_intensity, background=1.0)
        fom_a = calculate_fom(int_prev, sim_init_a)
        fom_b = calculate_fom(int_prev, sim_init_b)

        f_prev, ax_prev = plt.subplots(figsize=(10, 4.5))
        ax_prev.plot(two_th_prev, int_prev, "o", color="#9CA3AF", markersize=3, alpha=0.5, label="실측 데이터 (피팅 구간)")
        ax_prev.plot(two_th_prev, sim_init_a, color="#2563EB", lw=1.8, label=f"엔진 A (Parratt, 초기 오차={fom_a[1]:.1%})")
        ax_prev.plot(two_th_prev, sim_init_b, color="#D97706", lw=1.8, ls="--", label=f"엔진 B (Abeles, 초기 오차={fom_b[1]:.1%})")
        ax_prev.set_yscale("log")
        ax_prev.set_xlabel(r"$2\theta$ 각도 (도, deg)")
        ax_prev.set_ylabel("반사율 강도 (CPS)")
        ax_prev.legend()
        ax_prev.grid(True, ls="--", alpha=0.4)
        st.pyplot(f_prev)
        plt.close(f_prev)
    else:
        res: FitResult = st.session_state.fit_result

        # 최적 엔진 및 지표 요약
        c_win1, c_win2, c_win3, c_win4, c_win5 = st.columns(5)
        with c_win1:
            st.markdown(f'<div class="metric-box">🏆 <b>최종 채택 최적 엔진</b><br><span class="badge-winner">{res.best_engine}</span></div>', unsafe_allow_html=True)
        with c_win2:
            st.markdown(f'<div class="metric-box">📉 <b>로그 평균제곱오차 (Log-MSE)</b><br><h3>{res.log_mse:.5f}</h3></div>', unsafe_allow_html=True)
        with c_win3:
            st.markdown(f'<div class="metric-box">🎯 <b>R-팩터 (오차율)</b><br><h3>{res.r_factor:.2%}</h3></div>', unsafe_allow_html=True)
        with c_win4:
            chi2_color = "#16A34A" if res.reduced_chi2 < 10 else "#DC2626"
            st.markdown(f'<div class="metric-box">📊 <b>축소 카이제곱 (Reduced χ²)</b><br><h3 style="color:{chi2_color}">{res.reduced_chi2:.3f}</h3></div>', unsafe_allow_html=True)
        with c_win5:
            l_val_str = f"{res.sample_length_mm:.1f} mm" if res.sample_length_mm else "미적용"
            st.markdown(f'<div class="metric-box">📐 <b>시편 규격 (Footprint)</b><br><h3>{l_val_str}</h3></div>', unsafe_allow_html=True)

        if res.r_factor < 0.01 and res.reduced_chi2 < 2.0:
            st.markdown(f"""
            <div style="background-color: #ECFDF5; border: 1.5px solid #10B981; border-radius: 8px; padding: 12px 16px; margin: 12px 0;">
                <b style="color: #065F46; font-size: 1.05rem;">🏆 학술 논문 게재 최우수 정밀도 달성!</b>
                <span style="color: #047857; margin-left: 8px;">(R-Factor = <b>{res.r_factor:.2%}</b>, Reduced χ² = <b>{res.reduced_chi2:.2f}</b>, Log-MSE = <b>{res.log_mse:.5f}</b>)</span>
                <br><small style="color: #047857;">전반사 임계각 에지부터 1~5차 Kiessig Fringe 피크와 골 전체가 오차 범위 내에서 완벽하게 일치합니다.</small>
            </div>
            """, unsafe_allow_html=True)

        if res.ai_used:
            st.markdown(f'<span class="badge-ai">🤖 Gemini AI 제안 파라미터 경계 적용됨</span>: {res.ai_rationale}', unsafe_allow_html=True)
        else:
            st.caption(f"ℹ️ {res.ai_rationale}")

        # 메인 듀얼 플롯: 실측값 vs 피팅값 오버레이 + 하단 잔차 플롯 (논문용 세밀한 선/마커 스타일)
        fig_fit, (ax_top, ax_res) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.8, 1]})

        lbl_exp = f"Experimental XRR Data (Range: {res.start_two_theta:.2f}° ~ {res.end_two_theta:.2f}°)" if is_eng else f"실측 XRR 데이터 (피팅 구간: {res.start_two_theta:.2f}° ~ {res.end_two_theta:.2f}°)"
        lbl_fit = f"Optimal Fitted Model ({res.best_engine})" if is_eng else f"최적 피팅 모델 ({res.best_engine})"
        lbl_y_fit = r"Reflectivity Intensity ($\log_{10}$ CPS)" if is_eng else r"반사율 강도 ($\log_{10}$ CPS)"
        lbl_title_fit = "Specular Reflectivity: Experimental Data vs. Best Multi-Physics Model" if is_eng else "경면 반사율: 실측 데이터 vs. 최적 멀티 물리 엔진 시뮬레이션 곡선"
        lbl_span_5th = "1~5th Fringe Focus Region (Publication Analysis)" if is_eng else "1~5th Fringe 집중 정합 구간 (논문 분석 유효 영역)"
        lbl_span_3rd = "Low-Angle Fringe Focus Region" if is_eng else "저각 Fringe 우선 정합 구간"
        lbl_res_y = r"Residual $\Delta\log_{10}(I)$" if is_eng else r"잔차 $\Delta\log_{10}(I)$"
        lbl_res_x = r"$2\theta$ Angle (deg)" if is_eng else r"$2\theta$ 각도 (도, deg)"

        # 상단: 오버레이 곡선 (마커 2.2pt, 모델 선폭 1.2pt로 세밀화)
        ax_top.plot(res.two_theta, res.experimental_intensity, "o", color="#6B7280", markersize=2.2, alpha=0.55, label=lbl_exp)
        ax_top.plot(res.two_theta, res.simulated_intensity, color="#DC2626", lw=1.2, label=lbl_fit)
        ax_top.set_yscale("log")
        ax_top.set_ylabel(lbl_y_fit, fontsize=11, fontweight="bold")
        ax_top.set_title(lbl_title_fit, fontsize=13, fontweight="bold")
        ax_top.grid(True, which="both", ls="--", alpha=0.4)

        # 5th Kiessig fringe 집중 가중치/논문 구간 강조 표시
        if len(features.fringes) >= 5:
            th_5th_2th = features.fringes[4].two_theta
            span_end = min(th_5th_2th * 1.08, res.end_two_theta)
            ax_top.axvspan(res.start_two_theta, span_end, color="#FEF3C7", alpha=0.30, label=lbl_span_5th)
        elif len(features.fringes) >= 3:
            th_3rd_2th = features.fringes[2].two_theta
            ax_top.axvspan(res.start_two_theta, min(th_3rd_2th * 1.1, res.end_two_theta), color="#FEF3C7", alpha=0.30, label=lbl_span_3rd)

        ax_top.legend(loc="upper right", frameon=True, fontsize=9.5)

        # 하단: 각도별 잔차(Residual) 플롯 (선폭 0.9pt로 슬림화)
        ax_res.plot(res.two_theta, res.residual_log, color="#2563EB", lw=0.9)
        ax_res.axhline(0.0, color="#111827", ls="-", lw=0.8)
        ax_res.axhline(0.1, color="#EF4444", ls=":", lw=0.8)
        ax_res.axhline(-0.1, color="#EF4444", ls=":", lw=0.8)
        ax_res.set_xlabel(lbl_res_x, fontsize=11, fontweight="bold")
        ax_res.set_ylabel(lbl_res_y, fontsize=11, fontweight="bold")
        ax_res.set_ylim(-0.25, 0.25)
        ax_res.grid(True, ls="--", alpha=0.4)


        plt.tight_layout()
        st.pyplot(fig_fit)
        plt.close(fig_fit)

        # 멀티 엔진 벤치마크 리더보드
        st.markdown("#### 🏁 멀티 물리 엔진 벤치마크 리더보드 (Engine Leaderboard)")
        bench_a, bench_b, winner_name = benchmark_engines(
            res.theta, res.experimental_intensity, res.best_stack,
            scale=res.scale, background=res.background, wavelength_angstrom=data.wavelength_angstrom,
            sample_length_mm=res.sample_length_mm, beam_width_mm=res.beam_width_mm,
            resolution_fwhm_deg=res.resolution_fwhm_deg if enable_smearing else None
        )


        leaderboard_data = [
            {
                "순위": "🥇 최적 엔진 (Winner)" if bench_a.log_mse <= bench_b.log_mse else "🥈 차순위",
                "물리 계산 엔진": bench_a.engine_name,
                "Log-MSE": f"{bench_a.log_mse:.5f}",
                "R-팩터 오차율 (%)": f"{bench_a.r_factor:.2%}",
                "Reduced χ²": f"{bench_a.reduced_chi2:.3f}",
                "연산 시간 (ms)": f"{bench_a.computation_time_ms:.1f}"
            },
            {
                "순위": "🥇 최적 엔진 (Winner)" if bench_b.log_mse < bench_a.log_mse else "🥈 차순위",
                "물리 계산 엔진": bench_b.engine_name,
                "Log-MSE": f"{bench_b.log_mse:.5f}",
                "R-팩터 오차율 (%)": f"{bench_b.r_factor:.2%}",
                "Reduced χ²": f"{bench_b.reduced_chi2:.3f}",
                "연산 시간 (ms)": f"{bench_b.computation_time_ms:.1f}"
            }
        ]
        st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True)


# ----------------------------------------------------------------------
# [탭 3] 도출된 물성치 및 깊이 프로파일
# ----------------------------------------------------------------------

with tab3:
    st.markdown("### 📐 도출된 박막 물리 물성치 (Thickness, Density, Roughness)")

    active_stack = st.session_state.fit_result.best_stack if st.session_state.fit_result else current_stack

    # 최종 파라미터 요약 표
    param_table = []
    for idx, l in enumerate(active_stack.layers):
        props = calculate_optical_constants(l.formula, l.density, active_stack.wavelength_angstrom)
        param_table.append({
            "레이어 구분": f"박막 {idx+1}: {l.name}",
            "화학식": l.formula,
            "두께 (Å)": f"{l.thickness:.2f}",
            "두께 (nm)": f"{l.thickness_nm:.3f}",
            "거칠기 (Å)": f"{l.roughness:.2f}",
            "질량 밀도 (g/cm³)": f"{l.density:.2f}",
            "전자 밀도 (e/Å³)": f"{props.electron_density_e_A3:.3f}",
            "SLD 실수부 (10⁻⁶ Å⁻²)": f"{props.sld_real_A2 * 1e6:.3f}",
            "δ (10⁻⁵)": f"{props.delta * 1e5:.3f}",
            "β (10⁻⁶)": f"{props.beta * 1e6:.3f}"
        })

    # 기판 행
    sub_l = active_stack.substrate
    sub_p = calculate_optical_constants(sub_l.formula, sub_l.density, active_stack.wavelength_angstrom)
    param_table.append({
        "레이어 구분": f"하부 기판: {sub_l.name}",
        "화학식": sub_l.formula,
        "두께 (Å)": "벌크 (반무한)",
        "두께 (nm)": "벌크 (반무한)",
        "거칠기 (Å)": f"{sub_l.roughness:.2f}",
        "질량 밀도 (g/cm³)": f"{sub_l.density:.2f}",
        "전자 밀도 (e/Å³)": f"{sub_p.electron_density_e_A3:.3f}",
        "SLD 실수부 (10⁻⁶ Å⁻²)": f"{sub_p.sld_real_A2 * 1e6:.3f}",
        "δ (10⁻⁵)": f"{sub_p.delta * 1e5:.3f}",
        "β (10⁻⁶)": f"{sub_p.beta * 1e6:.3f}"
    })

    df_params = pd.DataFrame(param_table)
    st.dataframe(df_params, use_container_width=True)

    if len(active_stack.layers) > 1:
        total_th_a = sum(l.thickness for l in active_stack.layers)
        st.info(f"📏 **총 박막 물리 두께 (Total Film Thickness)**: **{total_th_a:.2f} Å ({total_th_a/10.0:.3f} nm)** (최표면 변질/산화층 + 벌크 박막 합산)")

    # 깊이 프로파일 (SLD 및 질량 밀도 vs 깊이 z)
    st.markdown("#### 🌊 연속 깊이 프로파일 (계면 거칠기 Erf 전이 반영)")
    z_depth, rho_depth, sld_depth = calculate_depth_profile(active_stack)

    fig_depth, (ax_rho, ax_sld) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    lbl_rho_x = "Depth from Surface z (nm)" if is_eng else "표면으로부터의 깊이 z (nm)"
    lbl_rho_y = r"Mass Density $\rho$ ($\mathrm{g/cm}^3$)" if is_eng else r"질량 밀도 $\rho$ ($\mathrm{g/cm}^3$)"
    lbl_rho_title = r"Mass Density Depth Profile ($\rho$ vs $z$)" if is_eng else r"질량 밀도 깊이 프로파일 ($\rho$ vs $z$)"
    lbl_sld_x = "Depth from Surface z (nm)" if is_eng else "표면으로부터의 깊이 z (nm)"
    lbl_sld_y = r"Real SLD ($10^{-6}\ \mathrm{\AA}^{-2}$)" if is_eng else r"SLD 실수부 ($10^{-6}\ \mathrm{\AA}^{-2}$)"
    lbl_sld_title = "Scattering Length Density (SLD) Profile" if is_eng else "산란 길이 밀도(SLD) 프로파일"

    ax_rho.plot(z_depth / 10.0, rho_depth, color="#1E40AF", lw=2.0)
    ax_rho.set_xlabel(lbl_rho_x, fontsize=11, fontweight="bold")
    ax_rho.set_ylabel(lbl_rho_y, fontsize=11, fontweight="bold")
    ax_rho.set_title(lbl_rho_title, fontsize=12, fontweight="bold")
    ax_rho.grid(True, ls="--", alpha=0.4)

    ax_sld.plot(z_depth / 10.0, sld_depth, color="#B45309", lw=2.0)
    ax_sld.set_xlabel(lbl_sld_x, fontsize=11, fontweight="bold")
    ax_sld.set_ylabel(lbl_sld_y, fontsize=11, fontweight="bold")
    ax_sld.set_title(lbl_sld_title, fontsize=12, fontweight="bold")
    ax_sld.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig_depth)
    plt.close(fig_depth)

    # 결과 파일 및 리포트 내보내기
    st.markdown("---")
    st.markdown("#### 💾 분석 결과 및 성적서 내보내기")

    col_csv1, col_csv2, col_rep = st.columns(3)

    # 1. 파라미터 CSV 다운로드
    with col_csv1:
        csv_params = df_params.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 도출 파라미터 다운로드 (CSV)",
            data=csv_params,
            file_name="xrr_도출_박막물성치.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 2. 피팅 곡선 CSV 다운로드
    with col_csv2:
        if st.session_state.fit_result:
            fit_r: FitResult = st.session_state.fit_result
            df_curve = pd.DataFrame({
                "TwoTheta_deg": fit_r.two_theta,
                "Theta_deg": fit_r.theta,
                "q_z_A-1": fit_r.q_z,
                "실측강도_CPS": fit_r.experimental_intensity,
                "피팅강도_CPS": fit_r.simulated_intensity,
                "로그잔차_Residual": fit_r.residual_log
            })
            csv_curve = df_curve.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 피팅 곡선 데이터 다운로드 (CSV)",
                data=csv_curve,
                file_name="xrr_피팅_곡선데이터.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.button("📥 피팅 곡선 다운로드 (피팅 후 활성화)", disabled=True, use_container_width=True)

    # 3. 상세 분석 성적서 TXT 내보내기
    with col_rep:
        fit_r = st.session_state.fit_result
        best_eng = fit_r.best_engine if fit_r else "미수행"
        mse_str = f"{fit_r.log_mse:.5f}" if fit_r else "미수행"
        r_str = f"{fit_r.r_factor*100:.2f}%" if fit_r else "미수행"
        chi2_str = f"{fit_r.reduced_chi2:.3f}" if fit_r else "미수행"
        range_str = f"[{fit_r.start_two_theta:.3f}°, {fit_r.end_two_theta:.3f}°]" if fit_r else "미수행"

        report_text = f"""================================================================================
                    X-선 반사율(XRR) 자동 분석 및 피팅 성적서
================================================================================
생성 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}
원시 데이터 파일: {data.filename}
X-선 광원 파장: {data.wavelength_angstrom:.5f} Å ({data.wavelength_nm:.5f} nm, Cu K-alpha 1)

[1. 물리 특성 자동 검출 요약]
- 전반사 임계각 (theta_c): {features.critical_angle_theta_deg:.4f} deg (2Theta = {features.critical_angle_2theta_deg:.4f} deg)
- 표면 평균 전자밀도: {features.electron_density_e_A3:.3f} e/Å³
- q_z 공간 FFT 분석 고유 박막 두께: {features.fft_thickness_angstrom:.1f} Å ({features.fft_thickness_nm:.2f} nm)
- Kiessig Fringe 간섭 주기 (Delta 2Theta): {features.delta_theta_deg * 2.0:.4f} deg
- 식별된 간섭 피크 수: {features.num_fringes_detected} 개
- 권장 유효 피팅 구간: 2Theta [{features.suggested_start_2theta:.3f} deg, {features.suggested_end_2theta:.3f} deg]

[2. 멀티 물리 엔진 피팅 결과]
- 최적 채택 엔진: {best_eng}
- 피팅 적용 각도 범위: {range_str}
- 로그 평균제곱오차 (Log-MSE): {mse_str}
- 결정학적 R-팩터 (오차율): {r_str}
- 축소 카이제곱 (Reduced Chi2): {chi2_str}

[3. 최종 도출된 층별 박막 물성치 요약]
{df_params.to_string(index=False)}

================================================================================
"""
        st.download_button(
            "📄 분석 성적서 다운로드 (TXT)",
            data=report_text.encode("utf-8-sig"),
            file_name="xrr_분석성적서.txt",
            mime="text/plain",
            use_container_width=True
        )
