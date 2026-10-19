# 🔬 XRR Auto-Fitting Studio (자동화 X-선 반사율 분석 및 멀티 물리 엔진 피팅 시스템)

본 프로그램은 **X-선 광학 및 박막 물리학(Thin-Film Physics)** 기반으로 개발된 자동화 데스크톱 솔루션입니다. Panalytical사의 `.xrdml` (XML 기반) 원시 데이터를 수동 조작 없이 입력받아 **임계각 분석, Kiessig Fringe 자동 검출, 적층 구조 최적화, 멀티 물리 엔진(Parratt + Nevot-Croce / Abelès Matrix) 시뮬레이션, Gemini AI + 차분 진화(Differential Evolution) 하이브리드 피팅, 그리고 물성치 도출 및 시각화**를 완벽하게 자동화합니다.

---

## 🌟 핵심 기능 요약

1. **원시 데이터 자동 파서 (`xrr_parser.py`)**:
   - Malvern Panalytical사의 `.xrdml` 파일의 XML 구조(스캔 각도, Counts, CPS, 계수 시간, 파장, 빔 감쇠기 보정 계수)를 자동으로 파싱.
   - $2\theta \to \theta$ 및 산란 벡터 $q_z = \frac{4\pi}{\lambda}\sin\theta$ 자동 계산.
   - CSV, TSV, DAT 등 표준 ASCII 파일 포맷도 유연하게 자동 감지 및 로드.

2. **임계각($\theta_c$) 및 Kiessig Fringe 자동 검출 (`xrr_features.py`)**:
   - $\log_{10}(I)$ 1차 미분(Savitzky-Golay 기반 $\frac{d\log I}{d\theta}$)의 급격한 변곡점을 추적하여 전반사 임계각($\theta_c$) 자동 검출.
   - $\theta_c$로부터 표면/기판의 평균 전자밀도 $\rho_e \approx \frac{\pi \theta_c^2}{\lambda^2 r_e}$ 역산출.
   - Kiessig Fringe(간섭 진동) 피크를 자동 검출하고, 3번째 진동 피크($\text{3rd fringe}$) 이상의 주기($\Delta\theta, \Delta q$)를 통해 박막 총 두께(Total Thickness)를 브래그 유사 공식 $d \approx \frac{2\pi}{\Delta q_z} \approx \frac{\lambda}{2\Delta\theta}$으로 1차 산출.

3. **재료 라이브러리 및 SLD 산출 (`xrr_materials.py`)**:
   - Cu $K\alpha_1$ 파장($1.54056\text{ \AA}$)에서의 원자 산란 인자($f_1, f_2$) 데이터베이스 내장.
   - 화학식(Formula)과 밀도로부터 굴절률 감쇠 계수($\delta, \beta$) 및 산란 길이 밀도(SLD) 자동 매핑.

4. **멀티 물리 엔진 앙상블 (`xrr_engine.py`)**:
   - **엔진 A**: NumPy 고속 벡터화 기반 Parratt 재귀 공식 + Névot-Croce 계면 거칠기 팩터($\exp(-2 k_{z,j} k_{z,j+1} \sigma_{j+1}^2)$).
   - **엔진 B**: Abelès $2\times 2$ 특성 전이 행렬(Transfer Matrix) 방법 + 계면 감쇠.
   - 피팅 후 각 엔진의 Log-MSE, Crystallographic R-factor, Reduced $\chi^2$를 벤치마킹하여 최적 엔진을 자동 선정 및 리포팅.

5. **AI + Differential Evolution 하이브리드 피팅 (`xrr_optimizer.py`)**:
   - **1단계 (Gemini AI 추론)**: 사용자의 Gemini API Key 입력 시, 측정 특성($\theta_c$, Kiessig Fringe 주기, 감쇠 기울기)과 타깃 스택을 AI에 전달하여 물리적으로 타당한 파라미터 탐색 범위(Bounds)와 초기값을 JSON으로 획득. (API 미입력 또는 오프라인 시 박막 물리 휴리스틱 엔진으로 자동 폴백)
   - **2단계 (전역 탐색)**: Differential Evolution(차분 진화 알고리즘)으로 국소 최솟값을 탈출하며 넓은 탐색 공간 최적화.
   - **3단계 (국소 정밀화)**: Nelder-Mead 심플렉스 알고리즘을 통한 초정밀 미세 수렴.
   - **저각도~중각도 가중치(Low-to-Mid q Weighting)**: 임계각 영역과 적어도 3번째 Kiessig fringe까지의 구간에 $4\times \sim 5\times$ 가중치를 집중 부여하여 메인 간섭 패턴의 완벽한 중첩(Overlap)을 보장.

6. **데스크톱 인터랙티브 GUI (`app.py`)**:
   - Streamlit 기반의 현대적이고 깔끔한 대시보드.
   - 실시간 진행 바, 오버레이 곡선 및 잔차($\Delta \log I$) 듀얼 플롯, 전자밀도/SLD 깊이 프로파일(Depth Profile), CSV/TXT 성적서 원클릭 다운로드 제공.

---

## 🚀 실행 방법 (Quick Start)

### 1. 원클릭 실행 (Windows)
프로젝트 폴더 내 `run_app.bat` 파일을 더블 클릭하면 브라우저에서 프로그램이 즉시 실행됩니다.

### 2. 터미널 수동 실행
```bash
# 필수 패키지 설치 (이미 설치되어 있을 경우 생략)
pip install -r requirements.txt

# 애플리케이션 시작
streamlit run app.py
```

---

## 🧪 제공되는 테스트 샘플 및 검증
- 시스템에는 Panalytical 실제 장비와 동일한 규격의 합성 데이터(`sample_sno_film.xrdml` 및 `sample_sno_film.csv`)가 포함되어 있습니다.
- 구조: $\text{Air} \to \text{Surface Contamination (1.2 nm)} \to \text{SnO Film (25.0 nm)} \to \text{SiO}_2\text{ Native Oxide (1.8 nm)} \to \text{Si Substrate}$
- 사이드바의 **'📥 Load Demo .xrdml'** 버튼을 클릭하여 즉시 기능을 시연할 수 있습니다.
- 자동화 파이프라인 검증 테스트 실행:
  ```bash
  python test_pipeline.py
  ```
