@echo off
chcp 65001 > nul
title XRR 자동 피팅 스튜디오 (XRR Auto-Fitting Studio)
echo ======================================================================
echo    XRR 자동 피팅 스튜디오를 실행합니다 (Streamlit 데스크톱 모드)...
echo ======================================================================
echo.
python -m streamlit run app.py --browser.gatherUsageStats false
pause
