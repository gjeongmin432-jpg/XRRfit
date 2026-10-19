@echo off
cd /d "%~dp0"
echo ========================================================
echo  Uploading code to GitHub: https://github.com/gjeongmin432-jpg/XRRfit
echo ========================================================
echo.
git push -u origin main
echo.
echo ========================================================
echo  Done! Please check the output above.
echo ========================================================
pause
