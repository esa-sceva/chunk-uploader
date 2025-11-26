@echo off
echo ========================================
echo Fixing Dependencies
echo ========================================
echo.

echo Upgrading timm to fix ImageNetInfo error...
pip install --upgrade "timm>=1.0.3"

echo.
echo Upgrading transformers...
pip install --upgrade "transformers>=4.36.0"

echo.
echo Upgrading sentence-transformers...
pip install --upgrade "sentence-transformers>=2.7.0"

echo.
echo ========================================
echo Done! Dependencies fixed.
echo ========================================
echo.
echo Now you can run:
echo   py main.py
echo.
pause

