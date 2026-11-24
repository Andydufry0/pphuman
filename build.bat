@echo off
setlocal
REM ===== MSVC + Windows SDK environment (manual, avoids vcvars/reg dependency) =====
set "MSVC=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "SDKV=10.0.26100.0"
set "INCLUDE=%MSVC%\include;%SDK%\Include\%SDKV%\ucrt;%SDK%\Include\%SDKV%\shared;%SDK%\Include\%SDKV%\um;%SDK%\Include\%SDKV%\winrt;%SDK%\Include\%SDKV%\cppwinrt"
set "LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKV%\ucrt\x64;%SDK%\Lib\%SDKV%\um\x64"
set "PATH=%MSVC%\bin\HostX64\x64;%SDK%\bin\%SDKV%\x64;%PATH%"

cd /d E:\DeepEpic\PPHuman-Qt

echo [1/2] qmake...
C:\Qt\Qt5.14.2\5.14.2\msvc2017_64\bin\qmake.exe PPHuman-Qt.pro
if errorlevel 1 (echo qmake failed! & exit /b 1)

echo [2/2] nmake...
nmake -f Makefile.Release
if errorlevel 1 (echo build failed! & exit /b 1)

echo.
echo OK: release\PPHuman-Qt.exe
