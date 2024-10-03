@echo off


REM Получаем путь к текущему bat-файлу
set "filepath=%~dp0"

REM Извлекаем только последнюю часть пути (имя папки)
for %%A in ("%filepath%.") do set "foldername=%%~nxA"

echo -----------------------------------
echo Calculation for %foldername%
echo -----------------------------------

set start=%time%

orca.exe %foldername%.inp > %foldername%.out

set end=%time%
set end=%time%
set options="tokens=1-4 delims=:."
for /f %options% %%a in ("%start%") do set start_h=%%a&set /a start_m=100%%b %% 100&set /a start_s=100%%c %% 100&set /a start_ms=100%%d %% 100
for /f %options% %%a in ("%end%") do set end_h=%%a&set /a end_m=100%%b %% 100&set /a end_s=100%%c %% 100&set /a end_ms=100%%d %% 100
set /a hours=%end_h%-%start_h%
set /a mins=%end_m%-%start_m%
set /a secs=%end_s%-%start_s%
set /a ms=%end_ms%-%start_ms%
if %hours% lss 0 set /a hours = 24%hours%
if %mins% lss 0 set /a hours = %hours% - 1 & set /a mins = 60%mins%
if %secs% lss 0 set /a mins = %mins% - 1 & set /a secs = 60%secs%
if %ms% lss 0 set /a secs = %secs% - 1 & set /a ms = 100%ms%
if 1%ms% lss 100 set ms=0%ms%
:: mission accomplished
set /a totalsecs = %hours%*3600 + %mins%*60 + %secs%

echo Elapsed time %hours%:%mins%:%secs%.%ms% (%totalsecs%.%ms%s total)


echo -----------------------------------
echo Plot for %foldername%
echo -----------------------------------



lualatex.exe   -synctex=1 -interaction=nonstopmode -shell-escape "energy".tex  >NUL 2>&1

:: delete files

del *.xyz *.densities *.engrad *.gbw *.opt *_property.txt *.allxyz *.relaxscanact.dat *.tmp *.ges ^
    *.gz energy.out *.aux *.log ^
    2>nul

echo Done!

pause


