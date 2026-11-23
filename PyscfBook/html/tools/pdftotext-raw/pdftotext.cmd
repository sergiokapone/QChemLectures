@echo off
rem lwarpmk calls "pdftotext -layout": with -layout a line broken inside a formula/anchor reorders pieces of a paragraph.
rem This wrapper (put first in PATH by build-html.ps1) runs the real pdftotext (%PDFTOTEXT_REAL%) with -raw: TeX output order.
setlocal
set "ARGS=%*"
set "ARGS=%ARGS:-layout=-raw%"
"%PDFTOTEXT_REAL%" %ARGS%
