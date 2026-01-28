@echo off
chcp 65001 >nul
:: === QwenIA 启动脚本 ===
title QwenIA

python utils/get_env.py
call set_env.bat

:: 检查 llama-server
if not exist "%SERVER%" (
    echo [ERROR] llama-server NOT FOUND, check your PATH settings.
    pause
    exit /b
)
powershell -Command "$existing = Get-Process | Where-Object { $_.ProcessName -like 'llama-server' }; if ($existing) { exit 100 } else { exit 0 }"

:: 启动 llama-server
if %ERRORLEVEL% EQU 100 (
    echo [INFO] llama-server is already running. Skipping startup.
) else (
    start "llama-server" "%SERVER%" --port %PORT% -m "%MODEL_PATH%" --mmproj "%MMPROJ_PATH%" -fa -ngl %gpu-layers% --keep %keep% --ctx-size %ctx-size% --temp %temperature% --top-k %top-k% --top-p %top-p% --repeat-penalty %repeat-penalty%
    powershell -Command "$p = Get-Process | Where-Object { $_.MainWindowTitle -eq 'llama-server' }; if ($p) { Add-Type -Namespace Native -Name Win32 -MemberDefinition '[DllImport(\"user32.dll\")]public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);'; [void][Native.Win32]::ShowWindowAsync($p.MainWindowHandle, 2); }"
)

:: 启动主程序
echo [INFO] Initializing QwenIA...
python inference.py

exit /b