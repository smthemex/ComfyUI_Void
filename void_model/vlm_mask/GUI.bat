@echo off
chcp 65001 >nul
title Point Selector GUI

echo ================================
echo Point Selector GUI 启动脚本
echo ================================

REM 检查Python是否已安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请确保已安装Python并添加到PATH环境变量中
    pause
    exit /b 1
)

REM 检查文件是否存在
if not exist "point_selector_gui.py" (
    echo 错误: 未找到 point_selector_gui.py 文件
    pause
    exit /b 1
)

echo 正在启动 Point Selector GUI...
echo.

REM 启动GUI程序
python point_selector_gui.py

REM 检查退出状态
if errorlevel 1 (
    echo.
    echo 程序异常退出，请检查错误信息
    pause
) else (
    echo.
    echo 程序正常结束
    pause
)