#!/bin/bash

if [ -f logs/api.pid ]; then
    PID=$(cat logs/api.pid)
    echo "正在停止API服务 (PID: $PID)..."
    kill $PID
    rm logs/api.pid
    echo "API服务已停止"
else
    echo "未找到运行的API服务"
fi 