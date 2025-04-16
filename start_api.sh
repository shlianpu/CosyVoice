#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p logs

# 在后台运行API服务，并将输出重定向到日志文件
nohup python api.py > logs/api.log 2>&1 &

# 保存进程ID
echo $! > logs/api.pid

echo "API服务已在后台启动，进程ID: $(cat logs/api.pid)"
echo "日志文件位置: logs/api.log"
echo "使用 'tail -f logs/api.log' 查看实时日志" 