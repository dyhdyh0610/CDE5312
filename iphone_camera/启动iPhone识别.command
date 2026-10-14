#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x ../.venv/bin/python ]; then
  echo "未找到项目的 Python 环境：../.venv/bin/python"
  read -r -p "按回车关闭窗口。"
  exit 1
fi
../.venv/bin/python iphone_yolo.py "$@"
iphone_exit_code=$?
if [ "$iphone_exit_code" -ne 0 ] && [ -t 0 ]; then
  read -r -p "按回车关闭窗口。"
fi
exit "$iphone_exit_code"
