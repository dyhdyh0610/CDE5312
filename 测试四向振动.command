#!/bin/bash
cd "$(dirname "$0")" || exit 1
./.venv/bin/python sony_haptics.py --port auto --test all
task_exit_code=$?
read -r -p "测试结束，按回车关闭窗口。"
exit "$task_exit_code"
