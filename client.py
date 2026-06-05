"""
Client helper - Đăng ký và bắt đầu thi.
Usage: uv run python client.py [register|evaluate|reset|result]
"""

import sys
import os
import requests

STUDENT_ID = os.environ.get("STUDENT_ID", "B22DCVT028")
TEACHER_BASE = os.environ.get("TEACHER_BASE", "http://192.168.50.218:8000/api/v1")
MY_SERVER_URL = os.environ.get("MY_SERVER_URL", "http://192.168.50.97:5000")

HEADERS = {"X-Student-ID": STUDENT_ID}


def register():
    """Đăng ký server với Teacher."""
    resp = requests.post(
        f"{TEACHER_BASE}/competition/register",
        headers=HEADERS,
        json={"server_url": MY_SERVER_URL},
    )
    print(f"[REGISTER] {resp.status_code}")
    print(resp.json())


def evaluate():
    """Bắt đầu quá trình thi."""
    resp = requests.post(
        f"{TEACHER_BASE}/competition/evaluate",
        headers=HEADERS,
    )
    print(f"[EVALUATE] {resp.status_code}")
    print(resp.json())


def reset():
    """Reset trạng thái thi."""
    resp = requests.post(
        f"{TEACHER_BASE}/competition/reset",
        headers=HEADERS,
    )
    print(f"[RESET] {resp.status_code}")
    print(resp.json())


def result():
    """Kiểm tra kết quả."""
    resp = requests.get(
        f"{TEACHER_BASE}/competition/result",
        headers=HEADERS,
    )
    print(f"[RESULT] {resp.status_code}")
    print(resp.json())


if __name__ == "__main__":
    commands = {
        "register": register,
        "evaluate": evaluate,
        "reset": reset,
        "result": result,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python client.py [{' | '.join(commands.keys())}]")
        print(f"\nConfig:")
        print(f"  STUDENT_ID = {STUDENT_ID}")
        print(f"  TEACHER_BASE = {TEACHER_BASE}")
        print(f"  MY_SERVER_URL = {MY_SERVER_URL}")
        sys.exit(1)

    commands[sys.argv[1]]()
