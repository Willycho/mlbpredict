"""데이터 수집 + 가공 실행 스크립트."""

import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.collector import collect_all
from data.processor import process_all


if __name__ == "__main__":
    collect_all()
    process_all()
