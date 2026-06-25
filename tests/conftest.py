import sys
from pathlib import Path

# 让测试能 import scripts/ 下的入口模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import pytest
import taichi as ti


@pytest.fixture
def ti_cpu():
    """每个用例全新 ti.cpu runtime（→ 全新 field），确定性、无 GPU。"""
    ti.init(arch=ti.cpu)
    yield
