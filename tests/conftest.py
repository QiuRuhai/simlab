import sys
from pathlib import Path

# 让测试能 import scripts/ 下的入口模块
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
