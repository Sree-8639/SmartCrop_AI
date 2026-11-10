"""
SmartCrop AI - Root Entrypoint for Streamlit Community Cloud & Deployment Platforms
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Run app/main.py directly
with open(ROOT_DIR / "app" / "main.py", encoding="utf-8") as f:
    code = compile(f.read(), str(ROOT_DIR / "app" / "main.py"), 'exec')
    exec(code, globals())
