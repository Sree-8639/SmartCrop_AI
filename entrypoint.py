"""
SmartCrop AI - Unified Production Process Supervisor
Starts both the FastAPI inference backend (port 8000) and the Streamlit frontend.
Compatible with Docker, Google Cloud Run, Streamlit Community Cloud, and local servers.
"""

import os
import sys
import time
import signal
import subprocess
from urllib.request import urlopen

def check_backend_ready(url="http://127.0.0.1:8000/api/health", retries=25, delay=1.0):
    for i in range(retries):
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    print("[SmartCrop Supervisor] Backend is healthy and ready!")
                    return True
        except Exception:
            time.sleep(delay)
    return False

def main():
    port = os.getenv("PORT", "8501")
    backend_port = os.getenv("BACKEND_PORT", "8000")

    print("==================================================================")
    print("🌱 SmartCrop AI — Multi-Modal Agricultural Intelligence Platform")
    print(f"Starting FastAPI Backend on port {backend_port}...")
    print(f"Starting Streamlit Frontend on port {port}...")
    print("==================================================================")

    # 1. Start FastAPI Backend
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "app.api.routes:app",
        "--host", "127.0.0.1",
        "--port", str(backend_port)
    ]
    backend_proc = subprocess.Popen(backend_cmd)

    # 2. Wait for backend readiness
    check_backend_ready(f"http://127.0.0.1:{backend_port}/api/health")

    # 3. Start Streamlit Frontend
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app/main.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    frontend_proc = subprocess.Popen(streamlit_cmd)

    def shutdown(signum, frame):
        print("\n[SmartCrop Supervisor] Graceful shutdown initiated...")
        frontend_proc.terminate()
        backend_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor both processes
    while True:
        if backend_proc.poll() is not None:
            print("[SmartCrop Supervisor] Error: FastAPI Backend exited unexpectedly.")
            frontend_proc.terminate()
            sys.exit(1)
        if frontend_proc.poll() is not None:
            print("[SmartCrop Supervisor] Streamlit Frontend exited.")
            backend_proc.terminate()
            sys.exit(0)
        time.sleep(2)

if __name__ == "__main__":
    main()

