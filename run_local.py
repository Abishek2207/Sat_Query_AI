import socket
import subprocess
import os
import time
import sys

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def main():
    print("===================================================")
    print("    SATQUERY AI - BULLETPROOF DEMO LAUNCHER        ")
    print("===================================================")
    
    port = find_free_port()
    print(f"[*] Found guaranteed free port for Backend: {port}")
    
    print("[*] Starting FastAPI Backend...")
    backend_env = os.environ.copy()
    backend_env["HF_HUB_OFFLINE"] = "1"
    backend_env["TRANSFORMERS_OFFLINE"] = "1"
    
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd="backend",
        env=backend_env
    )
    
    print("[*] Waiting for backend to initialize...")
    time.sleep(5)
    
    print("[*] Starting React Frontend...")
    frontend_env = os.environ.copy()
    frontend_env["VITE_API_BASE_URL"] = f"http://127.0.0.1:{port}"
    
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        env=frontend_env,
        shell=True
    )
    
    print("\n===================================================")
    print("ALL SYSTEMS GO! 🚀")
    print(f"Backend is running safely on port {port}")
    print("Keep this terminal open! Press Ctrl+C to close everything.")
    print("===================================================\n")
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Shutting down servers safely...")
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    main()
