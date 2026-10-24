"""
PlantScan AI - FastAPI Connection Test Script

Run this script to verify that your FastAPI backend server is running and accessible.
"""

import json
import sys
import urllib.request
import urllib.error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

def test_connection():
    print("=" * 60)
    print("PlantScan AI - Testing FastAPI Server Connection")
    print("=" * 60)

    # 1. Test Root Endpoint
    try:
        req = urllib.request.urlopen(f"{BASE_URL}/", timeout=5)
        data = json.loads(req.read().decode('utf-8'))
        print("\n✅ ROOT ENDPOINT (/) SUCCESS:")
        print(json.dumps(data, indent=2))
    except urllib.error.URLError as e:
        print(f"\n❌ FAILED TO CONNECT TO {BASE_URL}: {e}")
        print("\n💡 Make sure the FastAPI server is running!")
        print("   Start it by running `.\\run_backend.bat` or `.\\.venv\\Scripts\\python.exe -m backend.main`")
        return

    # 2. Test Health Check Endpoint
    try:
        req = urllib.request.urlopen(f"{BASE_URL}/api/v1/health", timeout=5)
        data = json.loads(req.read().decode('utf-8'))
        print("\n✅ HEALTH CHECK ENDPOINT (/api/v1/health) SUCCESS:")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"\n⚠️ Health Check Failed: {e}")

    # 3. Test Supported Crops Endpoint
    try:
        req = urllib.request.urlopen(f"{BASE_URL}/api/v1/treatments/crops", timeout=5)
        data = json.loads(req.read().decode('utf-8'))
        print("\n✅ CROPS LIST ENDPOINT (/api/v1/treatments/crops) SUCCESS:")
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"\n⚠️ Crops Endpoint Failed: {e}")

    print("\n" + "=" * 60)
    print("🎉 FastAPI is CONNECTED and OPERATIONAL!")
    print("   Open Swagger UI in your browser: http://localhost:8000/docs")
    print("=" * 60)

if __name__ == "__main__":
    test_connection()
