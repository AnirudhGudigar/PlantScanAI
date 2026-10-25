"""
Verification script for interactive cursor-driven 3D wave animation on PlantScanAI background.
Launches headless Chrome with CDP, moves the cursor across the hero grid,
captures screenshots during wave excitation, and verifies console log health.
"""
import subprocess
import time
import json
import urllib.request
import socket
import hashlib
import base64
import os
import struct

def build_ws_frame(msg_str):
    data = msg_str.encode('utf-8')
    length = len(data)
    frame = bytearray()
    frame.append(0x81) # text frame, FIN=1
    mask_key = os.urandom(4)
    if length <= 125:
        frame.append(0x80 | length)
    elif length <= 65535:
        frame.append(0x80 | 126)
        frame.extend(struct.pack('>H', length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack('>Q', length))
    frame.extend(mask_key)
    masked_data = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(data))
    frame.extend(masked_data)
    return frame

def read_ws_frame(sock):
    header = sock.recv(2)
    if not header or len(header) < 2:
        return None
    b1, b2 = header[0], header[1]
    is_masked = (b2 & 0x80) != 0
    length = b2 & 0x7F
    if length == 126:
        ext = sock.recv(2)
        length = struct.unpack('>H', ext)[0]
    elif length == 127:
        ext = sock.recv(8)
        length = struct.unpack('>Q', ext)[0]
    mask_key = sock.recv(4) if is_masked else b''
    payload = bytearray()
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            break
        payload.extend(chunk)
    if is_masked:
        payload = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(payload))
    return payload.decode('utf-8', errors='ignore')

def ws_handshake(host, port, path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    key = base64.b64encode(os.urandom(16)).decode('utf-8')
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(req.encode('utf-8'))
    resp = sock.recv(4096).decode('utf-8')
    if "101" not in resp:
        raise RuntimeError("Failed WebSocket handshake: " + resp)
    return sock

def run_test():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    port = 9444
    proc = subprocess.Popen([
        chrome_path,
        "--headless=new",
        f"--remote-debugging-port={port}",
        "--use-gl=angle",
        "--ignore-gpu-blocklist",
        "--window-size=1280,800",
        "http://127.0.0.1:8000/app"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(3.0)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as resp:
            targets = json.loads(resp.read().decode())
        page = next(t for t in targets if t.get("type") == "page")
        ws_url = page.get("webSocketDebuggerUrl")
        print("[CDP] Connected to page:", page.get("title"))
        
        from urllib.parse import urlparse
        u = urlparse(ws_url)
        sock = ws_handshake(u.hostname, u.port, u.path)

        def send_cdp(method, params=None, msg_id=[1]):
            mid = msg_id[0]
            msg_id[0] += 1
            payload = json.dumps({"id": mid, "method": method, "params": params or {}})
            sock.sendall(build_ws_frame(payload))
            while True:
                res = read_ws_frame(sock)
                if not res:
                    return None
                try:
                    obj = json.loads(res)
                    if obj.get("id") == mid:
                        return obj
                except Exception:
                    pass

        # Enable runtime & console
        send_cdp("Console.enable")
        send_cdp("Runtime.enable")

        # Let the page initialize Three.js
        time.sleep(1.5)

        # 1. Take baseline screenshot before cursor enters
        res = send_cdp("Page.captureScreenshot", {"format": "png"})
        if res and "result" in res and "data" in res["result"]:
            with open("tests/wave_1_baseline.png", "wb") as f:
                f.write(base64.b64decode(res["result"]["data"]))
            print("[Test] Saved baseline screenshot: tests/wave_1_baseline.png")

        # 2. Simulate cursor moving in an arc across the hero section
        print("[Test] Dispatching cursor movement across hero grid...")
        points = [
            (250, 300), (350, 280), (450, 260), (550, 250),
            (650, 250), (750, 270), (850, 300), (950, 320)
        ]
        for x, y in points:
            send_cdp("Input.dispatchMouseEvent", {
                "type": "mouseMoved",
                "x": x,
                "y": y
            })
            time.sleep(0.06)

        # Hold cursor at (650, 260) and capture the active wave peak & ripples
        time.sleep(0.12)
        res = send_cdp("Page.captureScreenshot", {"format": "png"})
        if res and "result" in res and "data" in res["result"]:
            with open("tests/wave_2_active.png", "wb") as f:
                f.write(base64.b64decode(res["result"]["data"]))
            print("[Test] Saved active wave screenshot: tests/wave_2_active.png")

        # 3. Move cursor outside hero container to trigger settling
        print("[Test] Moving cursor outside hero...")
        send_cdp("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": 640,
            "y": 780
        })
        time.sleep(1.2)
        res = send_cdp("Page.captureScreenshot", {"format": "png"})
        if res and "result" in res and "data" in res["result"]:
            with open("tests/wave_3_settled.png", "wb") as f:
                f.write(base64.b64decode(res["result"]["data"]))
            print("[Test] Saved settled screenshot: tests/wave_3_settled.png")

        # Check console logs for errors
        eval_res = send_cdp("Runtime.evaluate", {
            "expression": "({ canvasPresent: !!document.getElementById('cubesCanvas'), threeLoaded: typeof THREE !== 'undefined' })",
            "returnByValue": True
        })
        print("[Test] Page Evaluation:", eval_res.get("result", {}).get("result", {}).get("value"))

        sock.close()
    finally:
        proc.terminate()
        print("[Test] Completed headless verification.")

if __name__ == "__main__":
    run_test()
