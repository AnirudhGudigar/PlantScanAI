# 🚀 PlantScan AI: Step-by-Step Execution Guide

This guide provides simple, step-by-step instructions for running **PlantScan AI** on Windows (PowerShell, Command Prompt, or File Explorer).

---

## ⚡ Quick Start (1-Click Executable Scripts)

For convenience on Windows, pre-configured batch scripts are provided in the root directory:

| Action | Script File | Description |
| :--- | :--- | :--- |
| **Start Web App & API** | `run_backend.bat` | Double-click or run `.\run_backend.bat` to launch the FastAPI server & web app at `http://localhost:8000/app` |
| **Run Unit Test Suite** | `run_tests.bat` | Double-click or run `.\run_tests.bat` to execute all 23 automated Pytest tests |
| **Test Image Predictor** | `run_predictor.bat` | Run `.\run_predictor.bat <image_path> <crop_name>` to test disease prediction via CLI |

---

## 🌐 Web Interfaces & Endpoints

Once the backend server is running, access the following in your browser:

- 🌿 **PlantScan AI Web Pathology Lab & Dashboard**: [http://localhost:8000/app](http://localhost:8000/app) (or [http://localhost:8000/app/workflow](http://localhost:8000/app/workflow))
- 📖 **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 📋 **ReDoc API Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- 💚 **System Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 📋 Manual Step-by-Step Instructions

### Step 1: Open Terminal & Navigate to Project Directory

Open **PowerShell** or **Command Prompt (CMD)**:

```powershell
cd "d:\Sagarnag\major project\pgpt\PlantScanAI"
```

---

### Step 2: Python Virtual Environment

All required dependencies are pre-installed in the local `.venv`. You do **not** need to manually activate the virtual environment if you use `.venv\Scripts\python.exe` directly:

#### Option A: Direct Execution (Recommended in PowerShell)
```powershell
.\.venv\Scripts\python.exe --version
```

#### Option B: Activate Virtual Environment
If PowerShell blocks script execution due to `Restricted` policy:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

---

### Step 3: Run the Automated Unit Test Suite

Run the full automated test suite (covering health, inference routing, thread safety, auto-crop detection, lesion quantification, and Kannada/Hindi localization):

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -v
```

To run auto-detection and lesion segmentation tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_auto_detect_and_lesions.py -v
```

---

### Step 4: Start the FastAPI Server & Web Application

Launch the server using Uvicorn with auto-reload:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
1. Open your browser to **`http://localhost:8000/app`**.
2. The interactive landing page and diagnostic workspace will load with the real 3D background cubes grid.

---

### Step 5: Test Model Predictor CLI

Test pathology classification directly on any leaf image:

```powershell
.\.venv\Scripts\python.exe ml/inference/crop_predictor.py --image "path\to\leaf.jpg" --crop tomato --use-tflite
```

---

### Step 6: Run Mobile Application (React Native Frontend)

```powershell
cd mobile_app
npm install
npm start
```

---

## 🧊 Aceternity UI 3D Background Cubes Grid Component

PlantScanAI includes a faithful recreation of the **Aceternity UI Background Cubes Grid** with genuine 3D perspective and physics-based cursor wave interaction:

### Files & Implementations:
1. **Live Standalone Web Application**:
   - File: `backend/static/index.html` (lines 2668–3080)
   - Served directly by FastAPI at `http://localhost:8000/app` and `http://localhost:8000/app/workflow`.
2. **React / Next.js Component**:
   - Component: `components/ui/background-cubes-grid.tsx`
   - Demo Wrapper: `components/background-cubes-grid-demo.tsx`

### Component Props & Configuration (`BackgroundCubesGrid`):
```tsx
export type PerspectiveCubesProps = {
  variant?: "perspective";
  className?: string;
  gridSize?: number;      // Default: 42 (42x42 grid = 1,764 interactive cubes)
  gap?: number;           // Default: 0.035 (tight architectural seam between flat tiles)
  baseColor?: string;     // Default: "#161619" (dark graphite monochrome)
  backgroundColor?: string; // Default: "#000000" (pure black)
  tilt?: number;          // Default: 0.44 (25.2° elevation angle)
  azimuth?: number;       // Default: 0.785 (45° diagonal angle revealing Top, Left, Right faces)
};
```

### Behavioral & Visual Dynamics:
- **Resting State (No Interaction)**: Continuous, planar perspective grid of flat square tiles at baseline height $Y = 0$. Vertical cube walls are translated into negative space (`columnDepth = 12.0`, `geometry.translate(0, -6, 0)`) so side faces remain hidden beneath the surface until displaced.
- **Cursor-Driven 3D Wave**: Moving the mouse exerts a downward velocity impulse:
  $$\text{velocity}[k] \mathrel{-}= \text{strength} \times \exp\left(-\frac{\text{dist}^2}{2r^2}\right)$$
- **2D Laplacian Wave Physics**: A 9-point discrete Laplacian wave equation simulates continuous, fluid-like propagation across neighboring tiles. Sub-steps are interpolated along the cursor trajectory to ensure smooth deformation during fast movements.
- **Depression Wells & Soft Rounded Crests**: Tiles sink beneath the baseline ($Y < 0$) in localized, tactile depression wells ($r = 0.975$), followed by gentle outward travelling crests rounded off by $\tanh(1.5 h) \times 0.5735$ and smoothly clamped between $[-1.4245, +0.5735]$. Tiles are strictly bounded in rotation ($\le 1.8^\circ$), eliminating any flipping or tall pillar behavior.
- **Depth-Dependent Face Shading**: Top faces sink into cavity shadows (`#0a0a0c`) or catch ambient light when elevated (`#42454e`). Vertical side faces are revealed only when height differences occur, shaded in graphite and charcoal with crisp silver seam highlights.
- **Damped Return to Baseline**: When the pointer leaves or stops, the wave smoothly damps to zero (`flowDrag` and `settleDrag`), returning the surface to a flat, silent plane.

---

## 📑 Summary Command Reference

```powershell
# 1. Start Server & Open Web App (http://localhost:8000/app)
.\run_backend.bat

# 2. Run All Automated Unit Tests (23 passed)
.\run_tests.bat

# 3. CLI Leaf Disease Prediction
.\.venv\Scripts\python.exe ml/inference/crop_predictor.py --image "path/to/leaf.jpg" --crop tomato --use-tflite
```
