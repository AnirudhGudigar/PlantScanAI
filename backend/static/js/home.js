/**
 * PlantScanAI - Home Page 3D Experience & Brand Display
 * Three.js Aceternity 3D Cubes Grid with cursor-driven sand bowl deformation,
 * and Flicker-inspired dot-matrix dynamic brand typography.
 */

document.addEventListener('DOMContentLoaded', () => {
    /* --------------------------------------------------------------------------
       1. ACETERNITY BACKGROUND CUBES GRID (THREE.JS 3D PERSPECTIVE VARIANT)
       -------------------------------------------------------------------------- */
    (function initAceternity3DCubes() {
      const container = document.getElementById('cubesPerspectiveContainer');
      const canvas = document.getElementById('cubesCanvas');
      const hero = document.getElementById('hero');
      if (!container || !canvas || !hero || typeof THREE === 'undefined') return;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x060b18);
      scene.fog = new THREE.Fog(0x060b18, 10.0, 28.0);

      let width = container.clientWidth || window.innerWidth;
      let height = container.clientHeight || 640;

      const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
      const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
      renderer.setSize(width, height, false);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.25;

      const CAMERA_RADIUS = 13.2;
      // 25.2 deg elevation, 45 deg diagonal corner view:
      const tilt = 0.44;
      const baseAzimuth = 0.785; // 45° diagonal angle exposing Top, Left, and Right faces

      let targetMouseX = 0, targetMouseY = 0;
      let mouseX = 0, mouseY = 0;
      let idleTime = 0;
      let isMouseInside = false;

      function updateCamera(mx, my, ambientOffset) {
        const alpha = tilt + my * (Math.PI * 0.02);
        const beta = baseAzimuth + mx * (Math.PI * 0.03) + (ambientOffset || 0);
        camera.position.set(
          CAMERA_RADIUS * Math.cos(alpha) * Math.sin(beta),
          CAMERA_RADIUS * Math.sin(alpha),
          CAMERA_RADIUS * Math.cos(alpha) * Math.cos(beta)
        );
        camera.up.set(0, 1, 0);
        camera.lookAt(0, -0.1, 0);
      }
      updateCamera(0, 0, 0);

      // Directional & Ambient Illumination Rig (Crisp illumination with subtle electric-violet rim highlight)
      scene.add(new THREE.AmbientLight(0xe8ecff, 1.15));

      const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
      keyLight.position.set(-15, 20, 10);
      scene.add(keyLight);

      const rimLight = new THREE.DirectionalLight(0x8a72ec, 0.75);
      rimLight.position.set(18, 12, -10);
      scene.add(rimLight);

      const topLight = new THREE.DirectionalLight(0xffffff, 1.3);
      topLight.position.set(0, 25, 0);
      scene.add(topLight);

      // Extended 3D Cube Grid Layout (Spans full visible hero section seamlessly beyond all screen edges)
      const size = 72;
      const spacing = 1.30;
      const gap = 0.0455; // Proportional 1.30x baseline gap (0.035 * 1.30)
      const tileSize = spacing - gap; // 1.2545 (exact 1.30x baseline width/depth 0.965)
      const count = size * size;

      // Box geometry translated so its top face lies EXACTLY at Y = 0 (baseline)
      // and its side walls extend downward into the negative Y axis (-15.6).
      const columnDepth = 15.6; // exact 1.30x baseline height (12.0 * 1.30)
      const geometry = new THREE.BoxGeometry(tileSize, columnDepth, tileSize);
      geometry.translate(0, -columnDepth / 2, 0); // Top face at Y = 0!

      // Material: Pure white (#FFFFFF) top faces and soft light gray side faces matching reference image
      const material = new THREE.MeshPhongMaterial({
        color: 0xffffff,
        specular: 0x333338,
        shininess: 25,
        flatShading: false
      });

      // Shader injection: Pure white (#FFFFFF) top faces & soft light gray side faces matching reference
      material.onBeforeCompile = function (shader) {
        shader.vertexShader = shader.vertexShader
          .replace(
            '#include <common>',
            `#include <common>
            varying vec2 vUv;
            varying vec3 vWorldNormal;`
          )
          .replace(
            '#include <uv_vertex>',
            `#include <uv_vertex>
            vUv = uv;`
          )
          .replace(
            '#include <worldpos_vertex>',
            `#include <worldpos_vertex>
            vWorldNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);`
          );

        shader.fragmentShader = shader.fragmentShader
          .replace(
            '#include <common>',
            `#include <common>
            varying vec2 vUv;
            varying vec3 vWorldNormal;`
          )
          .replace(
            '#include <color_fragment>',
            `#include <color_fragment>
            // Subtle edge definition along tile seams
            vec2 edgeDist = min(vUv, vec2(1.0) - vUv);
            float edgeVal = min(edgeDist.x, edgeDist.y);
            float isEdge = 1.0 - smoothstep(0.008, 0.035, edgeVal);

            vec3 n = normalize(vWorldNormal);
            bool isTop = n.y > 0.5;

            if (isTop) {
              // TOP FACE: Pure bright white (#FFFFFF), clean and luminous
              diffuseColor.rgb = vec3(1.0, 1.0, 1.0);
            } else {
              // SIDE FACES: Soft, light gray matching reference image with natural directional shading
              vec3 sideBase = (abs(n.x) > 0.4) ? vec3(0.68, 0.69, 0.74) : vec3(0.60, 0.61, 0.66);
              diffuseColor.rgb = mix(sideBase, vec3(0.50, 0.51, 0.56), isEdge * 0.35);
            }`
          );
      };

      const mesh = new THREE.InstancedMesh(geometry, material, count);
      const dummy = new THREE.Object3D();
      // Center the grid extended along the camera's diagonal view line to cover all screen edges seamlessly
      const originX = -24.0;
      const originZ = -24.0;
      const minX = originX - ((size - 1) * spacing) / 2;
      const minZ = originZ - ((size - 1) * spacing) / 2;

      // Base grid positions
      const basePositions = new Float32Array(count * 2);
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          const index = i * size + j;
          const x = minX + i * spacing;
          const z = minZ + j * spacing;
          basePositions[index * 2] = x;
          basePositions[index * 2 + 1] = z;
          dummy.position.set(x, 0, z);
          dummy.rotation.set(0, 0, 0);
          dummy.updateMatrix();
          mesh.setMatrixAt(index, dummy.matrix);
        }
      }
      mesh.instanceMatrix.needsUpdate = true;
      scene.add(mesh);

      // Sand Indentation Simulation State (Sphere moving through sand)
      const SIM_HZ = 120;
      const STEP = 1 / SIM_HZ;
      const BRUSH_RADIUS = 0.2925; // Scaled proportionally with bowl radius (0.195 * 1.5)
      const MAX_BOWL_DEPTH = 2.493; // 75% increased downward depth (1.4245 * 1.75)
      const BOWL_RADIUS = 3.94875; // Bowl surface radius increased by 50% (2.6325 * 1.5)
      const flowPull = 1.15; // Faster trail recovery (half the previous recovery time)
      const settlePull = 4.40; // Quick, smooth return to flat baseline in half time
      const SETTLE_AFTER = 0.12;
      const SETTLE_RAMP = 0.65; // Natural trailing fade in half the time

      const heightField = new Float32Array(count);
      const lastHeightField = new Float32Array(count);
      const currentHeights = new Float32Array(count);

      const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const raycaster = new THREE.Raycaster();
      const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
      const hitPoint = new THREE.Vector3();

      let lastPointerGridX = null;
      let lastPointerGridZ = null;
      let targetPointerGridX = null;
      let targetPointerGridZ = null;
      let smoothPointerGridX = null;
      let smoothPointerGridZ = null;
      let isInteracting = false;
      let idleTimer = 0;

      function brush(gx, gz, radius = BRUSH_RADIUS) {
        const effRadius = Math.max(BOWL_RADIUS, radius * 13.5);
        const reach = Math.ceil(effRadius);
        const rSq = effRadius * effRadius;
        const ci = Math.round(gx);
        const cj = Math.round(gz);

        for (let i = Math.max(0, ci - reach); i <= Math.min(size - 1, ci + reach); i++) {
          for (let j = Math.max(0, cj - reach); j <= Math.min(size - 1, cj + reach); j++) {
            const dx = i - gx;
            const dz = j - gz;
            const distSq = dx * dx + dz * dz;
            if (distSq < rSq) {
              const d = Math.sqrt(distSq);
              const x = d / effRadius;
              const invX = 1 - x;
              const falloff = invX * invX * invX * (6 * x * x + 3 * x + 1); // C2 smootherstep
              const targetY = -MAX_BOWL_DEPTH * falloff;
              const k = i * size + j;
              // Pure downward one-sided indentation into sand
              if (targetY < heightField[k]) {
                heightField[k] = targetY;
              }
            }
          }
        }
      }

      function handlePointerMove(clientX, clientY) {
        if (prefersReduced) return;
        const rect = hero.getBoundingClientRect();
        targetMouseX = ((clientX - rect.left) / rect.width) * 2 - 1;
        targetMouseY = -(((clientY - rect.top) / rect.height) * 2 - 1);
        idleTimer = 0;
        isInteracting = true;

        raycaster.setFromCamera(new THREE.Vector2(targetMouseX, targetMouseY), camera);
        if (raycaster.ray.intersectPlane(groundPlane, hitPoint)) {
          const gx = (hitPoint.x - minX) / spacing;
          const gz = (hitPoint.z - minZ) / spacing;
          if (gx >= 0 && gx < size && gz >= 0 && gz < size) {
            targetPointerGridX = gx;
            targetPointerGridZ = gz;
          }
        }
      }

      function handlePointerLeave() {
        isInteracting = false;
        targetPointerGridX = null;
        targetPointerGridZ = null;
        smoothPointerGridX = null;
        smoothPointerGridZ = null;
        lastPointerGridX = null;
        lastPointerGridZ = null;
        targetMouseX = 0;
        targetMouseY = 0;
      }

      if (!prefersReduced) {
        hero.addEventListener('mousemove', (e) => {
          handlePointerMove(e.clientX, e.clientY);
        });

        hero.addEventListener('mouseleave', () => {
          handlePointerLeave();
        });

        // Touch event support on mobile & tablet displays
        hero.addEventListener('touchstart', (e) => {
          if (e.touches && e.touches.length > 0) {
            handlePointerMove(e.touches[0].clientX, e.touches[0].clientY);
          }
        }, { passive: true });

        hero.addEventListener('touchmove', (e) => {
          if (e.touches && e.touches.length > 0) {
            handlePointerMove(e.touches[0].clientX, e.touches[0].clientY);
          }
        }, { passive: true });

        hero.addEventListener('touchend', () => {
          handlePointerLeave();
        }, { passive: true });

        hero.addEventListener('touchcancel', () => {
          handlePointerLeave();
        }, { passive: true });
      }

      function applyPointer() {
        if (targetPointerGridX === null || targetPointerGridZ === null) return;
        if (lastPointerGridX === null || lastPointerGridZ === null) {
          lastPointerGridX = targetPointerGridX;
          lastPointerGridZ = targetPointerGridZ;
          brush(targetPointerGridX, targetPointerGridZ, BRUSH_RADIUS);
          return;
        }

        const dx = targetPointerGridX - lastPointerGridX;
        const dz = targetPointerGridZ - lastPointerGridZ;
        const dist = Math.hypot(dx, dz);
        if (dist > 0.001) {
          const samples = Math.max(1, Math.ceil(dist / 0.10));
          for (let s = 1; s <= samples; s++) {
            brush(
              lastPointerGridX + (dx * s) / samples,
              lastPointerGridZ + (dz * s) / samples,
              BRUSH_RADIUS
            );
          }
          lastPointerGridX = targetPointerGridX;
          lastPointerGridZ = targetPointerGridZ;
        } else {
          // Paused under cursor: maintain sphere bowl depression
          brush(targetPointerGridX, targetPointerGridZ, BRUSH_RADIUS);
        }
      }

      function physicsStep() {
        const ramp = Math.min(Math.max((idleTimer - SETTLE_AFTER) / SETTLE_RAMP, 0), 1);
        const s = ramp * ramp * (3 - 2 * ramp);
        const relax = Math.exp(-(flowPull + (settlePull - flowPull) * s) / SIM_HZ);

        lastHeightField.set(heightField);

        // One-sided monotonic sand relaxation: cubes smoothly return to 0 with zero wave oscillation
        for (let k = 0; k < count; k++) {
          if (heightField[k] < 0) {
            heightField[k] = heightField[k] * relax;
            if (heightField[k] > -0.0002) {
              heightField[k] = 0;
            }
          } else {
            heightField[k] = 0;
          }
        }
      }

      // Responsive viewport resizing
      const onResize = () => {
        width = container.clientWidth || hero.clientWidth || window.innerWidth;
        height = container.clientHeight || hero.clientHeight || 640;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
      };

      if ('ResizeObserver' in window) {
        new ResizeObserver(onResize).observe(container);
      } else {
        window.addEventListener('resize', onResize);
      }

      // Main 60 FPS rendering loop with fixed-step wave physics
      let animId = null;
      let lastTime = performance.now();
      let accumulator = 0;

      function renderLoop(now) {
        const delta = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        mouseX += (targetMouseX - mouseX) * 0.05;
        mouseY += (targetMouseY - mouseY) * 0.05;

        let ambientOffset = 0;
        if (!prefersReduced) {
          idleTimer += delta;
          ambientOffset = Math.sin(now * 0.00075) * 0.012; // Very subtle ambient tilt
        }

        updateCamera(mouseX, mouseY, ambientOffset);

        if (!prefersReduced) {
          applyPointer();

          accumulator = Math.min(accumulator + delta, STEP * 8);
          while (accumulator >= STEP) {
            physicsStep();
            accumulator -= STEP;
          }

          const alpha = accumulator / STEP;
          const ramp = Math.min(Math.max((idleTimer - SETTLE_AFTER) / SETTLE_RAMP, 0), 1);
          const s = ramp * ramp * (3 - 2 * ramp);
          const currentUpRate = flowPull + (settlePull - flowPull) * s;
          const downFactor = 1 - Math.exp(-22.0 * delta); // Smooth, continuous descent into bowl
          const upFactor = 1 - Math.exp(-currentUpRate * delta); // Smooth monotonic recovery in half time

          const rSq = BOWL_RADIUS * BOWL_RADIUS;
          const hasCursor = isInteracting && targetPointerGridX !== null && targetPointerGridZ !== null;

          if (hasCursor) {
            if (smoothPointerGridX === null || smoothPointerGridZ === null) {
              smoothPointerGridX = targetPointerGridX;
              smoothPointerGridZ = targetPointerGridZ;
            } else {
              // High-resolution sub-frame pointer smoothing for continuous, fluid bowl tracking
              const ptrRate = 1 - Math.exp(-30.0 * delta);
              smoothPointerGridX += (targetPointerGridX - smoothPointerGridX) * ptrRate;
              smoothPointerGridZ += (targetPointerGridZ - smoothPointerGridZ) * ptrRate;
            }
          } else {
            smoothPointerGridX = null;
            smoothPointerGridZ = null;
          }

          let anyNonZero = false;

          for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
              const k = i * size + j;
              const bx = basePositions[k * 2];
              const bz = basePositions[k * 2 + 1];

              // Interpolated trail height from simulation
              const h = lastHeightField[k] + (heightField[k] - lastHeightField[k]) * alpha;

              // Active bowl depression centered directly under cursor
              let activeBowl = 0.0;
              if (hasCursor && smoothPointerGridX !== null && smoothPointerGridZ !== null) {
                const dx = i - smoothPointerGridX;
                const dz = j - smoothPointerGridZ;
                const distSq = dx * dx + dz * dz;
                if (distSq < rSq) {
                  const d = Math.sqrt(distSq);
                  const x = d / BOWL_RADIUS;
                  const invX = 1 - x;
                  const falloff = invX * invX * invX * (6 * x * x + 3 * x + 1); // C2 smootherstep
                  activeBowl = -MAX_BOWL_DEPTH * falloff;
                }
              }

              const targetH = Math.min(activeBowl, h);
              const prevY = currentHeights[k];
              let newY = prevY;

              if (targetH < prevY) {
                // Smoothly descend into bowl without instantaneous snapping
                newY = prevY + (targetH - prevY) * downFactor;
              } else if (prevY < 0.0) {
                // Smoothly recover back up toward target/baseline
                newY = prevY + (targetH - prevY) * upFactor;
                if (newY > -0.0003 && targetH === 0.0) {
                  newY = 0.0;
                }
              }

              // Pure one-sided downward bowl displacement (sand indentation model)
              const vHeight = Math.max(-MAX_BOWL_DEPTH, Math.min(0.0, newY));
              currentHeights[k] = vHeight;

              // Removed excessive tilting & flipping: vertical displacement is primary
              // Subtle, bounded micro-tilt conforming to bowl contour (max ~1.5 degrees, strictly clamped)
              let tiltX = 0, tiltZ = 0;
              if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
                const rawTiltX = -(currentHeights[k + size] - currentHeights[k - size]) * 0.020;
                const rawTiltZ = (currentHeights[k + 1] - currentHeights[k - 1]) * 0.020;
                tiltX = Math.max(-0.028, Math.min(0.028, rawTiltX));
                tiltZ = Math.max(-0.028, Math.min(0.028, rawTiltZ));
              }

              if (Math.abs(vHeight) > 0.0003) {
                anyNonZero = true;
              }

              dummy.position.set(bx, vHeight, bz);
              dummy.rotation.set(tiltX, 0, tiltZ);
              dummy.updateMatrix();
              mesh.setMatrixAt(k, dummy.matrix);
            }
          }

          if (isInteracting || anyNonZero) {
            mesh.instanceMatrix.needsUpdate = true;
          }
        }

        renderer.render(scene, camera);

        if (!document.hidden) {
          animId = requestAnimationFrame(renderLoop);
        }
      }

      animId = requestAnimationFrame(renderLoop);

      document.addEventListener('visibilitychange', () => {
        if (!document.hidden && !animId) {
          lastTime = performance.now();
          animId = requestAnimationFrame(renderLoop);
        } else if (document.hidden && animId) {
          cancelAnimationFrame(animId);
          animId = null;
        }
      });
    })();

    /* --------------------------------------------------------------------------
       2. FLICKER-INSPIRED MONOCHROME DOT-MATRIX PLANTSCANAI TYPOGRAPHY
       -------------------------------------------------------------------------- */
    (function initDotMatrixTitle() {
      const canvas = document.getElementById('dotMatrixCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');

      const FONT_MAP = {
        'P': ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        'L': ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        'A': ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        'N': ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        'T': ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        'S': ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        'C': ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        'I': ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        ' ': ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
      };

      const word = "PLANTSCANAI";
      const dotRadius = 2.4;
      const dotSpacing = 7.5;
      const letterSpacing = 16;
      let frame = 0;

      function drawMatrix() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        frame += 0.035;

        const totalCharWidth = 5 * dotSpacing;
        const totalWidth = word.length * totalCharWidth + (word.length - 1) * letterSpacing;
        let startX = (canvas.width - totalWidth) / 2;
        const startY = 18;

        for (let i = 0; i < word.length; i++) {
          const char = word[i];
          const pattern = FONT_MAP[char] || FONT_MAP[' '];

          for (let row = 0; row < 7; row++) {
            const line = pattern[row];
            for (let col = 0; col < 5; col++) {
              const cx = startX + col * dotSpacing;
              const cy = startY + row * dotSpacing;

              if (line[col] === '1') {
                const pulse = Math.sin(frame + (i * 0.4) + (col * 0.2) + (row * 0.2));
                const alpha = 0.85 + pulse * 0.15;

                ctx.beginPath();
                ctx.arc(cx, cy, dotRadius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(196, 181, 253, ${alpha})`;
                ctx.shadowColor = 'rgba(167, 139, 250, 0.85)';
                ctx.shadowBlur = 4;
                ctx.fill();
                ctx.shadowBlur = 0;
              } else {
                ctx.beginPath();
                ctx.arc(cx, cy, 0.9, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(167, 139, 250, 0.08)';
                ctx.fill();
              }
            }
          }
          startX += totalCharWidth + letterSpacing;
        }

        requestAnimationFrame(drawMatrix);
      }

      drawMatrix();
    })();


});
