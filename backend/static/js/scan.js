/**
 * PlantScanAI - Scan Plant Diagnostic Studio Script
 * Complete foliar image ingestion, neural classification, lesion segmentation,
 * multilingual speech advisory, IPM tabbed protocols, and historical scan inspection.
 */

let selectedCrop = '';
let selectedPart = '';
let selectedFile = null;
window.currentResult = null;
let activeTab = 'organic';
window.showingBinaryMask = false;
let voiceRate = parseFloat(localStorage.getItem('plantscanai_voice_rate')) || 1.0;
let speechSynth = window.speechSynthesis;
let speechUtterance = null;

function renderSelectors() {
  const s = UI_STRINGS[currentLanguage] || UI_STRINGS.en;

  // Crop Pills
  const cropContainer = document.getElementById('cropPills');
  if (cropContainer) {
    cropContainer.innerHTML = '';
    Object.entries(s.crops).forEach(([val, label]) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'custom-pill crop-pill' + (selectedCrop === val ? ' active' : '');
      btn.dataset.crop = val;
      btn.textContent = label;
      btn.onclick = () => {
        selectedCrop = val;
        renderSelectors();
      };
      cropContainer.appendChild(btn);
    });
  }

  // Part Pills
  const partContainer = document.getElementById('partPills');
  if (partContainer) {
    partContainer.innerHTML = '';
    Object.entries(s.parts).forEach(([val, label]) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'custom-pill part-pill' + (selectedPart === val ? ' active' : '');
      btn.dataset.part = val;
      btn.textContent = label;
      btn.onclick = () => {
        selectedPart = val;
        renderSelectors();
      };
      partContainer.appendChild(btn);
    });
  }

  // Sample Cards
  const sampleContainer = document.getElementById('sampleCardsContainer');
  if (sampleContainer) {
    sampleContainer.innerHTML = '';
    s.samplesList.forEach(sample => {
      const div = document.createElement('div');
      div.className = 'sample-card';
      div.onclick = () => loadSample(sample.type);
      div.innerHTML = `
        <div class="sample-icon">${sample.icon}</div>
        <div class="sample-info">
          <div class="sample-title">${sample.title}</div>
          <div class="sample-badge">Click to load &amp; test</div>
        </div>
      `;
      sampleContainer.appendChild(div);
    });
  }
}

function initDragAndDrop() {
  const dropzone = document.getElementById('dropzone');
  if (dropzone) {
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });
    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files[0]) {
        handleFileSelect(files[0]);
      }
    });
  }

  const imgInput = document.getElementById('imageInput');
  if (imgInput) {
    imgInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0]);
      }
    });
  }
}

function handleFileSelect(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('dropzone').style.display = 'none';
    document.getElementById('previewContainer').style.display = 'block';
    document.getElementById('btnAnalyze').disabled = false;
  };
  reader.readAsDataURL(file);
}

function removeImage(e) {
  if (e) e.stopPropagation();
  stopSpeech();
  selectedFile = null;
  document.getElementById('previewImg').src = '';
  document.getElementById('previewContainer').style.display = 'none';
  document.getElementById('dropzone').style.display = 'block';
  document.getElementById('btnAnalyze').disabled = true;
  document.getElementById('imageInput').value = '';
}

function loadSample(sampleType) {
  stopSpeech();
  let crop = 'Tomato';
  let part = '';

  if (sampleType === 'apple_healthy') {
    crop = 'Apple';
    part = 'Leaf';
  } else if (sampleType === 'tomato_late_blight_fruit') {
    crop = 'Tomato';
    part = 'Fruit';
  } else if (sampleType === 'grape_black_rot') {
    crop = 'Grape';
    part = 'Fruit';
  } else if (sampleType === 'rice_leaf_blast') {
    crop = 'Rice';
    part = 'Leaf';
  }

  selectedCrop = crop;
  selectedPart = part;
  renderSelectors();

  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  if (sampleType === 'tomato_late_blight_fruit') {
    const grad = ctx.createRadialGradient(112, 112, 20, 112, 112, 90);
    grad.addColorStop(0, '#fef08a');
    grad.addColorStop(0.7, '#84cc16');
    grad.addColorStop(1, '#4d7c0f');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(112, 112, 75, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#3f2e1a';
    ctx.beginPath();
    ctx.arc(95, 95, 25, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#291b10';
    ctx.beginPath();
    ctx.arc(140, 125, 18, 0, Math.PI * 2);
    ctx.fill();
  } else {
    const grad = ctx.createRadialGradient(112, 112, 20, 112, 112, 110);
    grad.addColorStop(0, '#34d399');
    grad.addColorStop(0.7, '#059669');
    grad.addColorStop(1, '#064e3b');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 224, 224);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(112, 20);
    ctx.lineTo(112, 204);
    ctx.stroke();
  }

  canvas.toBlob((blob) => {
    const file = new File([blob], `${sampleType}.jpg`, { type: 'image/jpeg' });
    handleFileSelect(file);
  }, 'image/jpeg');
}

async function runDiagnosis() {
  if (!selectedFile) return;

  stopSpeech();
  const btn = document.getElementById('btnAnalyze');
  const spinner = document.getElementById('analyzeSpinner');
  const btnText = document.getElementById('btnAnalyzeText');
  const s = UI_STRINGS[currentLanguage] || UI_STRINGS.en;

  btn.disabled = true;
  spinner.style.display = 'inline-block';
  btnText.textContent = s.diagnosing;

  const formData = new FormData();
  formData.append('file', selectedFile);
  if (selectedCrop) {
    formData.append('crop_hint', selectedCrop);
  }
  if (selectedPart) {
    formData.append('plant_part_hint', selectedPart);
  }
  formData.append('language', currentLanguage);

  try {
    const res = await fetch('/api/v1/predict', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Diagnosis failed');
    }

    const data = await res.json();
    window.currentResult = data;
    renderResults(data);

    // Persist scan into session telemetry
    const previewImg = document.getElementById('previewImg');
    saveScanToHistory({
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      date: new Date().toLocaleDateString(),
      crop: data.crop || 'Plant',
      disease: data.display_name || data.predicted_class,
      canonical: data.predicted_class,
      confidence: data.confidence,
      health_score: data.plant_health_score !== undefined ? data.plant_health_score : 50,
      plant_part: data.plant_part || 'Leaf',
      is_healthy: data.is_healthy,
      thumbnail: previewImg ? previewImg.src : null,
      result_data: data
    });

    // Smooth scroll to results
    const resCard = document.getElementById('resultsCard');
    if (resCard) {
      resCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  } catch (err) {
    alert('Diagnostic Error: ' + err.message);
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
    btnText.textContent = s.analyzeBtn;
  }
}

function triggerCropOverride() {
  const cropContainer = document.getElementById('cropPills');
  if (cropContainer) {
    cropContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    cropContainer.style.transition = 'box-shadow 0.3s ease';
    cropContainer.style.boxShadow = '0 0 0 2px #ffffff';
    setTimeout(() => {
      cropContainer.style.boxShadow = 'none';
    }, 1500);
  }
}

function renderResults(data) {
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';

  // Meta & Crop Mode
  document.getElementById('resCropTag').textContent = data.crop || 'Plant';

  const cropMode = document.getElementById('resCropModeTag');
  const overrideBtn = document.getElementById('btnOverrideCrop');
  const candidateBar = document.getElementById('resCropCandidateBar');

  if (cropMode) {
    if (data.crop_detected_automatically) {
      const cConf = data.crop_confidence ? ` (${(data.crop_confidence * 100).toFixed(1)}%)` : '';
      cropMode.textContent = `Auto-Detected${cConf}`;
      cropMode.className = 'crop-mode-badge auto-mode';
      if (overrideBtn) overrideBtn.style.display = 'inline-flex';
    } else {
      cropMode.textContent = 'Manual Selection';
      cropMode.className = 'crop-mode-badge manual-mode';
      if (overrideBtn) overrideBtn.style.display = 'none';
    }
  }

  // Uncertain Crop Candidates Bar (interactive 1-click confirmation)
  if (candidateBar) {
    if (data.candidate_crops && data.candidate_crops.length > 0 &&
      (data.predicted_class === 'uncertain_crop_detection' || data.status === 'uncertain' || (data.crop_confidence && data.crop_confidence < 0.60))) {
      candidateBar.style.display = 'block';
      candidateBar.innerHTML = `<div style="font-size:0.8rem; margin-bottom:0.45rem; color:#d4d4d4;">
        <strong>⚠️ Confirm Crop Species:</strong> Auto-detection proposed candidate matches below. Click your target crop to route disease inference:
      </div>`;
      const btnGroup = document.createElement('div');
      btnGroup.style.display = 'flex';
      btnGroup.style.gap = '0.5rem';
      btnGroup.style.flexWrap = 'wrap';
      data.candidate_crops.forEach(c => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'custom-pill';
        b.style.fontSize = '0.78rem';
        b.style.padding = '0.35rem 0.75rem';
        b.style.cursor = 'pointer';
        b.style.background = '#171717';
        b.style.border = '1px solid #525252';
        b.style.color = '#ffffff';
        b.innerHTML = `<strong>${escapeHtml(c.crop)}</strong> <span style="opacity:0.75;">(${(c.confidence * 100).toFixed(1)}%)</span>`;
        b.onclick = () => {
          selectedCrop = c.crop;
          renderSelectors();
          runDiagnosis();
        };
        btnGroup.appendChild(b);
      });
      candidateBar.appendChild(btnGroup);
    } else {
      candidateBar.style.display = 'none';
      candidateBar.innerHTML = '';
    }
  }

  document.getElementById('resDiseaseName').textContent = data.display_name || data.predicted_class;
  document.getElementById('resCanonical').textContent = data.predicted_class;
  document.getElementById('resConfidence').textContent = (data.confidence * 100).toFixed(1) + '%';
  document.getElementById('resPart').textContent = data.plant_part || 'Leaf';
  document.getElementById('resLatency').textContent = (data.inference_time_ms || 28) + ' ms';

  const summaryText = data.recommendation ? data.recommendation.summary : (data.message || '');
  document.getElementById('resSummary').textContent = summaryText;

  // Health Score & Radial Gauge
  const score = data.plant_health_score !== undefined ? data.plant_health_score : 50;
  document.getElementById('gaugeNum').textContent = score;
  const circle = document.getElementById('gaugeCircle');
  if (circle) {
    const offset = 264 - (score / 100) * 264;
    circle.style.strokeDashoffset = offset;
  }

  // Status Badge
  const s = UI_STRINGS[currentLanguage] || UI_STRINGS.en;
  const statusBadge = document.getElementById('resStatusBadge');
  if (data.status === 'uncertain' || data.confidence < 0.45 || data.predicted_class === 'uncertain_crop_detection') {
    statusBadge.textContent = s.statusUncertain || 'Uncertain';
    statusBadge.className = 'status-badge status-uncertain';
  } else if (data.is_healthy) {
    statusBadge.textContent = s.statusHealthy || 'Healthy Leaf';
    statusBadge.className = 'status-badge status-healthy';
  } else {
    statusBadge.textContent = s.statusDiseased || 'Foliar Disease';
    statusBadge.className = 'status-badge status-diseased';
  }

  // Side-by-Side Images & Lesion Detection
  const previewImg = document.getElementById('previewImg');
  const origCompare = document.getElementById('imgOriginalCompare');
  if (previewImg && previewImg.src && origCompare) {
    origCompare.src = previewImg.src;
  }

  window.showingBinaryMask = false;
  const lesion = data.lesion_analysis;
  const overlayImg = document.getElementById('imgOverlayCompare');
  const statCount = document.getElementById('statLesionCount');
  const statArea = document.getElementById('statAffectedArea');
  const statPixels = document.getElementById('statLesionPixels');
  const statLeaf = document.getElementById('statLeafPixels');
  const sevBadge = document.getElementById('resLesionSeverity');
  const methodNote = document.getElementById('lesionMethodNote');

  if (document.getElementById('btnToggleMaskText')) {
    document.getElementById('btnToggleMaskText').textContent = s.btnMaskView || 'View Binary Mask';
  }
  if (document.getElementById('lblHighlightImg')) {
    document.getElementById('lblHighlightImg').textContent = s.highlightImg || 'Detected Lesions (Red Tint)';
  }

  if (lesion && lesion.overlay_image_base64) {
    if (overlayImg) overlayImg.src = lesion.overlay_image_base64;
    if (statCount) statCount.textContent = lesion.lesion_count;
    if (statArea) statArea.textContent = (lesion.affected_area_percentage || 0).toFixed(1) + '%';
    if (statPixels) statPixels.textContent = (lesion.lesion_area_pixels || 0).toLocaleString();
    if (statLeaf) statLeaf.textContent = (lesion.leaf_area_pixels || 0).toLocaleString();
    if (sevBadge) {
      sevBadge.textContent = lesion.severity_level || (lesion.lesion_detected ? 'Detected' : 'None');
      const sevLow = (lesion.severity_level || '').toLowerCase();
      if (sevLow.includes('severe')) {
        sevBadge.className = 'caution-badge caution-high';
      } else if (sevLow.includes('moderate')) {
        sevBadge.className = 'caution-badge caution-moderate';
      } else {
        sevBadge.className = 'caution-badge caution-low';
      }
    }
    if (methodNote && lesion.method_note) {
      methodNote.textContent = lesion.method_note;
    }
  } else {
    if (overlayImg) overlayImg.src = previewImg ? previewImg.src : '';
    if (statCount) statCount.textContent = '0';
    if (statArea) statArea.textContent = '0.0%';
    if (statPixels) statPixels.textContent = '0';
    if (statLeaf) statLeaf.textContent = 'N/A';
    if (sevBadge) {
      sevBadge.textContent = 'None';
      sevBadge.className = 'caution-badge caution-low';
    }
  }

  // Disclaimer
  const disc = data.recommendation ? data.recommendation.disclaimer : '';
  const discElem = document.getElementById('resDisclaimer');
  if (discElem) discElem.textContent = disc;

  // Render Active Tab Content
  renderTreatmentTabContent();
}

function toggleLesionMaskView() {
  if (!window.currentResult || !window.currentResult.lesion_analysis) return;
  const lesion = window.currentResult.lesion_analysis;
  const overlayImg = document.getElementById('imgOverlayCompare');
  const toggleBtn = document.getElementById('btnToggleMaskText');
  const lbl = document.getElementById('lblHighlightImg');
  const s = UI_STRINGS[currentLanguage] || UI_STRINGS.en;
  const previewImg = document.getElementById('previewImg');

  window.showingBinaryMask = !window.showingBinaryMask;
  if (window.showingBinaryMask && lesion.mask_image_base64) {
    if (overlayImg) overlayImg.src = lesion.mask_image_base64;
    if (toggleBtn) toggleBtn.textContent = s.btnOverlayView || 'View Red Overlay';
    if (lbl) lbl.textContent = s.maskImg || 'Binary Lesion Mask (Black/White)';
  } else {
    if (overlayImg) overlayImg.src = lesion.overlay_image_base64 || (previewImg ? previewImg.src : '');
    if (toggleBtn) toggleBtn.textContent = s.btnMaskView || 'View Binary Mask';
    if (lbl) lbl.textContent = s.highlightImg || 'Detected Lesions (Red Tint)';
  }
}

function updateVoiceRate(rateVal) {
  voiceRate = parseFloat(rateVal) || 1.0;
  try {
    localStorage.setItem('plantscanai_voice_rate', voiceRate);
  } catch (e) {}
}

function speakAdvisory() {
  if (!window.currentResult) return;
  if (!('speechSynthesis' in window)) {
    alert('Speech Synthesis is not supported in this browser.');
    return;
  }

  stopSpeech();

  const rec = window.currentResult.recommendation;
  const topOrganic = (rec && rec.organic && rec.organic[0]) ? rec.organic[0].name : '';
  let textToSpeak = '';

  if (currentLanguage === 'kn') {
    textToSpeak = `PlantScan AI ರೋಗನಿರ್ಣಯ: ಬೆಳೆ ${window.currentResult.crop}. ` +
      `ಪತ್ತೆಯಾದ ರೋಗ: ${window.currentResult.display_name}. ` +
      `ಸಸ್ಯದ ಭಾಗ: ${window.currentResult.plant_part}. ` +
      `ಸಸ್ಯದ ಆರೋಗ್ಯ ಸ್ಕೋರ್: ನೂರಕ್ಕೆ ${window.currentResult.plant_health_score}. ` +
      `${rec ? rec.summary : ''}. ` +
      (topOrganic ? `ಮುಖ್ಯ ಜೈವಿಕ ಪರಿಹಾರ: ${topOrganic}.` : '');
  } else if (currentLanguage === 'hi') {
    textToSpeak = `PlantScan AI पौध रोग निदान: फसल ${window.currentResult.crop}. ` +
      `पहचाना गया रोग: ${window.currentResult.display_name}. ` +
      `पौधे का अंग: ${window.currentResult.plant_part}. ` +
      `पौध स्वास्थ्य स्कोर: 100 में से ${window.currentResult.plant_health_score}. ` +
      `${rec ? rec.summary : ''}. ` +
      (topOrganic ? `मुख्य जैविक उपाय: ${topOrganic}.` : '');
  } else {
    textToSpeak = `PlantScan AI Diagnostic Report: Crop ${window.currentResult.crop}. ` +
      `Identified condition: ${window.currentResult.display_name}. ` +
      `Plant part: ${window.currentResult.plant_part}. ` +
      `Plant health score: ${window.currentResult.plant_health_score} out of 100. ` +
      `${rec ? rec.summary : ''}. ` +
      (topOrganic ? `Primary organic measure: ${topOrganic}.` : '');
  }

  speechUtterance = new SpeechSynthesisUtterance(textToSpeak);
  speechUtterance.rate = voiceRate;

  const voices = speechSynth.getVoices();
  let matchedVoice = null;
  if (currentLanguage === 'kn') {
    matchedVoice = voices.find(v => v.lang.includes('kn') || v.lang.includes('kannada'));
  } else if (currentLanguage === 'hi') {
    matchedVoice = voices.find(v => v.lang.includes('hi') || v.lang.includes('hindi'));
  } else {
    matchedVoice = voices.find(v => v.lang.includes('en-IN') || v.lang.includes('en-US') || v.lang.includes('en'));
  }
  if (matchedVoice) {
    speechUtterance.voice = matchedVoice;
  }
  speechUtterance.lang = currentLanguage === 'kn' ? 'kn-IN' : (currentLanguage === 'hi' ? 'hi-IN' : 'en-US');

  const btnStop = document.getElementById('btnVoiceStop');
  const waves = document.getElementById('audioWaves');
  if (btnStop) btnStop.style.display = 'inline-flex';
  if (waves) waves.style.display = 'flex';

  speechUtterance.onend = () => stopSpeech();
  speechUtterance.onerror = () => stopSpeech();

  speechSynth.speak(speechUtterance);
}

function stopSpeech() {
  if (speechSynth && speechSynth.speaking) {
    speechSynth.cancel();
  }
  const btnStop = document.getElementById('btnVoiceStop');
  const waves = document.getElementById('audioWaves');
  if (btnStop) btnStop.style.display = 'none';
  if (waves) waves.style.display = 'none';
}

function switchTreatmentTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.treatment-tab-btn').forEach((btn, idx) => {
    const matches = (tab === 'organic' && idx === 0) ||
      (tab === 'chemical' && idx === 1) ||
      (tab === 'prevention' && idx === 2);
    btn.classList.toggle('active', matches);
  });
  renderTreatmentTabContent();
}

function renderTreatmentTabContent() {
  if (!window.currentResult || !window.currentResult.recommendation) return;

  const container = document.getElementById('treatmentList');
  if (!container) return;
  container.innerHTML = '';
  const rec = window.currentResult.recommendation;

  if (activeTab === 'organic') {
    const items = rec.organic || [];
    if (items.length === 0) {
      const emptyText = currentLanguage === 'kn'
        ? 'ಈ ಸ್ಥಿತಿಗೆ ಯಾವುದೇ ನಿರ್ದಿಷ್ಟ ಜೈವಿಕ ಮಧ್ಯಸ್ಥಿಕೆಗಳ ಅಗತ್ಯವಿಲ್ಲ.'
        : (currentLanguage === 'hi' ? 'इस स्थिति के लिए किसी विशिष्ट जैविक उपाय की आवश्यकता नहीं है।' : 'No specific organic interventions required for this plant state.');
      container.innerHTML = `<div style="color:var(--text-muted); padding:1rem; text-align:center;">${emptyText}</div>`;
      return;
    }
    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'treatment-item';
      div.innerHTML = `
        <div class="treatment-item-title">
          <span>🌿 ${escapeHtml(item.name)}</span>
          <span class="caution-badge caution-low">${escapeHtml(item.caution_level || 'Safe')}</span>
        </div>
        <div class="treatment-item-desc">${escapeHtml(item.method || 'Apply as per package instructions.')}</div>
        <div class="treatment-meta-tags">
          ${item.application ? `<span class="meta-tag">App: ${escapeHtml(item.application)}</span>` : ''}
          ${item.frequency ? `<span class="meta-tag">Freq: ${escapeHtml(item.frequency)}</span>` : ''}
        </div>
      `;
      container.appendChild(div);
    });
  } else if (activeTab === 'chemical') {
    const items = rec.chemical || [];
    if (items.length === 0) {
      const safeText = currentLanguage === 'kn'
        ? '✅ ಯಾವುದೇ ರಾಸಾಯನಿಕ ಕೀಟನಾಶಕಗಳ ಅಗತ್ಯವಿಲ್ಲ. ಜೈವಿಕ ಮತ್ತು ಸಾಂಸ್ಕೃತಿಕ ಪದ್ಧತಿಗಳನ್ನು ಮುಂದುವರಿಸಿ.'
        : (currentLanguage === 'hi' ? '✅ किसी रासायनिक कीटनाशक की आवश्यकता नहीं है। जैविक व निवारक उपाय जारी रखें।' : '✅ No synthetic chemical pesticides recommended. Maintain organic & cultural practices.');
      container.innerHTML = `<div style="color:#ffffff; padding:1rem; text-align:center;">${safeText}</div>`;
      return;
    }
    items.forEach(item => {
      const cautionClass = item.caution_level && item.caution_level.includes('High') ? 'caution-high' : 'caution-moderate';
      const div = document.createElement('div');
      div.className = 'treatment-item';
      div.innerHTML = `
        <div class="treatment-item-title">
          <span>🧪 ${escapeHtml(item.name)}</span>
          <span class="caution-badge ${cautionClass}">${escapeHtml(item.caution_level || 'Moderate')}</span>
        </div>
        <div class="treatment-item-desc">
          ${item.dosage ? `<strong>Dosage:</strong> ${escapeHtml(item.dosage)}<br>` : ''}
          ${item.safety_precautions ? `<strong>Safety/PPE:</strong> ${escapeHtml(item.safety_precautions)}` : ''}
        </div>
        <div class="treatment-meta-tags">
          ${item.active_ingredient ? `<span class="meta-tag">Active: ${escapeHtml(item.active_ingredient)}</span>` : ''}
        </div>
      `;
      container.appendChild(div);
    });
  } else if (activeTab === 'prevention') {
    const items = rec.prevention || [];
    if (items.length === 0) {
      container.innerHTML = '<div style="color:var(--text-muted); padding:1rem; text-align:center;">Maintain general crop field hygiene and proper irrigation spacing.</div>';
      return;
    }
    items.forEach(tip => {
      const div = document.createElement('div');
      div.className = 'treatment-item';
      div.innerHTML = `
        <div class="treatment-item-title">
          <span>🛡️ Agronomic Prevention</span>
        </div>
        <div class="treatment-item-desc">${escapeHtml(tip)}</div>
      `;
      container.appendChild(div);
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initActiveNav('scan');
  renderSelectors();
  initDragAndDrop();
  updateTelemetryMetrics();

  // Check for historical inspection request in URL query params: ?inspect=<id>
  const urlParams = new URLSearchParams(window.location.search);
  const inspectId = urlParams.get('inspect');
  if (inspectId) {
    const history = getSessionHistory();
    const found = history.find(h => String(h.id) === String(inspectId));
    if (found && found.result_data) {
      window.currentResult = found.result_data;
      if (found.thumbnail) {
        const prevImg = document.getElementById('previewImg');
        if (prevImg) prevImg.src = found.thumbnail;
        const dropzone = document.getElementById('dropzone');
        if (dropzone) dropzone.style.display = 'none';
        const prevContainer = document.getElementById('previewContainer');
        if (prevContainer) prevContainer.style.display = 'block';
      }
      renderResults(found.result_data);
    }
  }
});
