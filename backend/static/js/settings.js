/**
 * PlantScanAI - Settings & Configuration Script
 * Vernacular speech options, backend diagnostics ping, and telemetry controls.
 */

function updateVoiceRate(rateVal) {
  try {
    localStorage.setItem('plantscanai_voice_rate', rateVal);
  } catch (e) {}
}

function initSettings() {
  // Voice engine check
  const speechBadge = document.getElementById('speechSupportBadge');
  if (speechBadge) {
    if ('speechSynthesis' in window) {
      speechBadge.textContent = 'Active (Web Speech API)';
      speechBadge.className = 'dash-nav-badge status-healthy';
    } else {
      speechBadge.textContent = 'Not Supported';
      speechBadge.className = 'dash-nav-badge status-diseased';
    }
  }

  // Set selected rate
  const selRate = document.getElementById('selVoiceRate');
  if (selRate) {
    const savedRate = localStorage.getItem('plantscanai_voice_rate') || '1.0';
    selRate.value = savedRate;
  }

  // Update language buttons active state in settings
  const lang = localStorage.getItem(LANG_STORAGE_KEY) || 'en';
  document.querySelectorAll('.settings-row .lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.id.toLowerCase().endsWith(lang));
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initActiveNav('settings');
  updateTelemetryMetrics();
  initSettings();
});
