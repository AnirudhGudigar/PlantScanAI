/**
 * PlantScanAI - Disease Insights Script
 * 10-Crop pathology reference matrix and pathogen profile cards.
 */

function renderDiseaseInsightsMatrix() {
  const grid = document.getElementById('insightsGrid');
  if (!grid) return;
  grid.innerHTML = '';

  CROP_INSIGHTS_DATA.forEach(item => {
    const card = document.createElement('div');
    card.className = 'crop-insight-card';
    card.innerHTML = `
      <div class="crop-insight-header">
        <div class="crop-insight-name">
          <span>${item.icon}</span> <span>${item.crop}</span>
        </div>
        <span class="dash-nav-badge">${item.pathogens.length} Classes</span>
      </div>
      <div style="font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; font-weight:600; margin-bottom:0.35rem;">
        Recognized Pathogens
      </div>
      <ul class="crop-disease-list">
        ${item.pathogens.map(p => `
          <li class="crop-disease-item">
            <span>${escapeHtml(p)}</span>
            <span style="font-size:0.68rem; color:#808080;">Neural</span>
          </li>
        `).join('')}
      </ul>
      <div style="margin-top:0.85rem; padding-top:0.65rem; border-top:1px solid var(--border-subtle); font-size:0.75rem; color:var(--text-secondary); line-height:1.45;">
        <strong>IPM Focus:</strong> ${escapeHtml(item.ipmRule)}
      </div>
    `;
    grid.appendChild(card);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initActiveNav('insights');
  updateTelemetryMetrics();
  renderDiseaseInsightsMatrix();
});
