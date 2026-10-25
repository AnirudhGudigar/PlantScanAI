/**
 * PlantScanAI - Overview Page Script
 * Telemetry KPI calculations and recent scan activity monitoring.
 */

function renderOverviewRecentActivity() {
  const recentContainer = document.getElementById('overviewRecentContainer');
  if (!recentContainer) return;
  const history = getSessionHistory();
  const count = history.length;

  if (count === 0) {
    recentContainer.innerHTML = `
      <div class="empty-state-box">
        <div class="empty-state-icon">📋</div>
        <div class="empty-state-title">No Scans Recorded Yet</div>
        <div class="empty-state-desc">Upload a plant leaf photo in the scanner or run verified sample cases to log diagnostic metrics.</div>
        <a href="/app/scan" class="btn-primary" style="display:inline-block; padding:0.6rem 1.2rem; font-size:0.82rem; text-decoration:none;">Launch Scanner</a>
      </div>
    `;
  } else {
    let rowsHtml = '';
    history.slice(0, 5).forEach((item) => {
      const statusClass = item.is_healthy ? 'status-healthy' : 'status-diseased';
      rowsHtml += `
        <tr>
          <td><span class="table-crop-tag">${escapeHtml(item.crop)}</span></td>
          <td style="font-weight:600; color:#ffffff;">${escapeHtml(item.disease)}</td>
          <td><span class="status-badge ${statusClass}">${item.is_healthy ? 'Healthy' : 'Diseased'}</span></td>
          <td style="font-family:'JetBrains Mono',monospace;">${(item.confidence * 100).toFixed(1)}%</td>
          <td style="font-family:'JetBrains Mono',monospace; color:#808080;">${item.timestamp}</td>
          <td>
            <button class="btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.72rem;" onclick="inspectHistoricalRecord(${item.id})">Inspect</button>
          </td>
        </tr>
      `;
    });

    recentContainer.innerHTML = `
      <div class="data-table-container">
        <table class="mono-table">
          <thead>
            <tr>
              <th>Crop</th>
              <th>Diagnosis</th>
              <th>Status</th>
              <th>Confidence</th>
              <th>Time</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      </div>
    `;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initActiveNav('overview');
  updateTelemetryMetrics();
  renderOverviewRecentActivity();
});
