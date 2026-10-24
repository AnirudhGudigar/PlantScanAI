/**
 * PlantScanAI - Analysis History Script
 * Comprehensive session scan telemetry table, record clearing, and inspector routing.
 */

function renderAnalysisHistoryTable() {
  const container = document.getElementById('fullHistoryContainer');
  if (!container) return;
  const history = getSessionHistory();

  if (history.length === 0) {
    container.innerHTML = `
      <div class="empty-state-box">
        <div class="empty-state-icon">⏱️</div>
        <div class="empty-state-title">Analysis History Is Empty</div>
        <div class="empty-state-desc">Every diagnosis executed through the live inference engine will appear here with complete telemetry and lesion statistics.</div>
        <a href="/app/scan" class="btn-primary" style="display:inline-block; padding:0.6rem 1.2rem; font-size:0.82rem; text-decoration:none;">Start First Scan</a>
      </div>
    `;
    return;
  }

  let rowsHtml = '';
  history.forEach((item, idx) => {
    const statusClass = item.is_healthy ? 'status-healthy' : 'status-diseased';
    rowsHtml += `
      <tr>
        <td style="font-family:'JetBrains Mono',monospace; color:#808080;">#${history.length - idx}</td>
        <td>
          ${item.thumbnail ? `<img src="${item.thumbnail}" class="table-thumb" alt="Thumb">` : '—'}
        </td>
        <td><span class="table-crop-tag">${escapeHtml(item.crop)}</span></td>
        <td style="font-weight:600; color:#ffffff;">${escapeHtml(item.disease)}</td>
        <td><span class="status-badge ${statusClass}">${item.is_healthy ? 'Healthy' : 'Diseased'}</span></td>
        <td style="font-family:'JetBrains Mono',monospace; font-weight:700;">${(item.confidence * 100).toFixed(1)}%</td>
        <td style="font-family:'JetBrains Mono',monospace;">${item.health_score}/100</td>
        <td style="font-family:'JetBrains Mono',monospace; color:#808080;">${item.date} ${item.timestamp}</td>
        <td>
          <button class="btn-secondary" style="padding:0.3rem 0.65rem; font-size:0.75rem;" onclick="inspectHistoricalRecord(${item.id})">View in Inspector</button>
        </td>
      </tr>
    `;
  });

  container.innerHTML = `
    <div class="data-table-container">
      <table class="mono-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Leaf</th>
            <th>Crop</th>
            <th>Diagnosis</th>
            <th>Status</th>
            <th>Confidence</th>
            <th>Health</th>
            <th>Date / Time</th>
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

document.addEventListener('DOMContentLoaded', () => {
  initActiveNav('history');
  updateTelemetryMetrics();
  renderAnalysisHistoryTable();
});
