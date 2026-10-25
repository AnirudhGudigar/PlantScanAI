/**
 * PlantScanAI - Shared Core Application Script
 * Provides global telemetry storage, multilingual localization, navigation active states,
 * floating dock animations, and shared IPM data across all pages.
 */

// Storage Keys
const HISTORY_STORAGE_KEY = 'plantscanai_session_history_v1';
const LANG_STORAGE_KEY = 'plantscanai_selected_language';

// 10-Crop Disease Insights & IPM Matrix Data
const CROP_INSIGHTS_DATA = [
  {
    crop: "Tomato",
    icon: "🍅",
    pathogens: ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot", "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus", "Mosaic Virus"],
    ipmRule: "Trichoderma harzianum or Bacillus subtilis foliar spray; copper hydroxide strictly if bacterial spot exceeds 15% leaf area."
  },
  {
    crop: "Apple",
    icon: "🍎",
    pathogens: ["Apple Scab (Venturia)", "Black Rot (Botryosphaeria)", "Cedar Apple Rust (Gymnosporangium)", "Healthy"],
    ipmRule: "Sulfur dusts or neem oil bio-fungicide during early spring bud break; destroy overwintering mummified fruit."
  },
  {
    crop: "Grape",
    icon: "🍇",
    pathogens: ["Black Rot (Guignardia)", "Esca / Black Measles", "Leaf Blight (Isariopsis)", "Healthy"],
    ipmRule: "Pruning infected canes 15cm below symptom margins; bio-fungicidal Ampelomyces quisqualis."
  },
  {
    crop: "Rice",
    icon: "🌾",
    pathogens: ["Leaf Blast (Magnaporthe)", "Brown Spot (Bipolaris)", "Neck Blast", "Healthy"],
    ipmRule: "Pseudomonas fluorescens seed treatment (10g/kg); split nitrogen application to prevent succulent foliar tissue."
  },
  {
    crop: "Potato",
    icon: "🥔",
    pathogens: ["Early Blight (Alternaria)", "Late Blight (Phytophthora)", "Healthy"],
    ipmRule: "Certified disease-free tubers; avoid overhead irrigation; copper oxychloride as emergency shield."
  },
  {
    crop: "Maize",
    icon: "🌽",
    pathogens: ["Common Rust (Puccinia)", "Gray Leaf Spot (Cercospora)", "Northern Corn Leaf Blight (Exserohilum)", "Healthy"],
    ipmRule: "Crop rotation with non-gramineous pulses; Bacillus amyloliquefaciens foliar spray."
  },
  {
    crop: "Bell Pepper",
    icon: "🫑",
    pathogens: ["Bacterial Spot (Xanthomonas)", "Healthy"],
    ipmRule: "Streptomycin sulphate seed soak; copper soap spray; avoid field work when foliage is wet."
  },
  {
    crop: "Banana",
    icon: "🍌",
    pathogens: ["Cordana Leaf Spot", "Panama Wilt (Fusarium)", "Healthy"],
    ipmRule: "Clean sucker selection; de-trashing dry diseased leaves; Trichoderma viride enriched compost."
  },
  {
    crop: "Mango",
    icon: "🥭",
    pathogens: ["Anthracnose (Colletotrichum)", "Healthy"],
    ipmRule: "Hot water dip (52°C for 5 min) for fruits; orchard sanitation and canopy pruning for sunlight penetration."
  },
  {
    crop: "Cotton",
    icon: "☁️",
    pathogens: ["Bacterial Blight (Xanthomonas)", "Healthy"],
    ipmRule: "Delinting seed with concentrated sulfuric acid; foliar spray of Paenibacillus macerans bio-control."
  }
];

// Multilingual Localization Strings
const UI_STRINGS = {
  en: {
    navTabHome: "Home",
    navTabScanner: "Scan Plant",
    navTabOverview: "Overview",
    navTabHistory: "Analysis History",
    navTabInsights: "Disease Insights",
    navTabSettings: "Settings",
    heroTag: "“See the damage. <span class='hero-tagline-highlight'>Understand the plant.”</span>",
    heroSub: "Automated deep-learning mobile & cloud pathology covering 10 major commercial crops with dual-stage foliar lesion segmentation and prioritized non-chemical integrated pest management.",
    uploadTitle: "Foliar Pathology &amp; Lesion Scanner",
    cropSelect: "Target Crop (Optional Hint):",
    partSelect: "Plant Part (Auto or Select):",
    dropText: "Drag & drop plant photo here, or browse",
    dropSub: "Supports JPEG, PNG, WebP (Max 10 MB)",
    samples: "Or test with verified sample cases:",
    analyzeBtn: "Run AI Diagnosis",
    diagnosing: "Diagnosing...",
    awaitingTitle: "Awaiting Plant Image",
    awaitingSub: "Upload an affected plant leaf or fruit to inspect instant crop routing, disease classification, foliar lesion boundaries, and IPM protocols.",
    tabOrg: "Organic (Priority 1)",
    tabChem: "Chemical",
    tabPrev: "Prevention",
    gaugeLabel: "Health",
    lblConfidence: "Confidence",
    lblPart: "Plant Part",
    lblLatency: "Latency",
    voiceBtn: "Listen to Advisory",
    voiceStop: "Stop",
    lesionTitle: "Foliar Lesion Detection &amp; Highlighting",
    originalImg: "Original Leaf Photo",
    highlightImg: "Detected Lesions (Red Tint)",
    maskImg: "Binary Lesion Mask (Black/White)",
    lblStatCount: "Lesions Found",
    lblStatArea: "Leaf Area Affected",
    lblStatPixels: "Lesion Pixels",
    lblStatLeaf: "Leaf Mask Pixels",
    btnMaskView: "View Binary Mask",
    btnOverlayView: "View Red Overlay",
    statusHealthy: "Healthy Leaf",
    statusDiseased: "Foliar Disease",
    statusUncertain: "Uncertain / Low Confidence",
    crops: {
      "": "🌐 Auto Detect",
      "Tomato": "🍅 Tomato",
      "Apple": "🍎 Apple",
      "Grape": "🍇 Grape",
      "Rice": "🌾 Rice",
      "Banana": "🍌 Banana",
      "Mango": "🥭 Mango",
      "Potato": "🥔 Potato",
      "Bell pepper": "🫑 Pepper",
      "Cotton": "☁️ Cotton",
      "Maize": "🌽 Maize"
    },
    parts: {
      "": "✨ Auto Detect",
      "Leaf": "🍃 Leaf",
      "Fruit": "🍅 Fruit",
      "Stem": "🪵 Stem"
    },
    samplesList: [
      { type: 'apple_healthy', icon: '🍏', title: 'Apple Healthy' },
      { type: 'tomato_late_blight_fruit', icon: '🍅', title: 'Tomato Late Blight (Fruit)' },
      { type: 'grape_black_rot', icon: '🍇', title: 'Grape Rot' },
      { type: 'rice_leaf_blast', icon: '🌾', title: 'Rice Blast' }
    ]
  },
  kn: {
    navTabHome: "ಮುಖಪುಟ",
    navTabScanner: "ಸಸ್ಯ ತಪಾಸಣೆ",
    navTabOverview: "ಅವಲೋಕನ",
    navTabHistory: "ವಿಶ್ಲೇಷಣೆ ಇತಿಹಾಸ",
    navTabInsights: "ರೋಗ ಮಾಹಿತಿ",
    navTabSettings: "ಸಂಯೋಜನೆಗಳು",
    heroTag: "“ಹಾನಿಯನ್ನು ನೋಡಿ. <span class='hero-tagline-highlight'>ಸಸ್ಯವನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ.”</span>",
    heroSub: "10 ಪ್ರಮುಖ ವಾಣಿಜ್ಯ ಬೆಳೆಗಳ ರೋಗ ಪತ್ತೆ, ಎಲೆ ಗಾಯ ವಿಭಾಗೀಕರಣ ಮತ್ತು ಸುಸ್ಥಿರ ಸಾವಯವ ಕೀಟ ನಿರ್ವಹಣಾ (IPM) ವ್ಯವಸ್ಥೆ.",
    uploadTitle: "ಎಲೆ ಮತ್ತು ಕಾಯಿ ರೋಗ ತಪಾಸಕ",
    cropSelect: "ಗುರಿ ಬೆಳೆ (ಐಚ್ಛಿಕ ಸುಳಿವು):",
    partSelect: "ಸಸ್ಯದ ಭಾಗ (ಸ್ವಯಂ ಅಥವಾ ಆಯ್ಕೆಮಾಡಿ):",
    dropText: "ಎಲೆ ಅಥವಾ ಹಣ್ಣಿನ ಫೋಟೋವನ್ನು ಇಲ್ಲಿ ಎಳೆಯಿರಿ, ಅಥವಾ ಆಯ್ಕೆಮಾಡಿ",
    dropSub: "JPEG, PNG, WebP ಬೆಂಬಲಿತವಾಗಿದೆ (ಗರಿಷ್ಠ 10 MB)",
    samples: "ಅಥವಾ ಪರಿಶೀಲಿಸಿದ ಮಾದರಿ ಪ್ರಕರಣಗಳೊಂದಿಗೆ ಪರೀಕ್ಷಿಸಿ:",
    analyzeBtn: "AI ರೋಗನಿರ್ಣಯ ಪ್ರಾರಂಭಿಸಿ",
    diagnosing: "ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
    awaitingTitle: "ಸಸ್ಯದ ಫೋಟೋಗಾಗಿ ಕಾಯಲಾಗುತ್ತಿದೆ",
    awaitingSub: "ರೋಗ ವರ್ಗೀಕರಣ, ಸಸ್ಯ ಆರೋಗ್ಯ ಸ್ಕೋರ್, ಗಾಯದ ವ್ಯಾಪ್ತಿ ಮತ್ತು ಪರಿಹಾರಗಳನ್ನು ವೀಕ್ಷಿಸಲು ಎಲೆ ಅಥವಾ ಕಾಯಿಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",
    tabOrg: "ಜೈವಿಕ / ಸಾವಯವ (ಆದ್ಯತೆ 1)",
    tabChem: "ರಾಸಾಯನಿಕ ಪರಿಹಾರಗಳು",
    tabPrev: "ಮುನ್ನೆಚ್ಚರಿಕೆ ಕ್ರಮಗಳು",
    gaugeLabel: "ಆರೋಗ್ಯ",
    lblConfidence: "ನಿಖರತೆ",
    lblPart: "ಸಸ್ಯದ ಭಾಗ",
    lblLatency: "ವೇಗ",
    voiceBtn: "ಧ್ವನಿ ಸಲಹೆ ಕೇಳಿ",
    voiceStop: "ನಿಲ್ಲಿಸಿ",
    lesionTitle: "ಎಲೆ ಗಾಯ ಮತ್ತು ಹಾನಿ ಪತ್ತೆ (Lesion Detection)",
    originalImg: "ಮೂಲ ಎಲೆಯ ಫೋಟೋ",
    highlightImg: "ಪತ್ತೆಯಾದ ರೋಗದ ಗಾಯಗಳು (ಕೆಂಪು ಬಣ್ಣ)",
    maskImg: "ಬೈನರಿ ಲೆಸಿಯನ್ ಮಾಸ್ಕ್ (Black/White)",
    lblStatCount: "ಪತ್ತೆಯಾದ ಗಾಯಗಳು",
    lblStatArea: "ಬಾಧಿತ ಎಲೆ ಪ್ರದೇಶ",
    lblStatPixels: "ಗಾಯದ ಪಿಕ್ಸೆಲ್‌ಗಳು",
    lblStatLeaf: "ಎಲೆಯ ಒಟ್ಟು ಪಿಕ್ಸೆಲ್‌ಗಳು",
    btnMaskView: "ಬೈನರಿ ಮಾಸ್ಕ್ ನೋಡಿ",
    btnOverlayView: "ಕೆಂಪು ಹೈಲೈಟ್ ನೋಡಿ",
    statusHealthy: "ಆರೋಗ್ಯಕರ ಎಲೆ",
    statusDiseased: "ರೋಗ ಬಾಧಿತ ಎಲೆ",
    statusUncertain: "ಅನಿಶ್ಚಿತ / ಕಡಿಮೆ ವಿಶ್ವಾಸ",
    crops: {
      "": "🌐 ಸ್ವಯಂ ಪತ್ತೆ",
      "Tomato": "🍅 ಟೊಮೆಟೊ",
      "Apple": "🍎 ಸೇಬು",
      "Grape": "🍇 ದ್ರಾಕ್ಷಿ",
      "Rice": "🌾 ಭತ್ತ",
      "Banana": "🍌 ಬಾಳೆ",
      "Mango": "🥭 ಮಾವು",
      "Potato": "🥔 ಆಲೂಗಡ್ಡೆ",
      "Bell pepper": "🫑 ಕ್ಯಾಪ್ಸಿಕಂ",
      "Cotton": "☁️ ಹತ್ತಿ",
      "Maize": "🌽 ಮೆಕ್ಕೆಜೋಳ"
    },
    parts: {
      "": "✨ ಸ್ವಯಂ ಪತ್ತೆ",
      "Leaf": "🍃 ಎಲೆ (Leaf)",
      "Fruit": "🍅 ಹಣ್ಣು / ಕಾಯಿ (Fruit)",
      "Stem": "🪵 ಕಾಂಡ (Stem)"
    },
    samplesList: [
      { type: 'apple_healthy', icon: '🍏', title: 'ಆರೋಗ್ಯಕರ ಸೇಬು' },
      { type: 'tomato_late_blight_fruit', icon: '🍅', title: 'ಟೊಮೆಟೊ ಲೇಟ್ ಬ್ಲೈಟ್ (ಹಣ್ಣು)' },
      { type: 'grape_black_rot', icon: '🍇', title: 'ದ್ರಾಕ್ಷಿ ಕಪ್ಪು ಕೊಳೆ' },
      { type: 'rice_leaf_blast', icon: '🌾', title: 'ಭತ್ತದ ಬೆಂಕಿ ರೋಗ' }
    ]
  },
  hi: {
    navTabHome: "होम",
    navTabScanner: "पौधा स्कैन",
    navTabOverview: "अवलोकन",
    navTabHistory: "इतिहास",
    navTabInsights: "रोग अंतर्दृष्टि",
    navTabSettings: "सेटिंग्स",
    heroTag: "“क्षति को देखें। <span class='hero-tagline-highlight'>पौधे को समझें।”</span>",
    heroSub: "10 प्रमुख फसलों के लिए स्वचालित डीप-लर्निंग पौध रोग निदान, पत्ती घाव रेखांकन एवं जैविक एकीकृत कीट प्रबंधन प्रणाली।",
    uploadTitle: "पत्ती एवं फल रोग स्कैनर",
    cropSelect: "लक्षित फसल (वैकल्पिक संकेत):",
    partSelect: "पौधे का अंग (स्वतः या चुनें):",
    dropText: "पत्ती या फल की फोटो यहाँ खींचें या ब्राउज़ करें",
    dropSub: "JPEG, PNG, WebP समर्थित (अधिकतम 10 MB)",
    samples: "या सत्यापित नमूना मामलों से परीक्षण करें:",
    analyzeBtn: "AI निदान प्रारंभ करें",
    diagnosing: "विश्लेषण हो रहा है...",
    awaitingTitle: "पौधे की फोटो की प्रतीक्षा",
    awaitingSub: "तुरंत रोग वर्गीकरण, स्वास्थ्य स्कोर, घाव रेखांकन और उपचार देखने के लिए पत्ती या फल की फोटो अपलोड करें।",
    tabOrg: "जैविक उपाय (प्राथमिकता 1)",
    tabChem: "रासायनिक उपाय",
    tabPrev: "रोकथाम और बचाव",
    gaugeLabel: "स्वास्थ्य",
    lblConfidence: "सटीकता",
    lblPart: "पौधे का अंग",
    lblLatency: "गति",
    voiceBtn: "आवाज़ में सुनें",
    voiceStop: "रोकें",
    lesionTitle: "पत्ती घाव एवं क्षति पहचान (Lesion Detection)",
    originalImg: "मूल पत्ती की तस्वीर",
    highlightImg: "पहचाने गए घाव (लाल रंग)",
    maskImg: "बाइनरी लीजन मास्क (Black/White)",
    lblStatCount: "पहचाने गए घाव",
    lblStatArea: "प्रभावित पत्ती क्षेत्र",
    lblStatPixels: "घाव पिक्सेल",
    lblStatLeaf: "पत्ती का कुल पिक्सेल",
    btnMaskView: "बाइनरी मास्क देखें",
    btnOverlayView: "लाल ओवरले देखें",
    statusHealthy: "स्वस्थ पत्ती",
    statusDiseased: "रोगग्रस्त पत्ती",
    statusUncertain: "अनिश्चित / कम विश्वास",
    crops: {
      "": "🌐 स्वतः पहचानें",
      "Tomato": "🍅 टमाटर",
      "Apple": "🍎 सेब",
      "Grape": "🍇 अंगूर",
      "Rice": "🌾 धान",
      "Banana": "🍌 केला",
      "Mango": "🥭 आम",
      "Potato": "🥔 आलू",
      "Bell pepper": "🫑 शिमला मिर्च",
      "Cotton": "☁️ कपास",
      "Maize": "🌽 मक्का"
    },
    parts: {
      "": "✨ स्वतः पहचानें",
      "Leaf": "🍃 पत्ती (Leaf)",
      "Fruit": "🍅 फल (Fruit)",
      "Stem": "🪵 तना (Stem)"
    },
    samplesList: [
      { type: 'apple_healthy', icon: '🍏', title: 'स्वस्थ सेब' },
      { type: 'tomato_late_blight_fruit', icon: '🍅', title: 'टमाटर पछेती झुलसा (फल)' },
      { type: 'grape_black_rot', icon: '🍇', title: 'अंगूर काला सड़न' },
      { type: 'rice_leaf_blast', icon: '🌾', title: 'धान झुलसा रोग' }
    ]
  }
};

let currentLanguage = localStorage.getItem(LANG_STORAGE_KEY) || 'en';

// Telemetry Storage & History Management
function getSessionHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (err) {
    console.warn('Could not read history from localStorage:', err);
    return [];
  }
}

function saveScanToHistory(entry) {
  try {
    const list = getSessionHistory();
    list.unshift(entry);
    if (list.length > 50) list.pop(); // Keep 50 most recent records
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(list));
    updateTelemetryMetrics();
  } catch (err) {
    console.warn('Could not persist scan to localStorage:', err);
  }
}

function clearAnalysisHistory() {
  if (confirm('Are you sure you want to clear session scan history?')) {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
    updateTelemetryMetrics();
    if (typeof renderAnalysisHistoryTable === 'function') {
      renderAnalysisHistoryTable();
    }
    if (typeof renderOverviewRecentActivity === 'function') {
      renderOverviewRecentActivity();
    }
  }
}

function updateTelemetryMetrics() {
  const history = getSessionHistory();
  const count = history.length;

  // Update nav history badge on all pages
  const navBadge = document.getElementById('navHistoryCount');
  if (navBadge) navBadge.textContent = count;

  // If on Overview page, update KPI metrics
  const kpiTotal = document.getElementById('kpiTotalScans');
  const kpiHealthy = document.getElementById('kpiHealthyCount');
  const kpiHealthyPct = document.getElementById('kpiHealthyPercent');
  const kpiDiseased = document.getElementById('kpiDiseasedCount');
  const kpiDiseasedPct = document.getElementById('kpiDiseasedPercent');
  const kpiAvgConf = document.getElementById('kpiAvgConfidence');

  if (kpiTotal) kpiTotal.textContent = count;

  if (count === 0) {
    if (kpiHealthy) kpiHealthy.textContent = '0';
    if (kpiHealthyPct) kpiHealthyPct.textContent = '0.0% of total scanned';
    if (kpiDiseased) kpiDiseased.textContent = '0';
    if (kpiDiseasedPct) kpiDiseasedPct.textContent = '0.0% required IPM attention';
    if (kpiAvgConf) kpiAvgConf.textContent = '--';
  } else {
    const healthyCount = history.filter(h => h.is_healthy).length;
    const diseasedCount = count - healthyCount;
    const healthyRatio = ((healthyCount / count) * 100).toFixed(1);
    const diseasedRatio = ((diseasedCount / count) * 100).toFixed(1);

    const totalConf = history.reduce((acc, h) => acc + (h.confidence || 0), 0);
    const avgConf = ((totalConf / count) * 100).toFixed(1) + '%';

    if (kpiHealthy) kpiHealthy.textContent = healthyCount;
    if (kpiHealthyPct) kpiHealthyPct.textContent = `${healthyRatio}% of total scanned`;
    if (kpiDiseased) kpiDiseased.textContent = diseasedCount;
    if (kpiDiseasedPct) kpiDiseasedPct.textContent = `${diseasedRatio}% required IPM attention`;
    if (kpiAvgConf) kpiAvgConf.textContent = avgConf;
  }
}

function inspectHistoricalRecord(id) {
  // Navigate to scan page with inspection ID query param
  window.location.href = `/app/scan?inspect=${id}`;
}

// Language Switching
function setLanguage(lang) {
  currentLanguage = lang;
  try {
    localStorage.setItem(LANG_STORAGE_KEY, lang);
  } catch (e) {}

  document.querySelectorAll('.lang-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.lang === lang || (b.id && b.id.toLowerCase().endsWith(lang)));
  });

  const s = UI_STRINGS[lang] || UI_STRINGS.en;

  // Nav labels
  if (document.getElementById('navTabHome')) document.getElementById('navTabHome').textContent = s.navTabHome || "Home";
  if (document.getElementById('navTabScanner')) document.getElementById('navTabScanner').textContent = s.navTabScanner;
  if (document.getElementById('navTabOverview')) document.getElementById('navTabOverview').textContent = s.navTabOverview;
  if (document.getElementById('navTabHistory')) document.getElementById('navTabHistory').textContent = s.navTabHistory;
  if (document.getElementById('navTabInsights')) document.getElementById('navTabInsights').textContent = s.navTabInsights;
  if (document.getElementById('navTabSettings')) document.getElementById('navTabSettings').textContent = s.navTabSettings;

  // Home page elements
  if (document.getElementById('heroMainTitle')) document.getElementById('heroMainTitle').innerHTML = s.heroTag;
  if (document.getElementById('heroSubtitle')) document.getElementById('heroSubtitle').textContent = s.heroSub;

  // Scan page elements
  if (document.getElementById('uploadTitle')) document.getElementById('uploadTitle').innerHTML = s.uploadTitle;
  if (document.getElementById('cropSelectLabel')) document.getElementById('cropSelectLabel').textContent = s.cropSelect;
  if (document.getElementById('partSelectLabel')) document.getElementById('partSelectLabel').textContent = s.partSelect;
  if (document.getElementById('dropzoneText')) document.getElementById('dropzoneText').textContent = s.dropText;
  if (document.getElementById('dropzoneSub')) document.getElementById('dropzoneSub').textContent = s.dropSub;
  if (document.getElementById('samplesHeading')) document.getElementById('samplesHeading').textContent = s.samples;
  if (document.getElementById('btnAnalyzeText')) document.getElementById('btnAnalyzeText').textContent = s.analyzeBtn;
  if (document.getElementById('awaitingTitle')) document.getElementById('awaitingTitle').textContent = s.awaitingTitle;
  if (document.getElementById('awaitingSub')) document.getElementById('awaitingSub').textContent = s.awaitingSub;
  if (document.getElementById('tabOrganicTitle')) document.getElementById('tabOrganicTitle').textContent = s.tabOrg;
  if (document.getElementById('tabChemicalTitle')) document.getElementById('tabChemicalTitle').textContent = s.tabChem;
  if (document.getElementById('tabPreventionTitle')) document.getElementById('tabPreventionTitle').textContent = s.tabPrev;
  if (document.getElementById('gaugeLabel')) document.getElementById('gaugeLabel').textContent = s.gaugeLabel;
  if (document.getElementById('labelConfidence')) document.getElementById('labelConfidence').textContent = s.lblConfidence;
  if (document.getElementById('labelPlantPart')) document.getElementById('labelPlantPart').textContent = s.lblPart;
  if (document.getElementById('labelLatency')) document.getElementById('labelLatency').textContent = s.lblLatency;
  if (document.getElementById('voiceBtnText')) document.getElementById('voiceBtnText').textContent = s.voiceBtn;
  if (document.getElementById('voiceStopText')) document.getElementById('voiceStopText').textContent = s.voiceStop;

  if (document.getElementById('lesionTitle')) {
    document.getElementById('lesionTitle').innerHTML = `<span>🔬</span> ${s.lesionTitle}`;
    if (document.getElementById('lblOriginalImg')) document.getElementById('lblOriginalImg').textContent = s.originalImg;
    if (document.getElementById('lblHighlightImg')) document.getElementById('lblHighlightImg').textContent = window.showingBinaryMask ? s.maskImg : s.highlightImg;
    if (document.getElementById('lblStatCount')) document.getElementById('lblStatCount').textContent = s.lblStatCount;
    if (document.getElementById('lblStatArea')) document.getElementById('lblStatArea').textContent = s.lblStatArea;
    if (document.getElementById('lblStatPixels')) document.getElementById('lblStatPixels').textContent = s.lblStatPixels;
    if (document.getElementById('lblStatLeaf')) document.getElementById('lblStatLeaf').textContent = s.lblStatLeaf;
    if (document.getElementById('btnToggleMaskText')) document.getElementById('btnToggleMaskText').textContent = window.showingBinaryMask ? s.btnOverlayView : s.btnMaskView;
  }

  // Trigger page-specific re-renders if available
  if (typeof renderSelectors === 'function') {
    renderSelectors();
  }
  if (typeof stopSpeech === 'function') {
    stopSpeech();
  }
  if (typeof renderResults === 'function' && window.currentResult) {
    renderResults(window.currentResult);
  }
}

// Floating Dock Magnification Animation
function initFloatingDock() {
  const dock = document.getElementById('floatingDock');
  if (!dock) return;
  const items = dock.querySelectorAll('.dock-item');

  dock.addEventListener('mousemove', (e) => {
    const mouseX = e.clientX;
    items.forEach(item => {
      const rect = item.getBoundingClientRect();
      const itemCenterX = rect.left + rect.width / 2;
      const dist = Math.abs(mouseX - itemCenterX);
      const maxDist = 120;
      if (dist < maxDist) {
        const scale = 1 + Math.pow(1 - dist / maxDist, 2) * 0.35;
        item.style.transform = `scale(${scale}) translateY(-4px)`;
      } else {
        item.style.transform = 'scale(1) translateY(0)';
      }
    });
  });

  dock.addEventListener('mouseleave', () => {
    items.forEach(item => {
      item.style.transform = 'scale(1) translateY(0)';
    });
  });
}

// Active Navigation State Initializer
function initActiveNav(pageKey) {
  // Update nav strip buttons
  document.querySelectorAll('.dash-nav-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.page === pageKey);
  });
  // Update dock items
  document.querySelectorAll('.dock-item').forEach(item => {
    item.classList.toggle('active', item.dataset.page === pageKey);
  });
}

// Health Check Ping Helper
async function pingBackendHealth() {
  const box = document.getElementById('pingResultBox');
  if (!box) return;
  box.style.display = 'block';
  box.textContent = 'Pinging /api/v1/health...';
  const start = performance.now();
  try {
    const res = await fetch('/api/v1/health');
    const elapsed = (performance.now() - start).toFixed(1);
    if (res.ok) {
      const json = await res.json();
      box.style.color = '#86efac';
      box.textContent = `✓ OK (${elapsed}ms) — Status: ${json.status || 'operational'}, Version: ${json.version || '1.0'}`;
    } else {
      box.style.color = '#fca5a5';
      box.textContent = `✗ HTTP Error ${res.status} (${elapsed}ms)`;
    }
  } catch (err) {
    box.style.color = '#fca5a5';
    box.textContent = `✗ Connection Failed: ${err.message}`;
  }
}

// HTML Escaping Helper
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Common DOMContentLoaded Initialization
document.addEventListener('DOMContentLoaded', () => {
  initFloatingDock();
  updateTelemetryMetrics();
  setLanguage(currentLanguage);

  // Initialize Speech Synthesis voice loading
  if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      window.speechSynthesis.getVoices();
    };
  }
});
