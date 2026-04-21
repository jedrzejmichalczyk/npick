import { parseFile } from './touchstone.js';

let Module = null;
let solver = null;
let muxSolver = null;
let loadData = null;

// Default S1P data: patch antenna with resonance at 2.45 GHz
const DEFAULT_S1P = `! Demo: patch antenna with resonance at 2.45 GHz
! Mismatched outside the resonance band - good candidate for matching
# GHz S RI R 50
1.000  -0.850   0.200
1.100  -0.830   0.250
1.200  -0.800   0.300
1.300  -0.760   0.350
1.400  -0.710   0.390
1.500  -0.650   0.420
1.600  -0.580   0.440
1.700  -0.500   0.450
1.800  -0.410   0.440
1.900  -0.320   0.410
2.000  -0.230   0.360
2.050  -0.185   0.330
2.100  -0.140   0.290
2.150  -0.100   0.245
2.200  -0.065   0.195
2.250  -0.038   0.140
2.300  -0.020   0.085
2.350  -0.012   0.030
2.400  -0.010  -0.025
2.420  -0.012  -0.048
2.440  -0.018  -0.070
2.450  -0.022  -0.080
2.460  -0.028  -0.090
2.480  -0.040  -0.108
2.500  -0.055  -0.125
2.550  -0.095  -0.170
2.600  -0.145  -0.220
2.650  -0.200  -0.270
2.700  -0.260  -0.320
2.800  -0.380  -0.400
2.900  -0.490  -0.440
3.000  -0.580  -0.450
3.100  -0.650  -0.430
3.200  -0.710  -0.400
3.300  -0.750  -0.360
3.400  -0.785  -0.320
3.500  -0.810  -0.280
3.600  -0.830  -0.240
3.700  -0.845  -0.200
3.800  -0.855  -0.165
3.900  -0.862  -0.135
4.000  -0.868  -0.108
`;

// ---- DOM elements ----
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const fileInfo    = document.getElementById('fileInfo');
const paramGroup  = document.getElementById('paramSelectGroup');
const paramSelect = document.getElementById('paramSelect');
const freqLeftEl  = document.getElementById('freqLeft');
const freqRightEl = document.getElementById('freqRight');
const orderEl     = document.getElementById('order');
const rlEl        = document.getElementById('returnLoss');
const tzEl        = document.getElementById('tzInput');
const runBtn      = document.getElementById('runBtn');
const statusEl    = document.getElementById('status');
const resultsPanel = document.getElementById('resultsPanel');
const resultBadge  = document.getElementById('resultBadge');
const resultDetails = document.getElementById('resultDetails');
const cmPanel      = document.getElementById('cmPanel');
const cmDiv        = document.getElementById('couplingMatrix');

// ---- Initialize WASM ----
async function initWasm() {
    try {
        Module = await createNpickModule();
        solver = new Module.SolverWrapper();
        muxSolver = new Module.MultiplexerWrapper();
        statusEl.className = '';
        if (muxStatusEl) muxStatusEl.textContent = '';
        if (loadData && loadData.freqs.length >= 3) {
            solver.set_load_data(loadData.freqs, loadData.re, loadData.im);
        }
        updateRunButton();
        refreshMuxRunBtn();
    } catch (e) {
        statusEl.textContent = 'Failed to load WASM module: ' + e.message;
        statusEl.className = 'error';
    }
}

function loadDefaultData() {
    currentFilename = 'demo.s1p';
    currentFileText = DEFAULT_S1P;
    loadFileData();
}

const loadDemoLink = document.getElementById('loadDemo');
loadDemoLink.addEventListener('click', (e) => {
    e.preventDefault();
    loadDefaultData();
});

initWasm();

// ---- File handling ----
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

let currentFilename = '';
let currentFileText = '';

function handleFile(file) {
    currentFilename = file.name;
    const reader = new FileReader();
    reader.onload = (e) => {
        currentFileText = e.target.result;
        loadFileData();
    };
    reader.readAsText(file);
}

function loadFileData(portParam) {
    portParam = portParam || paramSelect.value;
    loadData = parseFile(currentFileText, currentFilename, portParam);

    if (loadData.freqs.length < 3) {
        statusEl.textContent = 'Error: file has fewer than 3 data points.';
        statusEl.className = 'error';
        return;
    }

    fileInfo.textContent = `${currentFilename}: ${loadData.freqs.length} points, ` +
        `${fmtFreq(loadData.freqs[0])} - ${fmtFreq(loadData.freqs[loadData.freqs.length - 1])}`;
    fileInfo.classList.add('visible');

    if (loadData.numPorts >= 2) {
        paramGroup.style.display = '';
    } else {
        paramGroup.style.display = 'none';
    }

    // Auto-fill band edges: find the best-matched region
    const fMin = loadData.freqs[0];
    const fMax = loadData.freqs[loadData.freqs.length - 1];
    const fSpan = fMax - fMin;

    // Find frequency of minimum |Gamma| (resonance)
    let minMag = Infinity, minIdx = 0;
    for (let i = 0; i < loadData.freqs.length; i++) {
        const mag = Math.sqrt(loadData.re[i] ** 2 + loadData.im[i] ** 2);
        if (mag < minMag) { minMag = mag; minIdx = i; }
    }

    // Set band around the resonance: expand until |Gamma| exceeds a threshold
    const threshold = Math.min(0.5, minMag + 0.3);
    let iLeft = minIdx, iRight = minIdx;
    while (iLeft > 0) {
        const mag = Math.sqrt(loadData.re[iLeft - 1] ** 2 + loadData.im[iLeft - 1] ** 2);
        if (mag > threshold) break;
        iLeft--;
    }
    while (iRight < loadData.freqs.length - 1) {
        const mag = Math.sqrt(loadData.re[iRight + 1] ** 2 + loadData.im[iRight + 1] ** 2);
        if (mag > threshold) break;
        iRight++;
    }

    // Ensure minimum bandwidth (at least 10% of data range)
    const autoLeft = loadData.freqs[iLeft];
    const autoRight = loadData.freqs[iRight];
    if (autoRight - autoLeft < 0.1 * fSpan) {
        // Fallback: middle 40%
        freqLeftEl.value = +(fMin + 0.3 * fSpan).toPrecision(6);
        freqRightEl.value = +(fMax - 0.3 * fSpan).toPrecision(6);
    } else {
        freqLeftEl.value = +autoLeft.toPrecision(6);
        freqRightEl.value = +autoRight.toPrecision(6);
    }

    if (solver) {
        solver.set_load_data(loadData.freqs, loadData.re, loadData.im);
    }

    updateRunButton();
    plotLoad();
    statusEl.textContent = 'Load data imported. Adjust band and click Run.';
    statusEl.className = '';
}

paramSelect.addEventListener('change', () => {
    if (currentFileText) loadFileData(paramSelect.value);
});

// ---- Plotting ----
function plotLoad() {
    if (!loadData) return;

    const mag_db = loadData.freqs.map((f, i) => {
        const mag = Math.sqrt(loadData.re[i] ** 2 + loadData.im[i] ** 2);
        return mag > 1e-15 ? 20 * Math.log10(mag) : -300;
    });

    const trace = {
        x: loadData.freqs,
        y: mag_db,
        mode: 'lines',
        name: '|Load| (dB)',
        line: { color: '#2196F3', width: 2 }
    };

    const fl = parseFloat(freqLeftEl.value);
    const fr = parseFloat(freqRightEl.value);
    const shapes = (isFinite(fl) && isFinite(fr)) ? [{
        type: 'rect', xref: 'x', yref: 'paper',
        x0: fl, x1: fr, y0: 0, y1: 1,
        fillcolor: 'rgba(255,235,59,0.12)',
        line: { width: 0 }
    }] : [];

    Plotly.newPlot('plot', [trace], {
        xaxis: { title: 'Frequency' + (loadData.freqUnit ? ` (${loadData.freqUnit})` : '') },
        yaxis: { title: 'dB', range: [-50, 5] },
        margin: { t: 10, r: 20 },
        shapes
    }, { responsive: true });
}

function plotResults(resp) {
    if (!loadData) return;

    const fl = parseFloat(freqLeftEl.value);
    const fr = parseFloat(freqRightEl.value);

    const traces = [
        {
            x: loadData.freqs,
            y: loadData.freqs.map((f, i) => {
                const mag = Math.sqrt(loadData.re[i] ** 2 + loadData.im[i] ** 2);
                return mag > 1e-15 ? 20 * Math.log10(mag) : -300;
            }),
            mode: 'lines', name: '|Load|',
            line: { color: '#90CAF9', width: 1.5, dash: 'dash' }
        },
        {
            x: resp.freq, y: resp.g11_db,
            mode: 'lines', name: '|G11| (matched)',
            line: { color: '#E53935', width: 2.5 }
        },
        {
            x: resp.freq, y: resp.s11_db,
            mode: 'lines', name: '|S11| (filter)',
            line: { color: '#43A047', width: 1.5 }
        },
        {
            x: resp.freq, y: resp.s21_db,
            mode: 'lines', name: '|S21|',
            line: { color: '#00ACC1', width: 1, dash: 'dot' }
        }
    ];

    Plotly.newPlot('plot', traces, {
        xaxis: { title: 'Frequency' + (loadData.freqUnit ? ` (${loadData.freqUnit})` : '') },
        yaxis: { title: 'dB', range: [-50, 5] },
        margin: { t: 10, r: 20 },
        shapes: [{
            type: 'rect', xref: 'x', yref: 'paper',
            x0: fl, x1: fr, y0: 0, y1: 1,
            fillcolor: 'rgba(255,235,59,0.12)',
            line: { width: 0 }
        }],
        legend: { x: 0.01, y: 0.01, bgcolor: 'rgba(255,255,255,0.8)' }
    }, { responsive: true });
}

// ---- Solver ----
function updateRunButton() {
    runBtn.disabled = !(solver && loadData && loadData.freqs.length >= 3);
}

runBtn.addEventListener('click', runSolver);

function runSolver() {
    if (!solver || !loadData) return;

    const fl = parseFloat(freqLeftEl.value);
    const fr = parseFloat(freqRightEl.value);
    const order = parseInt(orderEl.value);
    const rl = parseFloat(rlEl.value);

    if (!isFinite(fl) || !isFinite(fr) || fl >= fr) {
        statusEl.textContent = 'Error: invalid frequency band.';
        statusEl.className = 'error';
        return;
    }
    if (order < 2 || order > 12) {
        statusEl.textContent = 'Error: order must be 2-12.';
        statusEl.className = 'error';
        return;
    }

    // Parse transmission zeros
    const tzReArr = [], tzImArr = [];
    const tzText = tzEl.value.trim();
    if (tzText) {
        for (const tok of tzText.split(',')) {
            const v = parseFloat(tok.trim());
            if (isFinite(v)) {
                tzReArr.push(v);
                tzImArr.push(0);
            }
        }
        if (tzReArr.length > order) {
            statusEl.textContent = 'Error: too many transmission zeros for this order.';
            statusEl.className = 'error';
            return;
        }
    }

    runBtn.disabled = true;
    statusEl.textContent = 'Solving...';
    statusEl.className = 'running';

    // Run async to let UI update
    setTimeout(() => {
        try {
            const success = solver.solve(fl, fr, order, rl, tzReArr, tzImArr);

            resultsPanel.style.display = '';

            if (success) {
                const rlDb = solver.get_achieved_rl_db();
                resultBadge.innerHTML = `<span class="result-badge success">${rlDb.toFixed(1)} dB return loss</span>`;
                resultDetails.innerHTML =
                    `<div style="font-size:0.78rem;margin-top:0.3rem;color:#555">` +
                    `Order ${order} filter` +
                    (tzReArr.length ? `, ${tzReArr.length} transmission zero${tzReArr.length > 1 ? 's' : ''}` : ', all-pole') +
                    `</div>`;

                displayCouplingMatrix(order);

                // Evaluate and plot response
                const fSpan = fr - fl;
                const resp = solver.evaluate_response(fl - 0.5 * fSpan, fr + 0.5 * fSpan, 501);

                // Convert val arrays to JS arrays
                const plotData = {
                    freq: jsArray(resp.freq),
                    load_db: jsArray(resp.load_db),
                    g11_db: jsArray(resp.g11_db),
                    s11_db: jsArray(resp.s11_db),
                    s21_db: jsArray(resp.s21_db),
                };
                plotResults(plotData);

                statusEl.textContent = `Done: ${rlDb.toFixed(1)} dB return loss achieved.`;
                statusEl.className = '';
            } else {
                const errMsg = solver.get_error_message();
                resultBadge.innerHTML = `<span class="result-badge error">Failed</span>`;
                resultDetails.innerHTML = `<div style="font-size:0.78rem">${errMsg}</div>`;
                cmPanel.style.display = 'none';
                statusEl.textContent = 'Solver failed: ' + errMsg;
                statusEl.className = 'error';
            }
        } catch (e) {
            resultsPanel.style.display = '';
            resultBadge.innerHTML = `<span class="result-badge error">Error</span>`;
            resultDetails.innerHTML = `<div style="font-size:0.78rem">${e.message || e}</div>`;
            cmPanel.style.display = 'none';
            statusEl.textContent = 'Error: ' + (e.message || e);
            statusEl.className = 'error';
        }

        runBtn.disabled = false;
    }, 50);
}

function displayCouplingMatrix(order) {
    const n = solver.get_cm_size();
    if (n === 0) { cmPanel.style.display = 'none'; return; }

    cmPanel.style.display = '';

    // Row/column labels: S, 1, 2, ..., n, L
    const labels = ['S'];
    for (let i = 1; i <= n - 2; i++) labels.push(String(i));
    labels.push('L');

    let html = '<table><tr><th></th>';
    for (const lbl of labels) html += `<th>${lbl}</th>`;
    html += '</tr>';

    for (let i = 0; i < n; i++) {
        html += `<tr><td class="row-header">${labels[i]}</td>`;
        for (let j = 0; j < n; j++) {
            const re = solver.get_cm_real(i, j);
            const im = solver.get_cm_imag(i, j);
            const mag = Math.sqrt(re * re + im * im);

            let cls = '';
            if (i === j) cls = 'diagonal';
            else if (mag < 1e-10) cls = 'zero';
            else cls = 'coupling';

            let text;
            if (mag < 1e-10) {
                text = '0';
            } else if (Math.abs(im) < 1e-10) {
                text = fmtNum(re);
            } else {
                text = fmtNum(re) + (im >= 0 ? '+' : '') + fmtNum(im) + 'j';
            }

            html += `<td class="${cls}">${text}</td>`;
        }
        html += '</tr>';
    }
    html += '</table>';
    cmDiv.innerHTML = html;
}

function fmtNum(v) {
    if (Math.abs(v) < 1e-10) return '0';
    if (Math.abs(v) >= 100) return v.toFixed(2);
    if (Math.abs(v) >= 10) return v.toFixed(3);
    return v.toFixed(4);
}

// Convert an emscripten val array to a plain JS array
function jsArray(v) {
    const len = v.length;
    const arr = new Array(len);
    for (let i = 0; i < len; i++) arr[i] = v[i];
    return arr;
}

// ---- Helpers ----
function fmtFreq(f) {
    if (f >= 1e9) return (f / 1e9).toPrecision(4) + ' GHz';
    if (f >= 1e6) return (f / 1e6).toPrecision(4) + ' MHz';
    if (f >= 1e3) return (f / 1e3).toPrecision(4) + ' kHz';
    return f.toPrecision(4) + ' Hz';
}

freqLeftEl.addEventListener('change', plotLoad);
freqRightEl.addEventListener('change', plotLoad);

// Load default data immediately so the plot is visible on page load
loadDefaultData();

// ============================================================================
// Multiplexer synthesis mode
// ============================================================================

const CHANNEL_COLORS = ['#E53935', '#43A047', '#1E88E5', '#F57C00', '#8E24AA', '#00ACC1'];

// DOM
const impedanceModeEl = document.getElementById('impedanceMode');
const multiplexerModeEl = document.getElementById('multiplexerMode');
const modeTabs = document.querySelectorAll('.mode-tab');
const channelListEl = document.getElementById('channelList');
const addChannelBtn = document.getElementById('addChannelBtn');
const junctionModelEl = document.getElementById('junctionModel');
const junctionAlphaGroupEl = document.getElementById('junctionAlphaGroup');
const junctionFileGroupEl = document.getElementById('junctionFileGroup');
const junctionDropZone = document.getElementById('junctionDropZone');
const junctionFileInput = document.getElementById('junctionFileInput');
const junctionFileInfoEl = document.getElementById('junctionFileInfo');
const muxCenterFreqEl = document.getElementById('muxCenterFreq');
const muxEquiItersEl = document.getElementById('muxEquiIters');
const muxRunBtn = document.getElementById('muxRunBtn');
const muxStatusEl = document.getElementById('muxStatus');
const muxResultsPanel = document.getElementById('muxResultsPanel');
const muxResultSummary = document.getElementById('muxResultSummary');
const cmChannelTabsEl = document.getElementById('cmChannelTabs');
const plotTitleEl = document.getElementById('plotTitle');
const cmTitleEl = document.getElementById('cmTitle');

// Current multiplexer state
let muxChannels = [];              // [{order, fl, fr, rl, tz (string)}, ...]
let muxJunctionSMatrices = {};     // {junction_index: {re: [9], im: [9]}} — populated from .s3p uploads
let muxResult = null;              // {cms: [...], rls: [...], resp: {freq, g11_db[], s21_db[]}}
let activeCmChannel = 0;

// ---- Mode switching ----
modeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        modeTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const mode = tab.dataset.mode;
        if (mode === 'impedance') {
            impedanceModeEl.style.display = '';
            multiplexerModeEl.style.display = 'none';
            plotTitleEl.textContent = 'Frequency Response';
            if (loadData) plotLoad();
            // Restore single-channel coupling matrix if available
            if (solver && solver.get_cm_size && solver.get_cm_size() > 0) {
                document.getElementById('cmPanel').style.display = '';
                cmChannelTabsEl.style.display = 'none';
                displayCouplingMatrix();
            } else {
                document.getElementById('cmPanel').style.display = 'none';
            }
        } else {
            impedanceModeEl.style.display = 'none';
            multiplexerModeEl.style.display = '';
            plotTitleEl.textContent = 'Multiplexer Response';
            if (muxResult) {
                plotMultiplexerResponse(muxResult);
                document.getElementById('cmPanel').style.display = '';
                cmChannelTabsEl.style.display = '';
                displayMuxCouplingMatrix(activeCmChannel);
            } else {
                document.getElementById('cmPanel').style.display = 'none';
                // Draw empty plot
                Plotly.newPlot('plot', [], {
                    xaxis: { title: 'Frequency (GHz)' },
                    yaxis: { title: 'dB' },
                    margin: { t: 10, r: 20 }
                }, { responsive: true });
            }
        }
    });
});

// ---- Channel list management ----
function renderChannelList() {
    let html = '<div class="channel-row header">' +
        '<div>#</div><div>Order</div><div>F<sub>left</sub> (GHz)</div>' +
        '<div>F<sub>right</sub> (GHz)</div><div>RL (dB)</div><div></div></div>';
    muxChannels.forEach((ch, i) => {
        html += `<div class="channel-row" data-ch="${i}">
            <div class="ch-label">${i + 1}</div>
            <div class="form-group"><input type="number" class="ch-order" min="2" max="12" step="1" value="${ch.order}"></div>
            <div class="form-group"><input type="number" class="ch-fl" step="0.01" value="${ch.fl}"></div>
            <div class="form-group"><input type="number" class="ch-fr" step="0.01" value="${ch.fr}"></div>
            <div class="form-group"><input type="number" class="ch-rl" step="0.5" value="${ch.rl}"></div>
            <button class="ch-remove" title="Remove">×</button>
        </div>`;
    });
    channelListEl.innerHTML = html;

    // Bind events
    channelListEl.querySelectorAll('.channel-row[data-ch]').forEach(row => {
        const idx = parseInt(row.dataset.ch);
        row.querySelector('.ch-order').addEventListener('change', e => {
            muxChannels[idx].order = Math.max(2, Math.min(12, parseInt(e.target.value) || 4));
        });
        row.querySelector('.ch-fl').addEventListener('change', e => {
            muxChannels[idx].fl = parseFloat(e.target.value);
        });
        row.querySelector('.ch-fr').addEventListener('change', e => {
            muxChannels[idx].fr = parseFloat(e.target.value);
        });
        row.querySelector('.ch-rl').addEventListener('change', e => {
            muxChannels[idx].rl = parseFloat(e.target.value);
        });
        row.querySelector('.ch-remove').addEventListener('click', () => {
            if (muxChannels.length > 2) {
                muxChannels.splice(idx, 1);
                renderChannelList();
                renderJunctionAlphaInputs();
            } else {
                muxStatusEl.textContent = 'Minimum 2 channels required.';
                muxStatusEl.className = 'error';
            }
        });
    });
}

addChannelBtn.addEventListener('click', () => {
    if (muxChannels.length >= 6) {
        muxStatusEl.textContent = 'Maximum 6 channels supported in the demo.';
        muxStatusEl.className = 'error';
        return;
    }
    // Default new channel: follow pattern from previous channel
    const last = muxChannels[muxChannels.length - 1];
    const bw = last ? (last.fr - last.fl) : 0.04;
    const guard = 0.02;
    const newFl = last ? last.fr + guard : 2.0;
    muxChannels.push({
        order: last ? last.order : 4,
        fl: +newFl.toFixed(3),
        fr: +(newFl + bw).toFixed(3),
        rl: last ? last.rl : 23
    });
    renderChannelList();
    renderJunctionAlphaInputs();
});

// ---- Junction model controls ----
function renderJunctionAlphaInputs() {
    const model = junctionModelEl.value;
    junctionFileGroupEl.style.display = model === 's3p' ? '' : 'none';
    if (model !== 'custom') {
        junctionAlphaGroupEl.style.display = 'none';
        junctionAlphaGroupEl.innerHTML = '';
        return;
    }
    const N = muxChannels.length;
    let html = '';
    for (let j = 0; j < N - 1; j++) {
        const existing = junctionAlphaGroupEl.querySelector(`.alpha-row[data-j="${j}"] input`);
        const v = existing ? existing.value : (1.0 / (N - j)).toFixed(3);
        html += `<div class="alpha-row" data-j="${j}">
            <label>T<sub>${j + 1}</sub> α</label>
            <input type="number" min="0.01" max="0.99" step="0.01" value="${v}">
        </div>`;
    }
    junctionAlphaGroupEl.innerHTML = html;
    junctionAlphaGroupEl.style.display = '';
}

junctionModelEl.addEventListener('change', renderJunctionAlphaInputs);

// ---- Junction S-parameter file upload ----
junctionDropZone.addEventListener('click', () => junctionFileInput.click());
junctionFileInput.addEventListener('change', (e) => handleJunctionFiles(e.target.files));
junctionDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    junctionDropZone.classList.add('dragover');
});
junctionDropZone.addEventListener('dragleave', () => junctionDropZone.classList.remove('dragover'));
junctionDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    junctionDropZone.classList.remove('dragover');
    handleJunctionFiles(e.dataTransfer.files);
});

function handleJunctionFiles(files) {
    muxJunctionSMatrices = {};
    if (!files || files.length === 0) return;
    const N = muxChannels.length;
    let loaded = 0;
    let infoParts = [];
    Array.from(files).forEach((file, k) => {
        if (k >= N - 1) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            const parsed = parseS3P(e.target.result);
            if (parsed) {
                // Evaluate at center frequency (use first row as simple default)
                const fc = parseFloat(muxCenterFreqEl.value);
                const S = interpolateS3P(parsed, fc);
                muxJunctionSMatrices[k] = S;
                infoParts[k] = `T${k + 1}: ${file.name}`;
                loaded++;
                if (loaded === Math.min(files.length, N - 1)) {
                    junctionFileInfoEl.textContent = infoParts.filter(x => x).join(', ');
                    junctionFileInfoEl.classList.add('visible');
                }
            }
        };
        reader.readAsText(file);
    });
}

// Minimal .s3p parser (Touchstone 3-port, magnitude/phase or real/imag).
function parseS3P(text) {
    const lines = text.split('\n');
    let fmt = 'MA', unit = 'GHz', z0 = 50;
    const rows = [];
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('!')) continue;
        if (trimmed.startsWith('#')) {
            const parts = trimmed.substring(1).trim().split(/\s+/);
            for (let i = 0; i < parts.length; i++) {
                const p = parts[i].toUpperCase();
                if (['HZ', 'KHZ', 'MHZ', 'GHZ'].includes(p)) unit = p;
                else if (['MA', 'DB', 'RI'].includes(p)) fmt = p;
                else if (p === 'R' && i + 1 < parts.length) z0 = parseFloat(parts[i + 1]);
            }
            continue;
        }
        const tokens = trimmed.split(/\s+/).map(parseFloat).filter(v => isFinite(v));
        if (tokens.length >= 1 + 2 * 9) {
            rows.push(tokens);
        }
    }
    if (rows.length === 0) return null;
    const scale = { HZ: 1e-9, KHZ: 1e-6, MHZ: 1e-3, GHZ: 1 }[unit] || 1;
    return { rows, fmt, scale };
}

function interpolateS3P(parsed, freqGhz) {
    const { rows, fmt, scale } = parsed;
    // Find nearest freq row
    let nearest = rows[0], minDist = Infinity;
    for (const r of rows) {
        const fGHz = r[0] * scale;
        const d = Math.abs(fGHz - freqGhz);
        if (d < minDist) { minDist = d; nearest = r; }
    }
    // Row format: freq, then 9 complex pairs (S11 S12 S13 S21 S22 S23 S31 S32 S33) in MA/DB/RI
    const re = [], im = [];
    for (let i = 0; i < 9; i++) {
        const a = nearest[1 + 2 * i];
        const b = nearest[2 + 2 * i];
        let cr, ci;
        if (fmt === 'RI') { cr = a; ci = b; }
        else if (fmt === 'DB') {
            const mag = Math.pow(10, a / 20);
            const ph = b * Math.PI / 180;
            cr = mag * Math.cos(ph); ci = mag * Math.sin(ph);
        } else { // MA
            const ph = b * Math.PI / 180;
            cr = a * Math.cos(ph); ci = a * Math.sin(ph);
        }
        re.push(cr); im.push(ci);
    }
    return { re, im };
}

// ---- Presets ----
document.querySelectorAll('[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => loadPreset(btn.dataset.preset));
});

function loadPreset(name) {
    muxJunctionSMatrices = {};
    junctionFileInfoEl.textContent = '';
    junctionFileInfoEl.classList.remove('visible');

    if (name === 'duplexer') {
        muxChannels = [
            { order: 4, fl: 1.88, fr: 1.92, rl: 23 },
            { order: 4, fl: 2.00, fr: 2.04, rl: 23 }
        ];
        muxCenterFreqEl.value = 1.96;
    } else if (name === 'triplexer') {
        muxChannels = [
            { order: 4, fl: 1.8, fr: 2.0, rl: 23 },
            { order: 4, fl: 2.1, fr: 2.3, rl: 23 },
            { order: 4, fl: 2.4, fr: 2.6, rl: 23 }
        ];
        muxCenterFreqEl.value = 2.2;
    }
    junctionModelEl.value = 'equal';
    renderChannelList();
    renderJunctionAlphaInputs();
    muxStatusEl.textContent = `${name[0].toUpperCase()}${name.slice(1)} preset loaded. Click Synthesize.`;
    muxStatusEl.className = '';
}

// Load duplexer preset on first open
loadPreset('duplexer');

// ---- Run multiplexer solver ----
function refreshMuxRunBtn() {
    muxRunBtn.disabled = !muxSolver || muxChannels.length < 2;
}

muxRunBtn.addEventListener('click', runMuxSolver);

function runMuxSolver() {
    if (!muxSolver) return;
    refreshMuxRunBtn();
    // Validate channels
    for (let i = 0; i < muxChannels.length; i++) {
        const c = muxChannels[i];
        if (!isFinite(c.fl) || !isFinite(c.fr) || c.fl >= c.fr) {
            muxStatusEl.textContent = `Channel ${i + 1}: invalid frequency band.`;
            muxStatusEl.className = 'error';
            return;
        }
        if (c.order < 2 || c.order > 12) {
            muxStatusEl.textContent = `Channel ${i + 1}: order must be 2-12.`;
            muxStatusEl.className = 'error';
            return;
        }
    }

    muxRunBtn.disabled = true;
    muxStatusEl.textContent = 'Synthesizing multiplexer…';
    muxStatusEl.className = 'running';

    setTimeout(() => {
        try {
            muxSolver.reset_channels();
            for (const c of muxChannels) {
                muxSolver.add_channel(c.order, c.fl, c.fr, c.rl, [], []);
            }
            muxSolver.set_center_frequency(parseFloat(muxCenterFreqEl.value));
            muxSolver.set_equiripple_iters(parseInt(muxEquiItersEl.value) || 0);

            const N = muxChannels.length;
            const model = junctionModelEl.value;
            for (let j = 0; j < N - 1; j++) {
                let alpha = 0.5;
                if (model === 'equal_power') alpha = 1.0 / (N - j);
                else if (model === 'custom') {
                    const row = junctionAlphaGroupEl.querySelector(`.alpha-row[data-j="${j}"] input`);
                    alpha = row ? parseFloat(row.value) : 0.5;
                }
                muxSolver.set_junction_alpha(j, alpha);

                if (model === 's3p' && muxJunctionSMatrices[j]) {
                    muxSolver.set_junction_s(j, muxJunctionSMatrices[j].re, muxJunctionSMatrices[j].im);
                }
            }

            const t0 = performance.now();
            const ok = muxSolver.solve();
            const dt = performance.now() - t0;

            if (!ok) {
                muxResultsPanel.style.display = 'none';
                document.getElementById('cmPanel').style.display = 'none';
                muxStatusEl.textContent = 'Solver failed: ' + muxSolver.get_error_message();
                muxStatusEl.className = 'error';
                muxResult = null;
                return;
            }

            const rls = [];
            for (let i = 0; i < N; i++) rls.push(muxSolver.get_achieved_rl_db(i));

            // Evaluate response over a generous sweep around the channels.
            let fMin = Infinity, fMax = -Infinity;
            for (const c of muxChannels) { fMin = Math.min(fMin, c.fl); fMax = Math.max(fMax, c.fr); }
            const pad = 0.2 * (fMax - fMin);
            const resp = muxSolver.evaluate_response(fMin - pad, fMax + pad, 1001);
            const respJs = {
                freq: jsArray(resp.freq),
                g11_db: [],
                s21_db: []
            };
            for (let i = 0; i < N; i++) {
                respJs.g11_db.push(jsArray(resp.g11_db[i]));
                respJs.s21_db.push(jsArray(resp.s21_db[i]));
            }

            muxResult = { rls, resp: respJs };
            plotMultiplexerResponse(muxResult);

            // Summary
            muxResultsPanel.style.display = '';
            const badges = muxChannels.map((c, i) => {
                const color = CHANNEL_COLORS[i % CHANNEL_COLORS.length];
                return `<span class="ch-badge"><span class="ch-color" style="background:${color}"></span>Ch ${i + 1}: ${rls[i].toFixed(1)} dB</span>`;
            }).join('');
            muxResultSummary.innerHTML =
                `<div><b>Synthesis complete</b> in ${(dt / 1000).toFixed(1)}s.</div>` +
                `<div class="channel-summary">${badges}</div>`;

            // Coupling matrix tabs
            document.getElementById('cmPanel').style.display = '';
            cmChannelTabsEl.style.display = '';
            activeCmChannel = 0;
            renderCmTabs();
            displayMuxCouplingMatrix(0);

            muxStatusEl.textContent = `Done in ${(dt / 1000).toFixed(1)}s.`;
            muxStatusEl.className = '';
        } catch (e) {
            muxStatusEl.textContent = 'Error: ' + (e.message || e);
            muxStatusEl.className = 'error';
        } finally {
            muxRunBtn.disabled = false;
        }
    }, 30);
}

function plotMultiplexerResponse(res) {
    const traces = [];
    const N = res.rls.length;
    for (let i = 0; i < N; i++) {
        const color = CHANNEL_COLORS[i % CHANNEL_COLORS.length];
        traces.push({
            x: res.resp.freq,
            y: res.resp.g11_db[i],
            mode: 'lines',
            name: `|G<sub>11</sub>| Ch ${i + 1}`,
            legendgroup: `ch${i}`,
            line: { color, width: 2 }
        });
        traces.push({
            x: res.resp.freq,
            y: res.resp.s21_db[i],
            mode: 'lines',
            name: `|S<sub>21</sub>| Ch ${i + 1}`,
            legendgroup: `ch${i}`,
            line: { color, width: 1, dash: 'dash' }
        });
    }
    const shapes = muxChannels.map((c, i) => ({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: c.fl, x1: c.fr, y0: 0, y1: 1,
        fillcolor: hexToRgba(CHANNEL_COLORS[i % CHANNEL_COLORS.length], 0.08),
        line: { width: 0 }
    }));
    Plotly.newPlot('plot', traces, {
        xaxis: { title: 'Frequency (GHz)' },
        yaxis: { title: 'dB', range: [-50, 2] },
        margin: { t: 10, r: 20 },
        shapes,
        legend: { orientation: 'h', y: -0.15, yanchor: 'top' }
    }, { responsive: true });
}

function hexToRgba(hex, a) {
    const v = parseInt(hex.slice(1), 16);
    return `rgba(${(v >> 16) & 0xff}, ${(v >> 8) & 0xff}, ${v & 0xff}, ${a})`;
}

function renderCmTabs() {
    const N = muxChannels.length;
    let html = '';
    for (let i = 0; i < N; i++) {
        html += `<button class="cm-tab ${i === activeCmChannel ? 'active' : ''}" data-ch="${i}">Ch ${i + 1}</button>`;
    }
    cmChannelTabsEl.innerHTML = html;
    cmChannelTabsEl.querySelectorAll('.cm-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            activeCmChannel = parseInt(btn.dataset.ch);
            renderCmTabs();
            displayMuxCouplingMatrix(activeCmChannel);
        });
    });
}

function displayMuxCouplingMatrix(ch) {
    const n = muxSolver.get_cm_size(ch);
    cmTitleEl.textContent = `Coupling Matrix — Channel ${ch + 1}`;
    if (n === 0) { cmDiv.innerHTML = '<em>Empty.</em>'; return; }

    const labels = ['S'];
    for (let i = 1; i <= n - 2; i++) labels.push(String(i));
    labels.push('L');

    let html = '<table><tr><th></th>';
    for (const lbl of labels) html += `<th>${lbl}</th>`;
    html += '</tr>';
    for (let i = 0; i < n; i++) {
        html += `<tr><td class="row-header">${labels[i]}</td>`;
        for (let j = 0; j < n; j++) {
            const re = muxSolver.get_cm_real(ch, i, j);
            const im = muxSolver.get_cm_imag(ch, i, j);
            const mag = Math.sqrt(re * re + im * im);
            let cls = i === j ? 'diagonal' : (mag < 1e-10 ? 'zero' : 'coupling');
            let text;
            if (mag < 1e-10) text = '0';
            else if (Math.abs(im) < 1e-10) text = fmtNum(re);
            else text = fmtNum(re) + (im >= 0 ? '+' : '') + fmtNum(im) + 'j';
            html += `<td class="${cls}">${text}</td>`;
        }
        html += '</tr>';
    }
    html += '</table>';
    cmDiv.innerHTML = html;
}
