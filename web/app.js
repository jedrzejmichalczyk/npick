import { parseFile } from './touchstone.js';

let Module = null;
let solver = null;
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
        statusEl.className = '';
        // Push already-loaded default data to the solver
        if (loadData && loadData.freqs.length >= 3) {
            solver.set_load_data(loadData.freqs, loadData.re, loadData.im);
        }
        updateRunButton();
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
