import { parseFile } from './touchstone.js';

let Module = null;
let solver = null;
let loadData = null;  // { freqs, re, im, freqUnit, numPorts }

// ---- DOM elements ----
const dropZone   = document.getElementById('dropZone');
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
const status      = document.getElementById('status');
const resultsPanel = document.getElementById('resultsPanel');
const resultBadge  = document.getElementById('resultBadge');
const resultDetails = document.getElementById('resultDetails');
const cmDiv        = document.getElementById('couplingMatrix');

// ---- Initialize WASM ----
async function initWasm() {
    try {
        Module = await createNpickModule();
        solver = new Module.SolverWrapper();
        status.textContent = 'Ready. Load an S-parameter file to begin.';
        status.className = '';
        updateRunButton();
    } catch (e) {
        status.textContent = 'Failed to load WASM module: ' + e.message;
        status.className = 'error';
    }
}

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
        status.textContent = 'Error: file has fewer than 3 data points.';
        status.className = 'error';
        return;
    }

    // Show file info
    fileInfo.textContent = `${currentFilename}: ${loadData.freqs.length} points, ` +
        `${fmtFreq(loadData.freqs[0])} - ${fmtFreq(loadData.freqs[loadData.freqs.length - 1])}`;
    fileInfo.classList.add('visible');

    // Show S-parameter selector for 2-port files
    if (loadData.numPorts >= 2) {
        paramGroup.style.display = '';
    } else {
        paramGroup.style.display = 'none';
    }

    // Auto-fill band edges (middle 60% of data range)
    const fMin = loadData.freqs[0];
    const fMax = loadData.freqs[loadData.freqs.length - 1];
    const fSpan = fMax - fMin;
    freqLeftEl.value = +(fMin + 0.2 * fSpan).toPrecision(6);
    freqRightEl.value = +(fMax - 0.2 * fSpan).toPrecision(6);

    // Send to solver
    if (solver) {
        solver.set_load_data(loadData.freqs, loadData.re, loadData.im);
    }

    updateRunButton();
    plotLoad();
    status.textContent = 'Load data imported. Adjust band and click Run.';
    status.className = '';
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

function plotResults(response) {
    if (!loadData) return;

    const freqs = [], load_db = [], g11_db = [], s11_db = [], s21_db = [];
    for (let i = 0; i < response.length; i++) {
        const p = response[i];
        freqs.push(p.freq);
        load_db.push(p.load_db);
        g11_db.push(p.g11_db);
        s11_db.push(p.s11_db);
        s21_db.push(p.s21_db);
    }

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
            x: freqs, y: g11_db,
            mode: 'lines', name: '|G11| (matched)',
            line: { color: '#E53935', width: 2.5 }
        },
        {
            x: freqs, y: s11_db,
            mode: 'lines', name: '|S11| (filter)',
            line: { color: '#43A047', width: 1.5 }
        },
        {
            x: freqs, y: s21_db,
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
        status.textContent = 'Error: invalid frequency band.';
        status.className = 'error';
        return;
    }
    if (order < 2 || order > 12) {
        status.textContent = 'Error: order must be 2-12.';
        status.className = 'error';
        return;
    }

    // Parse transmission zeros
    const tzReArr = [], tzImArr = [];
    const tzText = tzEl.value.trim();
    if (tzText) {
        for (const tok of tzText.split(',')) {
            const val = parseFloat(tok.trim());
            if (isFinite(val)) {
                tzReArr.push(val);
                tzImArr.push(0);
            }
        }
        if (tzReArr.length > order) {
            status.textContent = 'Error: too many transmission zeros for this order.';
            status.className = 'error';
            return;
        }
    }

    runBtn.disabled = true;
    status.textContent = 'Solving...';
    status.className = 'running';

    // Run async to let UI update
    setTimeout(() => {
        try {
            const result = solver.solve(fl, fr, order, rl, tzReArr, tzImArr);

            resultsPanel.style.display = '';

            if (result.success) {
                resultBadge.innerHTML = `<span class="result-badge success">` +
                    `${result.achieved_rl_db.toFixed(1)} dB</span>`;

                const interpFreqs = [];
                for (let i = 0; i < result.interp_freqs.size(); i++) {
                    interpFreqs.push(result.interp_freqs.get(i));
                }
                resultDetails.innerHTML =
                    `<div style="font-size:0.8rem;margin-top:0.5rem">` +
                    `Order ${order}, ${tzReArr.length || 'no'} TZ` +
                    `</div>`;

                // Show coupling matrix
                displayCouplingMatrix(result);

                // Evaluate and plot response
                const fSpan = fr - fl;
                const resp = solver.evaluate_response(fl - 0.5 * fSpan, fr + 0.5 * fSpan, 501);
                const points = [];
                for (let i = 0; i < resp.length; i++) points.push(resp[i]);
                plotResults(points);

                status.textContent = `Done: ${result.achieved_rl_db.toFixed(1)} dB return loss achieved.`;
                status.className = '';
            } else {
                resultBadge.innerHTML = `<span class="result-badge error">Failed</span>`;
                resultDetails.innerHTML = `<div style="font-size:0.8rem">${result.error_message}</div>`;
                cmDiv.innerHTML = '';
                status.textContent = 'Solver failed: ' + result.error_message;
                status.className = 'error';
            }
        } catch (e) {
            resultsPanel.style.display = '';
            resultBadge.innerHTML = `<span class="result-badge error">Error</span>`;
            resultDetails.innerHTML = `<div style="font-size:0.8rem">${e.message || e}</div>`;
            cmDiv.innerHTML = '';
            status.textContent = 'Error: ' + (e.message || e);
            status.className = 'error';
        }

        runBtn.disabled = false;
    }, 50);
}

function displayCouplingMatrix(result) {
    const n = result.cm_size;
    if (n === 0) { cmDiv.innerHTML = ''; return; }

    let html = '<table>';
    for (let i = 0; i < n; i++) {
        html += '<tr>';
        for (let j = 0; j < n; j++) {
            const idx = i * n + j;
            const re = result.cm_real.get(idx);
            const im = result.cm_imag.get(idx);
            const val = Math.abs(im) < 1e-10 ? re.toFixed(4) :
                `${re.toFixed(3)}${im >= 0 ? '+' : ''}${im.toFixed(3)}j`;
            html += `<td>${val}</td>`;
        }
        html += '</tr>';
    }
    html += '</table>';
    cmDiv.innerHTML = html;
}

// ---- Helpers ----
function fmtFreq(f) {
    if (f >= 1e9) return (f / 1e9).toPrecision(4) + ' GHz';
    if (f >= 1e6) return (f / 1e6).toPrecision(4) + ' MHz';
    if (f >= 1e3) return (f / 1e3).toPrecision(4) + ' kHz';
    return f.toPrecision(4) + ' Hz';
}

// Re-plot band shading when band edges change
freqLeftEl.addEventListener('change', plotLoad);
freqRightEl.addEventListener('change', plotLoad);
