import { parseFile } from './touchstone.js';

let Module = null;
let solver = null;
let loadData = null;

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
const cmDiv        = document.getElementById('couplingMatrix');

// ---- Initialize WASM ----
async function initWasm() {
    try {
        Module = await createNpickModule();
        solver = new Module.SolverWrapper();
        statusEl.textContent = 'Ready. Load an S-parameter file to begin.';
        statusEl.className = '';
        updateRunButton();
    } catch (e) {
        statusEl.textContent = 'Failed to load WASM module: ' + e.message;
        statusEl.className = 'error';
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

    // Auto-fill band edges (middle 60% of data range)
    const fMin = loadData.freqs[0];
    const fMax = loadData.freqs[loadData.freqs.length - 1];
    const fSpan = fMax - fMin;
    freqLeftEl.value = +(fMin + 0.2 * fSpan).toPrecision(6);
    freqRightEl.value = +(fMax - 0.2 * fSpan).toPrecision(6);

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
                resultBadge.innerHTML = `<span class="result-badge success">${rlDb.toFixed(1)} dB</span>`;
                resultDetails.innerHTML =
                    `<div style="font-size:0.8rem;margin-top:0.5rem">` +
                    `Order ${order}, ${tzReArr.length || 'no'} TZ</div>`;

                displayCouplingMatrix();

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
                resultDetails.innerHTML = `<div style="font-size:0.8rem">${errMsg}</div>`;
                cmDiv.innerHTML = '';
                statusEl.textContent = 'Solver failed: ' + errMsg;
                statusEl.className = 'error';
            }
        } catch (e) {
            resultsPanel.style.display = '';
            resultBadge.innerHTML = `<span class="result-badge error">Error</span>`;
            resultDetails.innerHTML = `<div style="font-size:0.8rem">${e.message || e}</div>`;
            cmDiv.innerHTML = '';
            statusEl.textContent = 'Error: ' + (e.message || e);
            statusEl.className = 'error';
        }

        runBtn.disabled = false;
    }, 50);
}

function displayCouplingMatrix() {
    const n = solver.get_cm_size();
    if (n === 0) { cmDiv.innerHTML = ''; return; }

    let html = '<table>';
    for (let i = 0; i < n; i++) {
        html += '<tr>';
        for (let j = 0; j < n; j++) {
            const re = solver.get_cm_real(i, j);
            const im = solver.get_cm_imag(i, j);
            const val = Math.abs(im) < 1e-10 ? re.toFixed(4) :
                `${re.toFixed(3)}${im >= 0 ? '+' : ''}${im.toFixed(3)}j`;
            html += `<td>${val}</td>`;
        }
        html += '</tr>';
    }
    html += '</table>';
    cmDiv.innerHTML = html;
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
