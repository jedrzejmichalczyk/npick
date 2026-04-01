/**
 * Touchstone file parser (.s1p, .s2p).
 *
 * Returns { freqs: number[], re: number[], im: number[], freq_unit: string }
 * where re + j*im is the reflection coefficient (S11 for .s1p, selectable for .s2p).
 */

export function parseTouchstone(text, portParam = 'S11') {
    const lines = text.split(/\r?\n/);
    let freqMultiplier = 1;
    let format = 'MA'; // MA = magnitude/angle, RI = real/imag, DB = dB/angle
    let numPorts = 1;
    let freqUnit = 'Hz';

    const freqs = [];
    const re = [];
    const im = [];

    for (const rawLine of lines) {
        const line = rawLine.trim();

        // Skip empty lines and comments
        if (!line || line.startsWith('!')) continue;

        // Option line
        if (line.startsWith('#')) {
            const tokens = line.substring(1).trim().toUpperCase().split(/\s+/);
            for (let i = 0; i < tokens.length; i++) {
                switch (tokens[i]) {
                    case 'HZ':  freqMultiplier = 1;    freqUnit = 'Hz'; break;
                    case 'KHZ': freqMultiplier = 1e3;  freqUnit = 'kHz'; break;
                    case 'MHZ': freqMultiplier = 1e6;  freqUnit = 'MHz'; break;
                    case 'GHZ': freqMultiplier = 1e9;  freqUnit = 'GHz'; break;
                    case 'MA':  format = 'MA'; break;
                    case 'RI':  format = 'RI'; break;
                    case 'DB':  format = 'DB'; break;
                }
            }
            continue;
        }

        // Data line
        const values = line.split(/\s+/).map(Number);
        if (values.length < 3 || values.some(isNaN)) continue;

        const freq = values[0] * freqMultiplier;

        // Determine which pair of values to use
        // .s1p: freq val1 val2
        // .s2p: freq S11_v1 S11_v2 S21_v1 S21_v2 S12_v1 S12_v2 S22_v1 S22_v2
        let v1, v2;
        if (values.length >= 9) {
            // 2-port data
            numPorts = 2;
            const paramIndex = { 'S11': 1, 'S21': 3, 'S12': 5, 'S22': 7 };
            const idx = paramIndex[portParam] || 1;
            v1 = values[idx];
            v2 = values[idx + 1];
        } else {
            // 1-port data
            v1 = values[1];
            v2 = values[2];
        }

        // Convert to real/imag
        let rVal, iVal;
        if (format === 'RI') {
            rVal = v1;
            iVal = v2;
        } else if (format === 'MA') {
            const mag = v1;
            const angRad = v2 * Math.PI / 180;
            rVal = mag * Math.cos(angRad);
            iVal = mag * Math.sin(angRad);
        } else if (format === 'DB') {
            const mag = Math.pow(10, v1 / 20);
            const angRad = v2 * Math.PI / 180;
            rVal = mag * Math.cos(angRad);
            iVal = mag * Math.sin(angRad);
        }

        freqs.push(freq);
        re.push(rVal);
        im.push(iVal);
    }

    return { freqs, re, im, freqUnit, numPorts };
}

/**
 * Parse a simple CSV with columns: freq, re, im
 */
export function parseCSV(text) {
    const lines = text.split(/\r?\n/);
    const freqs = [];
    const re = [];
    const im = [];

    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#') || line.startsWith('freq')) continue;

        const values = line.split(/[,\s]+/).map(Number);
        if (values.length >= 3 && !values.slice(0, 3).some(isNaN)) {
            freqs.push(values[0]);
            re.push(values[1]);
            im.push(values[2]);
        }
    }

    return { freqs, re, im, freqUnit: 'Hz', numPorts: 1 };
}

/**
 * Auto-detect format and parse.
 */
export function parseFile(text, filename, portParam = 'S11') {
    const ext = filename.toLowerCase().split('.').pop();
    if (ext === 's1p' || ext === 's2p' || ext === 's3p' || ext === 's4p') {
        return parseTouchstone(text, portParam);
    }
    return parseCSV(text);
}
