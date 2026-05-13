
/**
 * FoldNet – 2D Secondary Structure Visualiser
 * Draws proper protein topology with helices (cylinders/ribbons), strands (arrows), coils (lines)
 */

/**
 * Parse raw SS array into segments: [{type:'H'|'E'|'C', start, end}]
 */
function parseSSSegments(ssArray, classToChar) {
    const segments = [];
    if (!ssArray || ssArray.length === 0) return segments;
    let cur = classToChar[ssArray[0]];
    let start = 0;
    for (let i = 1; i <= ssArray.length; i++) {
        const c = i < ssArray.length ? classToChar[ssArray[i]] : null;
        if (c !== cur) {
            segments.push({ type: cur, start, end: i - 1 });
            cur = c;
            start = i;
        }
    }
    return segments;
}

/**
 * Draw a rich SVG-based 2D topology diagram for the sequence.
 * Helices = pink cylinders (rounded rectangles), Strands = teal arrows, Coils = thin grey lines
 */
function draw2DStructure(containerId, ssArray, classToChar, sequence) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    const segments = parseSSSegments(ssArray, classToChar);
    const L = ssArray.length;
    if (L === 0) return;

    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const W = container.clientWidth || 800;
    const H = 160;
    const PAD = 20;
    const drawW = W - PAD * 2;

    // Color scheme
    const colors = {
        H: { fill: '#e63946', stroke: '#c1121f', text: '#fff' },
        E: { fill: '#f4a261', stroke: '#e76f51', text: '#fff' },
        C: { fill: isDark ? '#4b5563' : '#9ca3af', stroke: isDark ? '#6b7280' : '#6b7280', text: isDark ? '#e5e7eb' : '#fff' }
    };
    const bgColor = isDark ? '#1f2937' : '#f9fafb';
    const lineColor = isDark ? '#4b5563' : '#cbd5e1';
    const textColor = isDark ? '#9ca3af' : '#6b7280';

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', H);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.style.display = 'block';

    // Background
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('width', W); bg.setAttribute('height', H);
    bg.setAttribute('fill', bgColor); bg.setAttribute('rx', 10);
    svg.appendChild(bg);

    // Central baseline
    const midY = H / 2;
    const lineH = 4; // baseline height
    const baseline = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    baseline.setAttribute('x', PAD); baseline.setAttribute('y', midY - lineH / 2);
    baseline.setAttribute('width', drawW); baseline.setAttribute('height', lineH);
    baseline.setAttribute('fill', lineColor); baseline.setAttribute('rx', 2);
    svg.appendChild(baseline);

    // Draw each segment
    segments.forEach(seg => {
        const xStart = PAD + (seg.start / L) * drawW;
        const xEnd = PAD + ((seg.end + 1) / L) * drawW;
        const segW = xEnd - xStart;
        const col = colors[seg.type];
        const label = seg.type === 'H' ? 'Helix' : seg.type === 'E' ? 'Strand' : 'Coil';
        const len = seg.end - seg.start + 1;

        if (seg.type === 'H') {
            // Helix: tall rounded rectangle with cylinder gradient
            const hH = 54;
            const hY = midY - hH / 2;
            const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            const gid = `hgrad_${seg.start}`;
            const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            grad.setAttribute('id', gid);
            grad.setAttribute('x1', '0%'); grad.setAttribute('y1', '0%');
            grad.setAttribute('x2', '0%'); grad.setAttribute('y2', '100%');
            const s1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            s1.setAttribute('offset', '0%'); s1.setAttribute('stop-color', lighten(col.fill, 40));
            const s2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            s2.setAttribute('offset', '50%'); s2.setAttribute('stop-color', col.fill);
            const s3 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            s3.setAttribute('offset', '100%'); s3.setAttribute('stop-color', col.stroke);
            grad.appendChild(s1); grad.appendChild(s2); grad.appendChild(s3);
            defs.appendChild(grad);
            svg.appendChild(defs);

            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', xStart); rect.setAttribute('y', hY);
            rect.setAttribute('width', segW); rect.setAttribute('height', hH);
            rect.setAttribute('rx', 10); rect.setAttribute('ry', 10);
            rect.setAttribute('fill', `url(#${gid})`);
            rect.setAttribute('stroke', col.stroke); rect.setAttribute('stroke-width', '1.5');
            addTooltip(rect, `${label}: residues ${seg.start + 1}–${seg.end + 1} (L=${len})`);
            svg.appendChild(rect);

            // Helix coil lines
            if (segW > 20) {
                for (let lx = xStart + 8; lx < xEnd - 4; lx += 8) {
                    const vl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    vl.setAttribute('x1', lx); vl.setAttribute('y1', hY + 4);
                    vl.setAttribute('x2', lx); vl.setAttribute('y2', hY + hH - 4);
                    vl.setAttribute('stroke', 'rgba(255,255,255,0.2)');
                    vl.setAttribute('stroke-width', '1');
                    svg.appendChild(vl);
                }
            }

            // Label
            if (segW > 30) {
                const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                txt.setAttribute('x', xStart + segW / 2);
                txt.setAttribute('y', midY + 5);
                txt.setAttribute('text-anchor', 'middle');
                txt.setAttribute('font-size', segW > 50 ? '11' : '9');
                txt.setAttribute('font-family', "'Inter', sans-serif");
                txt.setAttribute('fill', col.text);
                txt.setAttribute('font-weight', '600');
                txt.textContent = segW > 50 ? 'α' : 'H';
                svg.appendChild(txt);
            }

        } else if (seg.type === 'E') {
            // Strand: arrow shape
            const sH = 40;
            const sY = midY - sH / 2;
            const arrowW = Math.min(16, segW * 0.3); // arrow head width
            const bodyW = segW - arrowW;

            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            const bx = xStart;
            const d = bodyW > 0
                ? `M ${bx} ${sY + 8} L ${bx + bodyW} ${sY + 8} L ${bx + bodyW} ${sY} L ${xEnd} ${midY} L ${bx + bodyW} ${sY + sH} L ${bx + bodyW} ${sY + sH - 8} L ${bx} ${sY + sH - 8} Z`
                : `M ${bx} ${sY} L ${xEnd} ${midY} L ${bx} ${sY + sH} Z`;
            path.setAttribute('d', d);
            path.setAttribute('fill', col.fill);
            path.setAttribute('stroke', col.stroke);
            path.setAttribute('stroke-width', '1.5');
            path.setAttribute('stroke-linejoin', 'round');
            addTooltip(path, `β-Strand: residues ${seg.start + 1}–${seg.end + 1} (L=${len})`);
            svg.appendChild(path);

            if (segW > 35) {
                const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                txt.setAttribute('x', xStart + bodyW / 2);
                txt.setAttribute('y', midY + 4);
                txt.setAttribute('text-anchor', 'middle');
                txt.setAttribute('font-size', segW > 55 ? '11' : '9');
                txt.setAttribute('font-family', "'Inter', sans-serif");
                txt.setAttribute('fill', col.text);
                txt.setAttribute('font-weight', '600');
                txt.textContent = 'β';
                svg.appendChild(txt);
            }

        } else {
            // Coil: thin squiggly line (represented as a thin rect or sine wave)
            const coilH = 6;
            if (segW > 4) {
                const cr = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                cr.setAttribute('x', xStart); cr.setAttribute('y', midY - coilH / 2);
                cr.setAttribute('width', segW); cr.setAttribute('height', coilH);
                cr.setAttribute('rx', 3);
                cr.setAttribute('fill', col.fill);
                cr.setAttribute('stroke', col.stroke);
                cr.setAttribute('stroke-width', '1');
                addTooltip(cr, `Coil: residues ${seg.start + 1}–${seg.end + 1} (L=${len})`);
                svg.appendChild(cr);
            }
        }
    });

    // Residue number labels
    const tickPositions = getTickPositions(L);
    tickPositions.forEach(pos => {
        const tx = PAD + (pos / L) * drawW;
        const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        tick.setAttribute('x1', tx); tick.setAttribute('y1', midY + 30);
        tick.setAttribute('x2', tx); tick.setAttribute('y2', midY + 36);
        tick.setAttribute('stroke', textColor); tick.setAttribute('stroke-width', '1');
        svg.appendChild(tick);

        const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        lbl.setAttribute('x', tx);
        lbl.setAttribute('y', midY + 48);
        lbl.setAttribute('text-anchor', 'middle');
        lbl.setAttribute('font-size', '10');
        lbl.setAttribute('font-family', "'Inter', sans-serif");
        lbl.setAttribute('fill', textColor);
        lbl.textContent = pos + 1;
        svg.appendChild(lbl);
    });

    container.appendChild(svg);
}

function lighten(hex, amount) {
    // Quick hex lightener
    hex = hex.replace('#', '');
    let r = parseInt(hex.substr(0, 2), 16);
    let g = parseInt(hex.substr(2, 2), 16);
    let b = parseInt(hex.substr(4, 2), 16);
    r = Math.min(255, r + amount);
    g = Math.min(255, g + amount);
    b = Math.min(255, b + amount);
    return `rgb(${r},${g},${b})`;
}

function getTickPositions(L) {
    const step = L <= 50 ? 10 : L <= 100 ? 20 : L <= 200 ? 50 : 100;
    const ticks = [];
    for (let i = step - 1; i < L; i += step) ticks.push(i);
    return ticks;
}

function addTooltip(el, text) {
    el.style.cursor = 'pointer';
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = text;
    el.appendChild(title);
}

/**
 * Draw a binary contact map (thresholded) as an SVG dot-plot
 * This renders contacts as dots on a 2D grid for a crisp protein contact map look.
 */
function drawBinaryContactMap(containerId, contactMatrix, threshold = 0.5) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    const L = contactMatrix.length;
    if (L === 0) return;

    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const size = Math.min(container.clientWidth || 400, 450);
    const PAD = 30;
    const gridSize = size - PAD * 2;
    const cellSize = gridSize / L;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    svg.style.display = 'block';
    svg.style.margin = '0 auto';

    const bgColor = isDark ? '#111827' : '#f8fafc';
    const gridColor = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
    const dotColor = isDark ? '#818cf8' : '#4f46e5';
    const axisColor = isDark ? '#4b5563' : '#cbd5e1';
    const textColor = isDark ? '#9ca3af' : '#6b7280';

    // Background
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('width', size); bg.setAttribute('height', size);
    bg.setAttribute('fill', bgColor); bg.setAttribute('rx', 8);
    svg.appendChild(bg);

    // Grid lines every 10 residues
    const step = L <= 50 ? 10 : L <= 100 ? 20 : 50;
    for (let i = 0; i <= L; i += step) {
        const pos = PAD + (i / L) * gridSize;
        const hl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        hl.setAttribute('x1', PAD); hl.setAttribute('y1', pos);
        hl.setAttribute('x2', PAD + gridSize); hl.setAttribute('y2', pos);
        hl.setAttribute('stroke', gridColor); hl.setAttribute('stroke-width', '0.5');
        svg.appendChild(hl);

        const vl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        vl.setAttribute('x1', pos); vl.setAttribute('y1', PAD);
        vl.setAttribute('x2', pos); vl.setAttribute('y2', PAD + gridSize);
        vl.setAttribute('stroke', gridColor); vl.setAttribute('stroke-width', '0.5');
        svg.appendChild(vl);

        if (i > 0 && i < L) {
            const xl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            xl.setAttribute('x', pos); xl.setAttribute('y', PAD - 4);
            xl.setAttribute('text-anchor', 'middle');
            xl.setAttribute('font-size', '9');
            xl.setAttribute('font-family', "'Inter',sans-serif");
            xl.setAttribute('fill', textColor);
            xl.textContent = i;
            svg.appendChild(xl);

            const yl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            yl.setAttribute('x', PAD - 4); yl.setAttribute('y', pos + 3);
            yl.setAttribute('text-anchor', 'end');
            yl.setAttribute('font-size', '9');
            yl.setAttribute('font-family', "'Inter',sans-serif");
            yl.setAttribute('fill', textColor);
            yl.textContent = i;
            svg.appendChild(yl);
        }
    }

    // Axes
    const axisRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    axisRect.setAttribute('x', PAD); axisRect.setAttribute('y', PAD);
    axisRect.setAttribute('width', gridSize); axisRect.setAttribute('height', gridSize);
    axisRect.setAttribute('fill', 'none');
    axisRect.setAttribute('stroke', axisColor); axisRect.setAttribute('stroke-width', '1.5');
    svg.appendChild(axisRect);

    // Diagonal line
    const diag = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    diag.setAttribute('x1', PAD); diag.setAttribute('y1', PAD);
    diag.setAttribute('x2', PAD + gridSize); diag.setAttribute('y2', PAD + gridSize);
    diag.setAttribute('stroke', axisColor); diag.setAttribute('stroke-width', '0.5');
    diag.setAttribute('stroke-dasharray', '3,3');
    svg.appendChild(diag);

    // Dots for contacts
    const dotR = Math.max(0.8, Math.min(2.5, cellSize * 0.4));
    for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
            if (contactMatrix[i][j] >= threshold) {
                const prob = contactMatrix[i][j];
                const cx = PAD + (j / L) * gridSize + cellSize / 2;
                const cy = PAD + (i / L) * gridSize + cellSize / 2;
                const cx2 = PAD + (i / L) * gridSize + cellSize / 2;
                const cy2 = PAD + (j / L) * gridSize + cellSize / 2;

                // Upper triangle
                const dot1 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                dot1.setAttribute('cx', cx); dot1.setAttribute('cy', cy);
                dot1.setAttribute('r', dotR);
                dot1.setAttribute('fill', dotColor);
                dot1.setAttribute('opacity', Math.min(1, 0.4 + prob * 0.6));
                svg.appendChild(dot1);

                // Lower triangle (mirror)
                const dot2 = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                dot2.setAttribute('cx', cx2); dot2.setAttribute('cy', cy2);
                dot2.setAttribute('r', dotR);
                dot2.setAttribute('fill', dotColor);
                dot2.setAttribute('opacity', Math.min(1, 0.4 + prob * 0.6));
                svg.appendChild(dot2);
            }
        }
    }

    container.appendChild(svg);
}

/**
 * Compute contact map statistics
 */
function computeContactStats(contactMatrix, threshold = 0.5) {
    const L = contactMatrix.length;
    let total = 0, sr = 0, mr = 0, lr = 0;
    for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) {
            if (contactMatrix[i][j] >= threshold) {
                total++;
                const sep = j - i;
                if (sep <= 6) sr++;
                else if (sep <= 24) mr++;
                else lr++;
            }
        }
    }
    return { total, sr, mr, lr, L };
}

// Export
window.draw2DStructure = draw2DStructure;
window.drawBinaryContactMap = drawBinaryContactMap;
window.computeContactStats = computeContactStats;
window.parseSSSegments = parseSSSegments;
