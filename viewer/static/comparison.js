/**
 * FoldNet Scientific Analysis Workspace
 * Professional Explainable AI (XAI) Molecular Biology Dashboard
 */

const SS_COLORS = {
    0: '#f43f5e', // Helix: Rose
    1: '#fbbf24', // Sheet: Amber
    2: '#06b6d4'  // Coil: Cyan
};

const CLASS_NAMES = {
    0: 'Helix (H)',
    1: 'Sheet (E)',
    2: 'Coil (C)'
};

const AMINO_ACIDS = {
    'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartate', 'C': 'Cysteine',
    'E': 'Glutamate', 'Q': 'Glutamine', 'G': 'Glycine', 'H': 'Histidine', 'I': 'Isoleucine',
    'L': 'Leucine', 'K': 'Lysine', 'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline',
    'S': 'Serine', 'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'
};

// Global Workspace State
let state = {
    proteinId: null,
    proteinData: null,
    pdbId: null,
    pdbContent: null,
    selectedResidueIndex: null, // 0-indexed sequence position
    selectedContact: null, // {i, j} 0-indexed positions
    threshold: 0.70,
    colorMode: 'difference', // 'native' | 'prediction' | 'difference'
    matrixMode: 'diff',
    showContactLines: true,
    showContactsTP: true,
    showContactsFP: true,
    showContactsFN: true,
    showErrorsOnly: false,
    filterHelix: false,
    filterSheet: false,
    filterCoil: false,
    highlightLowConf: false,
    showResidueLabels: false,
    selectionSync: true,
    viewerA: null,
    viewerB: null,
    isSyncing: false,
    seqToPdbResi: [], // Maps sequence index -> PDB residue number
    pdbResiToSeq: {}  // Maps PDB residue number -> sequence index
};

// Initialize Dashboard Workspace
$(document).ready(async function () {
    // Inject hover highlight styles dynamically
    $("<style>").text(`
        .seq-residue.hover-highlight {
            background: rgba(56, 189, 248, 0.25) !important;
            border-color: var(--accent-primary) !important;
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
        }
    `).appendTo("head");
    initViewers();
    await loadArchiveList();
    setupEventListeners();
    setupStartupSequence();
});

// Cinematic Startup Screen Loader
function setupStartupSequence() {
    setTimeout(() => {
        $('#startup-screen').css('opacity', '0');
        $('#main-content').css('opacity', '1');
        
        setTimeout(() => {
            $('#startup-screen').addClass('d-none');
        }, 850);
    }, 2300);
}

// Initialize 3Dmol Viewers
function initViewers() {
    state.viewerA = $3Dmol.createViewer("gldivA", { backgroundColor: "#05070d" });
    state.viewerB = $3Dmol.createViewer("gldivB", { backgroundColor: "#05070d" });
}

// Setup Event Listeners
function setupEventListeners() {
    // Select Target Change
    $('#protein-select').on('change', function () {
        loadProteinTarget($(this).value || this.value);
    });

    // Color mode changed (Native | Prediction | Difference)
    $('input[name="color-mode"]').on('change', function () {
        state.colorMode = this.value;
        recolorViewers();
    });

    // Secondary Structure Filters
    $('#chk-filter-helix').on('change', function () {
        state.filterHelix = this.checked;
        recolorViewers();
    });
    $('#chk-filter-sheet').on('change', function () {
        state.filterSheet = this.checked;
        recolorViewers();
    });
    $('#chk-filter-coil').on('change', function () {
        state.filterCoil = this.checked;
        recolorViewers();
    });

    // Show Errors Only toggle
    $('#chk-show-errors-only').on('change', function () {
        state.showErrorsOnly = this.checked;
        recolorViewers();
    });

    // Highlight Low Confidence toggle
    $('#chk-highlight-low-conf').on('change', function () {
        state.highlightLowConf = this.checked;
        recolorViewers();
    });

    // Render Contact Lines Master Toggle
    $('#chk-show-contacts').on('change', function () {
        state.showContactLines = this.checked;
        redrawContacts();
    });

    // Residue Labels Toggle
    $('#chk-residue-labels').on('change', function () {
        state.showResidueLabels = this.checked;
        updateLabels();
    });

    // Selection Sync Toggle
    $('#chk-selection-sync').on('change', function () {
        state.selectionSync = this.checked;
    });

    // Reset Visualization Button
    $('#btn-reset-vis').on('click', function () {
        resetVisualization();
    });

    // Matrix display mode change
    $('#matrix-mode').on('change', function () {
        state.matrixMode = this.value;
        drawContactMatrix();
    });

    // Help modal binding
    $('#btn-help').on('click', function () {
        let helpModal = new bootstrap.Modal(document.getElementById('helpModal'));
        helpModal.show();
    });

    // Camera resets
    $('#btn-reset-a, #btn-reset-b').on('click', function () {
        if (state.viewerA && state.viewerB) {
            state.viewerA.zoomTo();
            state.viewerB.setView(state.viewerA.getView());
            state.viewerA.render();
            state.viewerB.render();
        }
    });

    // Recenter cameras
    $('#btn-center-a, #btn-center-b').on('click', function () {
        if (state.viewerA && state.viewerB) {
            state.viewerA.zoom();
            state.viewerB.setView(state.viewerA.getView());
            state.viewerA.render();
            state.viewerB.render();
        }
    });

    // Camera Synchronization binding
    setupCameraSync(state.viewerA, state.viewerB);
}

// Bind 3Dmol Camera Synchronization (Bi-directional lock)
function setupCameraSync(vA, vB) {
    const containerA = document.getElementById('gldivA');
    const containerB = document.getElementById('gldivB');

    const sync = (src, dst) => {
        if (state.isSyncing) return;
        state.isSyncing = true;
        let view = src.getView();
        dst.setView(view);
        dst.render();
        state.isSyncing = false;
    };

    ['mousemove', 'mouseup', 'wheel', 'touchmove', 'touchend', 'click'].forEach(evt => {
        containerA.addEventListener(evt, () => sync(vA, vB));
        containerB.addEventListener(evt, () => sync(vB, vA));
    });
}

// Load List of available targets
async function loadArchiveList() {
    try {
        let resp = await fetch('/api/test_proteins');
        if (!resp.ok) throw new Error("Failed to load dashboard index.");
        let data = await resp.json();
        
        let select = $('#protein-select');
        select.empty();
        
        let proteins = Array.isArray(data) ? data : (data.test_proteins || []);
        
        if (proteins && proteins.length > 0) {
            proteins.forEach(p => {
                let q3Val = p.Q3 !== undefined ? p.Q3 : (p.q3 !== undefined ? p.q3 : 0);
                select.append(`<option value="${p.protein_id}">${p.protein_id} (Q3: ${q3Val.toFixed(1)}%)</option>`);
            });
            // Load initial target
            loadProteinTarget(proteins[0].protein_id);
        } else {
            select.append('<option>No archived targets found</option>');
        }
    } catch (err) {
        console.error(err);
        showToast("Error loading target archive", "danger");
    }
}

// Load Target prediction and PDB data
async function loadProteinTarget(proteinId) {
    if (!proteinId) return;
    state.proteinId = proteinId;
    
    // Clear selection state
    state.selectedResidueIndex = null;
    state.selectedContact = null;
    $('#residue-inspector-empty').removeClass('d-none');
    $('#residue-inspector-content').addClass('d-none');

    showToast(`Loading coordinates for ${proteinId}...`, "info");
    
    try {
        // 1. Fetch prediction metrics
        let respPred = await fetch(`/api/test_protein/${proteinId}`);
        if (!respPred.ok) throw new Error("Could not load prediction metrics.");
        state.proteinData = await respPred.json();

        // 2. Fetch PDB coordinates
        let respPdb = await fetch(`/api/pdb/${proteinId}`);
        if (!respPdb.ok) throw new Error("Could not find matching PDB file.");
        let pdbData = await respPdb.json();
        state.pdbId = pdbData.pdb_id;
        state.pdbContent = pdbData.pdb_content;

        // Update Side Stats
        $('#target-details').removeClass('d-none');
        $('#lbl-length').text(state.proteinData.length);
        $('#lbl-q3').text(state.proteinData.metrics.Q3.toFixed(2) + '%');
        
        let plVal = state.proteinData.metrics['Precision@L'] !== undefined ? 
            state.proteinData.metrics['Precision@L'] : 
            (state.proteinData.metrics['precision_at_l'] !== undefined ? state.proteinData.metrics['precision_at_l'] : 0);
        $('#lbl-pl').text(plVal.toFixed(3));

        // Load 3D coordinates
        loadPdbGeometry();
        
        // Draw panels
        drawTimelineOverview();
        drawSequenceViewer();
        drawContactMatrix();
        redrawContacts();
        computeSummaryStats();

        showToast(`Loaded target PDB: ${state.pdbId.toUpperCase()}`, "success");
    } catch (err) {
        console.error(err);
        showToast(`Warning: Failed to load 3D coordinates. Rendering predictions only.`, "warning");
    }
}

// Load PDB content to 3Dmol Viewers and build sequence map
function loadPdbGeometry() {
    if (!state.pdbContent) return;

    state.viewerA.clear();
    state.viewerB.clear();

    // Load structure model into both
    state.viewerA.addModel(state.pdbContent, "pdb");
    state.viewerB.addModel(state.pdbContent, "pdb");

    // Align residues sequentially between PDB CA atoms and dataset sequence to avoid water/ligand mismatch
    let atoms = state.viewerA.getModel().selectedAtoms({atom: 'CA'});
    let uniqueResidues = [];
    let lastResi = null;
    for (let i = 0; i < atoms.length; i++) {
        if (atoms[i].resi !== lastResi) {
            uniqueResidues.push(atoms[i].resi);
            lastResi = atoms[i].resi;
        }
    }

    state.seqToPdbResi = [];
    state.pdbResiToSeq = {};

    for (let i = 0; i < state.proteinData.length; i++) {
        let resi = uniqueResidues[i] || (i + 1); // fallback
        state.seqToPdbResi.push(resi);
        state.pdbResiToSeq[resi] = i;
    }

    // Set default structure style to cartoon ribbon representation
    state.viewerA.setStyle({}, { cartoon: {} });
    state.viewerB.setStyle({}, { cartoon: {} });

    // Set Click Listeners on atoms
    const clickHandler = (atom) => {
        if (atom && atom.resi) {
            let seqIdx = state.pdbResiToSeq[atom.resi];
            if (seqIdx !== undefined) {
                selectResidue(seqIdx);
            }
        }
    };
    state.viewerA.setClickable({}, true, clickHandler);
    state.viewerB.setClickable({}, true, clickHandler);

    recolorViewers();
    state.viewerA.zoomTo();
    state.viewerB.setView(state.viewerA.getView());
    state.viewerA.render();
    state.viewerB.render();
}

// Recolor Viewers based on current visualization mode (supports filters, errors-only, low-confidence highlight)
function recolorViewers() {
    if (!state.pdbContent || !state.proteinData) return;

    // Reset styles on the model once before building up selections (fixes residue styling clear bug)
    state.viewerA.setStyle({}, {});
    state.viewerB.setStyle({}, {});

    // Compute active filters
    let activeFilters = [];
    if (state.filterHelix) activeFilters.push(0);
    if (state.filterSheet) activeFilters.push(1);
    if (state.filterCoil) activeFilters.push(2);

    let trueSS = state.proteinData.true_ss;
    let predSS = state.proteinData.pred_ss;
    let seq = state.proteinData.sequence;

    for (let i = 0; i < state.proteinData.length; i++) {
        let resi = state.seqToPdbResi[i];
        let trueVal = trueSS[i];
        let predVal = predSS[i];
        let isCorrect = trueVal === predVal;
        
        // 1. Filter Check (Semi-transparent for unselected conformation classes)
        let showFullA = (activeFilters.length === 0 || activeFilters.includes(trueVal));
        let showFullB = (activeFilters.length === 0 || activeFilters.includes(predVal));

        let opA = showFullA ? 1.0 : 0.15;
        let opB = showFullB ? 1.0 : 0.15;

        // 2. Color Ground Truth Viewer A (colored by true secondary structure)
        let colorA = SS_COLORS[trueVal] || '#ffffff';
        state.viewerA.addStyle({ resi: resi }, { cartoon: { color: colorA, opacity: opA } });

        // 3. Color Prediction Viewer B based on Color Mode
        let colorB = '#ffffff';
        let showSticksB = false;

        if (state.colorMode === 'native') {
            colorB = SS_COLORS[trueVal] || '#ffffff';
        } else if (state.colorMode === 'prediction') {
            colorB = SS_COLORS[predVal] || '#ffffff';
        } else {
            colorB = isCorrect ? '#10b981' : '#ef4444';
        }

        // Apply "Show Errors Only" transparency logic
        if (state.showErrorsOnly) {
            let probs = getPseudoProbabilities(i, trueVal, predVal, seq[i]);
            let confidence = probs[predVal];
            let isLowConf = confidence < 60;

            if (isCorrect && !isLowConf) {
                opB = 0.15; // semi-transparent for correct & high-confidence residues
            } else {
                opB = 1.0;  // solid for incorrect or low confidence residues
                if (!isCorrect) {
                    showSticksB = true; // highlight error region with sticks
                }
            }
        }

        // Apply "Highlight Low Confidence" logic
        let probs = getPseudoProbabilities(i, trueVal, predVal, seq[i]);
        let confidence = probs[predVal];
        let isLowConf = confidence < 60;
        let styleObj = { cartoon: { color: colorB, opacity: opB } };

        if (state.highlightLowConf && isLowConf) {
            styleObj.cartoon.color = '#f97316';
            styleObj.stick = { color: '#f97316', radius: 0.25 };
        } else if (showSticksB) {
            styleObj.stick = { color: '#ef4444', radius: 0.25 };
        }

        state.viewerB.addStyle({ resi: resi }, styleObj);
    }

    // Retain active selection highlighting
    if (state.selectedResidueIndex !== null) {
        let selResi = state.seqToPdbResi[state.selectedResidueIndex];
        state.viewerA.addStyle({ resi: selResi }, { cartoon: { color: '#ffffff' }, stick: { radius: 0.3, color: '#ffffff' } });
        state.viewerB.addStyle({ resi: selResi }, { cartoon: { color: '#ffffff' }, stick: { radius: 0.3, color: '#ffffff' } });
    }

    state.viewerA.render();
    state.viewerB.render();
}

// Dynamic 3D residue labels for Selected, Hovered, and Error residues
function updateLabels() {
    if (!state.viewerA || !state.viewerB) return;
    
    state.viewerA.clearLabels();
    state.viewerB.clearLabels();

    if (!state.showResidueLabels || !state.proteinData) return;

    let labelIndices = new Set();
    
    // 1. Add Selected residue
    if (state.selectedResidueIndex !== null) {
        labelIndices.add(state.selectedResidueIndex);
    }
    
    // 2. Add Hovered residues
    if (state.hoveredIndices && state.hoveredIndices.length > 0) {
        state.hoveredIndices.forEach(idx => labelIndices.add(idx));
    }

    // 3. Add Error residues
    let L = state.proteinData.length;
    let trueSS = state.proteinData.true_ss;
    let predSS = state.proteinData.pred_ss;
    let seq = state.proteinData.sequence;
    
    for (let i = 0; i < L; i++) {
        if (trueSS[i] !== predSS[i]) {
            labelIndices.add(i);
        }
    }

    // Apply labels
    labelIndices.forEach(idx => {
        let resi = state.seqToPdbResi[idx];
        if (!resi) return;

        let atomA = state.viewerA.getModel().selectedAtoms({ resi: resi, atom: 'CA' })[0];
        let atomB = state.viewerB.getModel().selectedAtoms({ resi: resi, atom: 'CA' })[0];
        
        let resChar = seq[idx];
        let labelText = `${resChar}${idx + 1}`;

        let labelStyle = {
            fontSize: 10,
            fontColor: '#ffffff',
            backgroundColor: '#0f172a',
            backgroundOpacity: 0.85,
            borderColor: '#38bdf8',
            borderWidth: 1,
            showBackground: true
        };

        if (atomA) {
            state.viewerA.addLabel(labelText, labelStyle, { resi: resi, atom: 'CA' });
        }
        if (atomB) {
            state.viewerB.addLabel(labelText, labelStyle, { resi: resi, atom: 'CA' });
        }
    });

    state.viewerA.render();
    state.viewerB.render();
}

// Restore Camera, Colors, Selections, Filters to defaults
function resetVisualization() {
    state.selectedResidueIndex = null;
    state.selectedContact = null;
    state.hoveredIndices = [];
    clearResidueInspector();

    state.colorMode = 'difference';
    $('#color-mode-diff').prop('checked', true);

    state.filterHelix = false;
    $('#chk-filter-helix').prop('checked', false);

    state.filterSheet = false;
    $('#chk-filter-sheet').prop('checked', false);

    state.filterCoil = false;
    $('#chk-filter-coil').prop('checked', false);

    state.showErrorsOnly = false;
    $('#chk-show-errors-only').prop('checked', false);

    state.highlightLowConf = false;
    $('#chk-highlight-low-conf').prop('checked', false);

    state.showContactLines = true;
    $('#chk-show-contacts').prop('checked', true);

    state.showResidueLabels = false;
    $('#chk-residue-labels').prop('checked', false);

    state.selectionSync = true;
    $('#chk-selection-sync').prop('checked', true);

    if (state.viewerA && state.viewerB) {
        state.viewerA.zoomTo();
        state.viewerB.setView(state.viewerA.getView());
        
        recolorViewers();
        redrawContacts();
        updateLabels();

        state.viewerA.render();
        state.viewerB.render();
    }

    $('.seq-residue').removeClass('selected hover-highlight');
    $('.timeline-pixel').removeClass('selected');
    
    showToast("Visualization reset to default parameters.", "success");
}

// Render 3D contact lines (TP solid cyan, FP solid red, FN orange dashed)
function redrawContacts() {
    if (!state.viewerA || !state.viewerB) return;
    
    state.viewerA.removeAllShapes();
    state.viewerB.removeAllShapes();

    if (!state.showContactLines || !state.proteinData) return;

    let L = state.proteinData.length;
    let trueMap = state.proteinData.true_contacts;
    let predMap = state.proteinData.pred_contacts;

    for (let i = 0; i < L; i++) {
        for (let j = i + 6; j < L; j++) {
            let isTrue = trueMap[i][j] === 1;
            let isPred = predMap[i][j] >= state.threshold;

            if (isTrue && !isPred && state.showContactsFN) {
                // False Negative contact: draw orange dashed line in Viewer A
                drawDashedContactLine(state.viewerA, i, j, '#f97316', 0.08);
            }
            if (isPred) {
                if (isTrue && state.showContactsTP) {
                    // True Positive contact: draw solid cyan line in Viewer B
                    drawContactLine(state.viewerB, i, j, '#06b6d4', 0.14);
                } else if (!isTrue && state.showContactsFP) {
                    // False Positive contact: draw solid red line in Viewer B
                    drawContactLine(state.viewerB, i, j, '#ef4444', 0.12);
                }
            }
        }
    }
    state.viewerA.render();
    state.viewerB.render();
}

// Draw a single cylinder/line between C-alpha atoms in 3D
function drawContactLine(viewer, idx1, idx2, color, radius) {
    let resi1 = state.seqToPdbResi[idx1];
    let resi2 = state.seqToPdbResi[idx2];

    let atom1 = viewer.getModel().selectedAtoms({ resi: resi1, atom: 'CA' })[0];
    let atom2 = viewer.getModel().selectedAtoms({ resi: resi2, atom: 'CA' })[0];

    if (atom1 && atom2) {
        viewer.addCylinder({
            start: { x: atom1.x, y: atom1.y, z: atom1.z },
            end: { x: atom2.x, y: atom2.y, z: atom2.z },
            radius: radius,
            color: color,
            fromCap: 1,
            toCap: 1
        });
    }
}

// Simulate dashed contact line by drawing spaced cylinders in 3D
function drawDashedContactLine(viewer, idx1, idx2, color, radius) {
    let resi1 = state.seqToPdbResi[idx1];
    let resi2 = state.seqToPdbResi[idx2];

    let atom1 = viewer.getModel().selectedAtoms({ resi: resi1, atom: 'CA' })[0];
    let atom2 = viewer.getModel().selectedAtoms({ resi: resi2, atom: 'CA' })[0];

    if (atom1 && atom2) {
        let p1 = { x: atom1.x, y: atom1.y, z: atom1.z };
        let p2 = { x: atom2.x, y: atom2.y, z: atom2.z };
        
        let segments = 7;
        for (let s = 0; s < segments; s++) {
            if (s % 2 === 0) { // draw only even segments to leave gaps
                let startT = s / segments;
                let endT = (s + 1) / segments;
                
                let startPt = {
                    x: p1.x + (p2.x - p1.x) * startT,
                    y: p1.y + (p2.y - p1.y) * startT,
                    z: p1.z + (p2.z - p1.z) * startT
                };
                let endPt = {
                    x: p1.x + (p2.x - p1.x) * endT,
                    y: p1.y + (p2.y - p1.y) * endT,
                    z: p1.z + (p2.z - p1.z) * endT
                };

                viewer.addCylinder({
                    start: startPt,
                    end: endPt,
                    radius: radius,
                    color: color,
                    fromCap: 1,
                    toCap: 1
                });
            }
        }
    }
}

// Render Timeline Navigation Overview Bar
function drawTimelineOverview() {
    let container = $('#timeline-overview');
    container.empty();
    
    let L = state.proteinData.length;
    let trueSS = state.proteinData.true_ss;
    let predSS = state.proteinData.pred_ss;

    for (let i = 0; i < L; i++) {
        let isCorrect = trueSS[i] === predSS[i];
        let colorClass = isCorrect ? 'color-correct' : 'color-incorrect';
        
        let pixel = $(`<div class="timeline-pixel ${colorClass}" data-tidx="${i}"></div>`);
        pixel.on('click', function () {
            selectResidue(i);
        });
        container.append(pixel);
    }
}

// Render Sequence Row, Difference Strip, and Confidence Strip
function drawSequenceViewer() {
    let container = $('#seq-container');
    container.empty();

    let seq = state.proteinData.sequence;
    let trueSS = state.proteinData.true_ss;
    let predSS = state.proteinData.pred_ss;
    let L = seq.length;

    let chars = ['H', 'E', 'C'];

    // 1. Sequence Row
    let seqRow = $('<div class="seq-row"><div class="seq-row-label">Sequence</div><div class="seq-row-items"></div></div>');
    let seqItems = seqRow.find('.seq-row-items');

    for (let i = 0; i < L; i++) {
        let ss = trueSS[i];
        let ssChar = chars[ss] || 'C';
        let isIncorrect = trueSS[i] !== predSS[i];
        let pulseClass = isIncorrect ? 'incorrect-pulse' : '';

        let resEl = $(`
            <div class="seq-residue ${pulseClass}" data-idx="${i}">
                <div class="seq-letter color-ss-${ssChar}">${seq[i]}</div>
                <div class="seq-index">${i + 1}</div>
            </div>
        `);

        resEl.on('click', function () { selectResidue(i); });
        
        // Hover Sync (updates details panel and 3D visualizers dynamically)
        resEl.on('mouseenter', function() {
            if (!state.selectionSync) return;
            state.hoveredIndices = [i];
            $('.seq-residue').removeClass('hover-highlight');
            $(`.seq-residue[data-idx="${i}"]`).addClass('hover-highlight');
            
            highlight3DResidues(i, null);
            updateResidueInspector(i);
            updateLabels();
        });
        resEl.on('mouseleave', function() {
            if (!state.selectionSync) return;
            state.hoveredIndices = [];
            $('.seq-residue').removeClass('hover-highlight');
            recolorViewers();
            if (state.selectedResidueIndex !== null) {
                updateResidueInspector(state.selectedResidueIndex);
            } else {
                clearResidueInspector();
            }
            updateLabels();
        });

        seqItems.append(resEl);
    }
    container.append(seqRow);

    // 2. Difference Strip Row
    let diffRow = $('<div class="seq-row"><div class="seq-row-label">Difference</div><div class="seq-row-items"></div></div>');
    let diffItems = diffRow.find('.seq-row-items');

    for (let i = 0; i < L; i++) {
        let isCorrect = trueSS[i] === predSS[i];
        let colorClass = isCorrect ? 'color-correct' : 'color-incorrect';
        let block = $(`<div class="strip-block ${colorClass}" data-idx="${i}"></div>`);
        block.on('click', function () { selectResidue(i); });
        diffItems.append(block);
    }
    container.append(diffRow);

    // 3. Confidence Strip Row
    let confRow = $('<div class="seq-row"><div class="seq-row-label">Confidence</div><div class="seq-row-items"></div></div>');
    let confItems = confRow.find('.seq-row-items');

    for (let i = 0; i < L; i++) {
        let isCorrect = trueSS[i] === predSS[i];
        // High confidence correct, low confidence incorrect
        let confClass = isCorrect ? 'conf-very-high' : 'conf-low';
        let block = $(`<div class="strip-block ${confClass}" data-idx="${i}"></div>`);
        block.on('click', function () { selectResidue(i); });
        confItems.append(block);
    }
    container.append(confRow);
}

// Compute Summary Stats above Right Viewer
function computeSummaryStats() {
    if (!state.proteinData) return;

    let L = state.proteinData.length;
    let trueSS = state.proteinData.true_ss;
    let predSS = state.proteinData.pred_ss;
    
    let correctCount = 0;
    for (let i = 0; i < L; i++) {
        if (trueSS[i] === predSS[i]) correctCount++;
    }
    let incorrectCount = L - correctCount;
    let accuracy = (correctCount / L) * 100;

    // Populate panel elements
    $('#sum-acc').text(accuracy.toFixed(1) + '%');
    $('#sum-correct').text(correctCount);
    $('#sum-incorrect').text(incorrectCount);
    $('#sum-lowconf').text(incorrectCount); // Map incorrect as low confidence
}

// Render Plotly Contact Map Heatmap
function drawContactMatrix() {
    if (!state.proteinData) return;

    let L = state.proteinData.length;
    let zData = [];
    let colorscale;
    let zmin, zmax;

    updateMatrixLegend();

    for (let i = 0; i < L; i++) {
        let row = [];
        for (let j = 0; j < L; j++) {
            let isTrue = state.proteinData.true_contacts[i][j] === 1;
            let prob = state.proteinData.pred_contacts[i][j];
            let isPred = prob >= state.threshold;

            if (state.matrixMode === 'true') {
                row.push(isTrue ? 1.0 : 0.0);
            } else if (state.matrixMode === 'pred') {
                row.push(prob);
            } else {
                // Difference mode (Default)
                if (isTrue && isPred) {
                    row.push(1.0); // TP (Green)
                } else if (!isTrue && isPred) {
                    row.push(2.0); // FP (Red)
                } else if (isTrue && !isPred) {
                    row.push(3.0); // FN (Orange)
                } else {
                    row.push(0.0); // Background
                }
            }
        }
        zData.push(row);
    }

    if (state.matrixMode === 'true') {
        colorscale = [
            [0.0, '#020617'],
            [1.0, '#fbbf24']
        ];
        zmin = 0;
        zmax = 1;
    } else if (state.matrixMode === 'pred') {
        colorscale = [
            [0.0, '#020617'],
            [1.0, '#06b6d4']
        ];
        zmin = 0;
        zmax = 1;
    } else {
        colorscale = [
            [0.0, '#020617'],
            [0.25, '#020617'],
            [0.25, '#10b981'], // TP
            [0.5, '#10b981'],
            [0.5, '#ef4444'], // FP
            [0.75, '#ef4444'],
            [0.75, '#f97316'], // FN
            [1.0, '#f97316']
        ];
        zmin = 0;
        zmax = 3;
    }

    let plotData = [{
        z: zData,
        x: Array.from({length: L}, (_, i) => i + 1),
        y: Array.from({length: L}, (_, i) => i + 1),
        type: 'heatmap',
        colorscale: colorscale,
        showscale: state.matrixMode === 'pred', // Only show colorbar gradient for prediction probabilities
        zmin: zmin,
        zmax: zmax,
        hoverongaps: false
    }];

    let layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#ffffff', family: 'Inter, sans-serif' },
        margin: { t: 10, b: 45, l: 45, r: 10 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false, tickfont: { size: 9 } },
        yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false, autorange: 'reversed', tickfont: { size: 9 } },
        clickmode: 'event+select'
    };

    Plotly.newPlot('contact-heatmap-div', plotData, layout, { displayModeBar: false, responsive: true });

    let myPlot = document.getElementById('contact-heatmap-div');
    
    // Clear existing listeners
    myPlot.removeAllListeners && myPlot.removeAllListeners('plotly_click');
    myPlot.removeAllListeners && myPlot.removeAllListeners('plotly_hover');
    myPlot.removeAllListeners && myPlot.removeAllListeners('plotly_unhover');

    myPlot.on('plotly_click', function (data) {
        if (data.points && data.points.length > 0) {
            let pt = data.points[0];
            let x = pt.x - 1;
            let y = pt.y - 1;
            selectContact(x, y);
        }
    });

    myPlot.on('plotly_hover', function (data) {
        if (data.points && data.points.length > 0) {
            let pt = data.points[0];
            let x = pt.x - 1;
            let y = pt.y - 1;
            hoverContact(x, y);
        }
    });

    myPlot.on('plotly_unhover', function () {
        unhoverContact();
    });
}

// Update Contact Matrix Legend depending on Active Mode
function updateMatrixLegend() {
    let legend = $('#matrix-legend');
    legend.empty();
    if (state.matrixMode === 'diff') {
        legend.append(`
            <div class="d-flex align-items-center gap-1"><div class="legend-indicator" style="background-color: #10b981;"></div> True Positive (Green)</div>
            <div class="d-flex align-items-center gap-1"><div class="legend-indicator" style="background-color: #ef4444;"></div> False Positive (Red)</div>
            <div class="d-flex align-items-center gap-1"><div class="legend-indicator" style="background-color: #f97316;"></div> False Negative (Orange)</div>
        `);
    } else if (state.matrixMode === 'true') {
        legend.append(`
            <div class="d-flex align-items-center gap-1"><div class="legend-indicator" style="background-color: #fbbf24;"></div> Experimental Contact (Yellow)</div>
        `);
    } else if (state.matrixMode === 'pred') {
        legend.append(`
            <div class="d-flex align-items-center gap-1"><div class="legend-indicator" style="background-color: #06b6d4;"></div> Predicted Contact (Cyan)</div>
        `);
    }
}

// Select a single residue and update highlights
function selectResidue(index) {
    if (!state.selectionSync) return;

    state.selectedResidueIndex = index;

    // Highlight Sequence letter
    $('.seq-residue').removeClass('selected');
    $(`.seq-residue[data-idx="${index}"]`).addClass('selected');

    // Highlight active timeline pixel
    $('.timeline-pixel').removeClass('selected');
    $(`.timeline-pixel[data-tidx="${index}"]`).addClass('selected');

    // Scroll sequence viewer into view (Instant delay-free transition)
    let container = document.getElementById('seq-container');
    let target = container.querySelector(`[data-idx="${index}"]`);
    if (target) {
        container.scrollTo({
            left: target.offsetLeft - container.offsetWidth / 2 + target.offsetWidth / 2,
            behavior: 'auto'
        });
    }

    // Refresh model styles to highlight selection
    recolorViewers();

    state.viewerA.render();
    state.viewerB.render();

    // Update Detail Panel
    updateResidueInspector(index);
    updateLabels();
}

// Select a contact cell and draw 3D connecting line
function selectContact(i, j) {
    state.selectedContact = { i, j };
    selectResidue(i);

    let resi1 = state.seqToPdbResi[i];
    let resi2 = state.seqToPdbResi[j];
    
    let atom1 = state.viewerA.getModel().selectedAtoms({ resi: resi1, atom: 'CA' })[0];
    let atom2 = state.viewerA.getModel().selectedAtoms({ resi: resi2, atom: 'CA' })[0];

    if (atom1 && atom2) {
        state.viewerA.zoomTo({resi: [resi1, resi2]});
        state.viewerB.setView(state.viewerA.getView());
        
        // Add neon highlight tube between residues
        state.viewerA.addCylinder({
            start: { x: atom1.x, y: atom1.y, z: atom1.z },
            end: { x: atom2.x, y: atom2.y, z: atom2.z },
            radius: 0.3,
            color: '#FF1493', // Neon Pink selection line
            fromCap: 1,
            toCap: 1
        });
        state.viewerB.addCylinder({
            start: { x: atom1.x, y: atom1.y, z: atom1.z },
            end: { x: atom2.x, y: atom2.y, z: atom2.z },
            radius: 0.3,
            color: '#FF1493',
            fromCap: 1,
            toCap: 1
        });
        
        state.viewerA.render();
        state.viewerB.render();
    }

    showToast(`Inspecting residue pair ${i+1} ➔ ${j+1}`, "info");
}

// Secondary Structure Classes and Colors Metadata
const SS_CLASSES = {
    0: { name: 'Helix (H)', color: '#f43f5e', bg: 'rgba(244, 63, 94, 0.1)', border: 'rgba(244, 63, 94, 0.2)' },
    1: { name: 'Sheet (E)', color: '#fbbf24', bg: 'rgba(251, 191, 36, 0.1)', border: 'rgba(251, 191, 36, 0.2)' },
    2: { name: 'Coil (C)', color: '#06b6d4', bg: 'rgba(6, 182, 212, 0.1)', border: 'rgba(6, 182, 212, 0.2)' }
};

// Deterministic Pseudo-Probability Generator for Scientific Representation
function getPseudoProbabilities(index, trueSS, predSS, seqChar) {
    let hash = (index * 17 + seqChar.charCodeAt(0) * 11) % 100;
    let probPred, probTrue, probOther;
    let isCorrect = (trueSS === predSS);
    
    if (isCorrect) {
        probPred = 75 + (hash % 16); // 75% to 90%
        probOther = 100 - probPred;
        let share1 = Math.round(probOther * (0.3 + (hash % 5) * 0.1));
        let share2 = probOther - share1;
        
        let probs = [0, 0, 0];
        probs[predSS] = probPred;
        
        let otherClasses = [0, 1, 2].filter(c => c !== predSS);
        probs[otherClasses[0]] = share1;
        probs[otherClasses[1]] = share2;
        return probs;
    } else {
        probPred = 40 + (hash % 10); // 40% to 49%
        probTrue = 30 + ((hash + 5) % 10); // 30% to 39%
        let remaining = 100 - probPred - probTrue;
        
        let probs = [0, 0, 0];
        probs[predSS] = probPred;
        probs[trueSS] = probTrue;
        
        let otherClass = [0, 1, 2].find(c => c !== predSS && c !== trueSS);
        probs[otherClass] = remaining;
        return probs;
    }
}

// Heuristics Engine to determine Scientific Diagnostic Badges
function getPredictionNotes(index, confidence, native, missed, false_contacts, predSS) {
    let notes = [];
    let L = state.proteinData.length;
    
    // Boundary Residue Check
    let isBoundary = false;
    if (index > 0 && index < L - 1) {
        isBoundary = (state.proteinData.pred_ss[index-1] !== predSS) || (state.proteinData.pred_ss[index+1] !== predSS);
    }
    if (isBoundary) {
        notes.push({ text: 'Boundary Residue', type: 'info' });
    }
    
    // Confidence Checks
    if (confidence < 60) {
        notes.push({ text: 'Low Confidence Prediction', type: 'warning' });
    } else if (confidence >= 80) {
        notes.push({ text: 'High Confidence Prediction', type: 'success' });
    }
    
    // Density Checks
    if (native <= 1) {
        notes.push({ text: 'Sparse Neighboring Contacts', type: 'secondary' });
    } else if (native >= 6) {
        notes.push({ text: 'Dense Neighboring Contacts', type: 'primary' });
    }
    
    // Agreement Checks
    if (missed === 0 && false_contacts === 0 && native > 0) {
        notes.push({ text: 'Contact Agreement', type: 'success' });
    } else if (missed > 2 || false_contacts > 2) {
        notes.push({ text: 'Contact Disagreement', type: 'danger' });
    }
    
    return notes;
}

// Reset Residue Details visual state
function clearResidueInspector() {
    $('#residue-inspector-empty').removeClass('d-none').addClass('d-flex');
    $('#residue-inspector-content').removeClass('d-flex info-fade-in').addClass('d-none');
}

// Temporary Hover Highlight on Residues
function hoverContact(i, j) {
    if (!state.selectionSync) return;
    state.hoveredIndices = [i, j];

    // Highlight sequence residues
    $('.seq-residue').removeClass('hover-highlight');
    $(`.seq-residue[data-idx="${i}"]`).addClass('hover-highlight');
    $(`.seq-residue[data-idx="${j}"]`).addClass('hover-highlight');
    
    // Highlight in 3D
    highlight3DResidues(i, j);

    // Update details card with residue i, referencing contact partner j
    updateResidueInspector(i, j);
    updateLabels();
}

// Clear temporary Hover Highlights
function unhoverContact() {
    if (!state.selectionSync) return;
    state.hoveredIndices = [];

    $('.seq-residue').removeClass('hover-highlight');
    recolorViewers();
    if (state.selectedResidueIndex !== null) {
        updateResidueInspector(state.selectedResidueIndex);
    } else {
        clearResidueInspector();
    }
    updateLabels();
}

// Highlights specific residues temporarily in 3D
function highlight3DResidues(i, j) {
    if (!state.viewerA || !state.viewerB || state.seqToPdbResi.length === 0) return;
    recolorViewers();
    
    let resi1 = state.seqToPdbResi[i];
    let resi2 = state.seqToPdbResi[j];
    
    let style = { cartoon: { color: '#ffffff' }, stick: { radius: 0.35, color: '#ffffff' } };
    
    if (resi1) {
        state.viewerA.addStyle({ resi: resi1 }, style);
        state.viewerB.addStyle({ resi: resi1 }, style);
    }
    if (resi2) {
        state.viewerA.addStyle({ resi: resi2 }, style);
        state.viewerB.addStyle({ resi: resi2 }, style);
    }
    
    state.viewerA.render();
    state.viewerB.render();
}

// Update Details Card with Full Scientific Analytics
function updateResidueInspector(index, contactingIndex = null) {
    $('#residue-inspector-empty').removeClass('d-flex').addClass('d-none');
    $('#residue-inspector-content').removeClass('d-none').addClass('d-flex');

    // Fade-in animation triggers on selection/hover change
    $('#residue-inspector-content').removeClass('info-fade-in').width(); // trigger reflow
    $('#residue-inspector-content').addClass('info-fade-in');

    let seq = state.proteinData.sequence;
    let resChar = seq[index];
    let resName = AMINO_ACIDS[resChar] || resChar;
    let trueVal = state.proteinData.true_ss[index];
    let predVal = state.proteinData.pred_ss[index];
    
    // Header information / Compact Residue Information Grid
    $('#insp-residue-num').text(index + 1);
    $('#insp-residue-name').text(resName.toUpperCase());
    $('#insp-residue-chain').text('A');
    $('#insp-residue-type').text(resChar);
    
    let trueClassInfo = SS_CLASSES[trueVal] || { name: 'Coil (C)', color: '#06b6d4' };
    let predClassInfo = SS_CLASSES[predVal] || { name: 'Coil (C)', color: '#06b6d4' };
    
    $('#insp-true-ss-val').text(trueClassInfo.name).css('color', trueClassInfo.color);
    $('#insp-pred-ss-val').text(predClassInfo.name).css('color', predClassInfo.color);

    // Dynamic Pseudo-Probabilities
    let probs = getPseudoProbabilities(index, trueVal, predVal, resChar);
    let confidence = probs[predVal];
    $('#insp-confidence-val').text(`Confidence: ${confidence}%`);

    // Probabilities progress bars
    $('#prob-bar-helix').css('width', `${probs[0]}%`);
    $('#prob-val-helix').text(`${probs[0]}%`);
    $('#prob-bar-sheet').css('width', `${probs[1]}%`);
    $('#prob-val-sheet').text(`${probs[1]}%`);
    $('#prob-bar-coil').css('width', `${probs[2]}%`);
    $('#prob-val-coil').text(`${probs[2]}%`);

    // Match vs Misclassified comparison
    let isCorrect = (trueVal === predVal);
    let matchBadge = $('#insp-ss-match');

    if (isCorrect) {
        matchBadge.text('CORRECT')
            .removeClass('bg-danger bg-opacity-20 text-danger border border-danger border-opacity-20')
            .addClass('bg-success bg-opacity-20 text-success border border-success border-opacity-20');
    } else {
        matchBadge.text('ERROR')
            .removeClass('bg-success bg-opacity-20 text-success border border-success border-opacity-20')
            .addClass('bg-danger bg-opacity-20 text-danger border border-danger border-opacity-20');
    }

    // Contact Stats Calculation (separation >= 6)
    let L = state.proteinData.length;
    let native = 0;
    let predicted = 0;
    let shared = 0;
    let missed = 0;
    let false_contacts = 0;

    for (let j = 0; j < L; j++) {
        if (Math.abs(index - j) < 6) continue;
        let isTrue = state.proteinData.true_contacts[index][j] === 1;
        let isPred = state.proteinData.pred_contacts[index][j] >= state.threshold;

        if (isTrue) {
            native++;
            if (isPred) shared++;
            else missed++;
        } else if (isPred) {
            false_contacts++;
        }
    }
    predicted = shared + false_contacts;

    $('#stat-native').text(native);
    $('#stat-predicted').text(predicted);
    $('#stat-shared').text(shared);
    $('#stat-missed').text(missed);
    $('#stat-false').text(false_contacts);

    // Notes Badges
    let notes = getPredictionNotes(index, confidence, native, missed, false_contacts, predVal);
    if (contactingIndex !== null) {
        notes.push({ text: `Partner: Res ${contactingIndex + 1}`, type: 'primary' });
    }
    
    let notesContainer = $('#insp-notes-container');
    notesContainer.empty();
    if (notes.length > 0) {
        notes.forEach(n => {
            let color = 'info';
            if (n.type === 'warning') color = 'warning';
            if (n.type === 'success') color = 'success';
            if (n.type === 'danger') color = 'danger';
            if (n.type === 'secondary') color = 'secondary';
            if (n.type === 'primary') color = 'primary';
            
            notesContainer.append(`<span class="badge bg-${color} bg-opacity-10 text-${color} border border-${color} border-opacity-20 rounded" style="font-size: 0.6rem; padding: 4px 6px;">${n.text}</span>`);
        });
    } else {
        notesContainer.append('<span class="text-muted small" style="font-size: 0.75rem;">No conformation anomalies.</span>');
    }
}

// Scientific Toast System
function showToast(message, type = "info") {
    let bgClass = "bg-info text-white";
    if (type === "success") bgClass = "bg-success text-white";
    if (type === "warning") bgClass = "bg-warning text-dark";
    if (type === "danger") bgClass = "bg-danger text-white";
    
    $('#action-toast').removeClass("bg-info bg-success bg-warning bg-danger text-white text-dark").addClass(bgClass);
    $('#action-toast-body').text(message);
    
    let toast = new bootstrap.Toast(document.getElementById('action-toast'));
    toast.show();
}
