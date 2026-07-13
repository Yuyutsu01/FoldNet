/**
 * FoldNet Structural Comparison Workspace
 * Professional Molecular Visualization & Interactive Analysis Panel
 */

const SS_COLORS = {
    0: '#f43f5e', // Helix: Rose
    1: '#fbbf24', // Strand: Amber
    2: '#06b6d4'  // Coil: Cyan
};

const CLASS_NAMES = {
    0: 'Helix (H)',
    1: 'Strand (E)',
    2: 'Coil (C)'
};

const AMINO_ACIDS = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
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
    coloringMode: 'ss',
    matrixMode: 'dual',
    showContactLines: true,
    showTrueContacts: true,
    showPredContacts: true,
    viewerA: null,
    viewerB: null,
    isSyncing: false,
    seqToPdbResi: [], // Maps sequence index -> PDB residue number
    pdbResiToSeq: {}  // Maps PDB residue number -> sequence index
};

// Initialize Dashboard Workspace
$(document).ready(async function () {
    initViewers();
    await loadArchiveList();
    setupEventListeners();
});

// Initialize 3Dmol Viewers
function initViewers() {
    state.viewerA = $3Dmol.createViewer("gldivA", { backgroundColor: "#0b0f19" });
    state.viewerB = $3Dmol.createViewer("gldivB", { backgroundColor: "#0b0f19" });
}

// Setup Event Listeners
function setupEventListeners() {
    // Select Target Change
    $('#protein-select').on('change', function () {
        loadProteinTarget($(this).value || this.value);
    });

    // Coloring mode changed
    $('#coloring-mode').on('change', function () {
        state.coloringMode = this.value;
        recolorViewers();
    });

    // Toggle contact lines
    $('#chk-show-contacts, #chk-contacts-true, #chk-contacts-pred').on('change', function () {
        state.showContactLines = $('#chk-show-contacts').is(':checked');
        state.showTrueContacts = $('#chk-contacts-true').is(':checked');
        state.showPredContacts = $('#chk-contacts-pred').is(':checked');
        redrawContacts();
    });

    // Threshold slider change
    $('#threshold-slider').on('input', function () {
        let val = parseFloat(this.value);
        $('#threshold-val').text(val.toFixed(2));
        state.threshold = val;
        redrawContacts();
        drawContactMatrix();
    });

    // Matrix display mode change
    $('#matrix-mode').on('change', function () {
        state.matrixMode = this.value;
        drawContactMatrix();
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
        
        if (data.test_proteins && data.test_proteins.length > 0) {
            data.test_proteins.forEach(p => {
                select.append(`<option value="${p.protein_id}">${p.protein_id} (Q3: ${p.q3.toFixed(1)}%)</option>`);
            });
            // Load initial target
            loadProteinTarget(data.test_proteins[0].protein_id);
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
        $('#lbl-pl').text(state.proteinData.metrics.Precision@L ? state.proteinData.metrics.Precision@L.toFixed(3) : state.proteinData.metrics['Precision@L'].toFixed(3));

        // Load 3D coordinates
        loadPdbGeometry();
        
        // Draw panels
        drawSequenceViewer();
        drawContactMatrix();
        redrawContacts();

        showToast(`Successfully loaded target PDB: ${state.pdbId.toUpperCase()}`, "success");
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

    // Align residues sequentially between PDB atoms and dataset sequence
    let atoms = state.viewerA.getModel().selectedAtoms({});
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

    // Set structure style
    state.viewerA.setStyle({}, { cartoon: { style: 'rect', thickness: 0.1 } });
    state.viewerB.setStyle({}, { cartoon: { style: 'rect', thickness: 0.1 } });

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

// Recolor Viewers based on current visualization mode
function recolorViewers() {
    if (!state.pdbContent) return;

    for (let i = 0; i < state.proteinData.length; i++) {
        let resi = state.seqToPdbResi[i];
        let colorA = '#FFFFFF';
        let colorB = '#FFFFFF';

        if (state.coloringMode === 'ss') {
            colorA = SS_COLORS[state.proteinData.true_ss[i]] || '#ffffff';
            colorB = SS_COLORS[state.proteinData.pred_ss[i]] || '#ffffff';
        } else if (state.coloringMode === 'correctness') {
            let isCorrect = state.proteinData.true_ss[i] === state.proteinData.pred_ss[i];
            colorA = isCorrect ? '#10b981' : '#ef4444';
            colorB = colorA;
        } else if (state.coloringMode === 'confidence') {
            // Confidence gradient based on match/prob
            let isCorrect = state.proteinData.true_ss[i] === state.proteinData.pred_ss[i];
            colorA = '#64748b'; // native constant grey
            colorB = isCorrect ? '#047857' : '#b91c1c'; // dark emerald vs dark red
        }

        // Apply style to Cartoon representation
        state.viewerA.setStyle({ resi: resi }, { cartoon: { color: colorA } });
        state.viewerB.setStyle({ resi: resi }, { cartoon: { color: colorB } });
    }

    // Retain selection highlighting
    if (state.selectedResidueIndex !== null) {
        let selResi = state.seqToPdbResi[state.selectedResidueIndex];
        state.viewerA.setStyle({ resi: selResi }, { cartoon: { color: '#ffffff' }, stick: { radius: 0.25, color: '#ffffff' } });
        state.viewerB.setStyle({ resi: selResi }, { cartoon: { color: '#ffffff' }, stick: { radius: 0.25, color: '#ffffff' } });
    }

    state.viewerA.render();
    state.viewerB.render();
}

// Render 3D contact lines
function redrawContacts() {
    if (!state.viewerA || !state.viewerB) return;
    
    // Clear previous shapes
    state.viewerA.removeAllShapes();
    state.viewerB.removeAllShapes();

    if (!state.showContactLines || !state.proteinData) return;

    let modelA = state.viewerA.getModel();
    let modelB = state.viewerB.getModel();
    if (!modelA || !modelB) return;

    let L = state.proteinData.length;
    let trueMap = state.proteinData.true_contacts;
    let predMap = state.proteinData.pred_contacts;

    for (let i = 0; i < L; i++) {
        for (let j = i + 6; j < L; j++) { // Evaluate with separation >= 6 (Medium & Long range)
            let isTrue = trueMap[i][j] === 1;
            let isPred = predMap[i][j] >= state.threshold;

            if (isTrue && state.showTrueContacts && !isPred) {
                // False Negative contact: draw thin grey line
                drawContactLine(state.viewerA, i, j, '#4b5563', 0.08);
            }
            if (isPred && state.showPredContacts) {
                if (isTrue) {
                    // True Positive contact: draw cyan line
                    drawContactLine(state.viewerB, i, j, '#06b6d4', 0.15);
                } else {
                    // False Positive contact: draw red line
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

// Render Sequence Stream
function drawSequenceViewer() {
    let container = $('#seq-container');
    container.empty();

    let seq = state.proteinData.sequence;
    let trueSS = state.proteinData.true_ss;

    let chars = ['H', 'E', 'C'];

    for (let i = 0; i < seq.length; i++) {
        let ss = trueSS[i];
        let ssChar = chars[ss] || 'C';
        
        let resEl = $(`
            <div class="seq-residue" data-idx="${i}">
                <div class="seq-letter color-ss-${ssChar}">${seq[i]}</div>
                <div class="seq-index">${i + 1}</div>
            </div>
        `);

        resEl.on('click', function () {
            selectResidue(i);
        });

        container.append(resEl);
    }
}

// Render Plotly Contact Map Heatmap
function drawContactMatrix() {
    if (!state.proteinData) return;

    let L = state.proteinData.length;
    let zData = [];
    
    // Fill Matrix data
    for (let i = 0; i < L; i++) {
        let row = [];
        for (let j = 0; j < L; j++) {
            if (state.matrixMode === 'dual') {
                // Diagonal overlay: upper predicted, lower true contacts
                if (i <= j) {
                    row.push(state.proteinData.pred_contacts[i][j]);
                } else {
                    row.push(state.proteinData.true_contacts[i][j]);
                }
            } else {
                // Difference map
                row.push(state.proteinData.true_contacts[i][j] - state.proteinData.pred_contacts[i][j]);
            }
        }
        zData.push(row);
    }

    let colorscale = state.matrixMode === 'dual' ? 'Viridis' : 'RdBu';

    let plotData = [{
        z: zData,
        x: Array.from({length: L}, (_, i) => i + 1),
        y: Array.from({length: L}, (_, i) => i + 1),
        type: 'heatmap',
        colorscale: colorscale,
        showscale: true,
        zmin: state.matrixMode === 'dual' ? 0 : -1,
        zmax: 1
    }];

    let layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#64748b', family: 'Inter, sans-serif' },
        margin: { t: 10, b: 40, l: 40, r: 10 },
        xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
        yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zeroline: false },
        clickmode: 'event+select'
    };

    Plotly.newPlot('contact-heatmap-div', plotData, layout, { displayModeBar: false });

    // Handle Contact Map Clicks
    let myPlot = document.getElementById('contact-heatmap-div');
    myPlot.on('plotly_click', function (data) {
        if (data.points && data.points.length > 0) {
            let pt = data.points[0];
            let x = pt.x - 1;
            let y = pt.y - 1;
            selectContact(x, y);
        }
    });
}

// Select a single residue and update highlights
function selectResidue(index) {
    state.selectedResidueIndex = index;

    // Highlight Sequence letter
    $('.seq-residue').removeClass('selected');
    $(`.seq-residue[data-idx="${index}"]`).addClass('selected');

    // Scroll sequence viewer into view
    let container = document.getElementById('seq-container');
    let target = container.querySelector(`[data-idx="${index}"]`);
    if (target) {
        container.scrollTo({
            left: target.offsetLeft - container.offsetWidth / 2 + target.offsetWidth / 2,
            behavior: 'smooth'
        });
    }

    // Refresh model styles to highlight selection
    recolorViewers();

    // Render 3D selection highlights
    let resi = state.seqToPdbResi[index];
    state.viewerA.setStyle({ resi: resi }, { cartoon: { color: '#ffffff' }, stick: { radius: 0.25, color: '#ffffff' } });
    state.viewerB.setStyle({ resi: resi }, { cartoon: { color: '#ffffff' }, stick: { radius: 0.25, color: '#ffffff' } });
    state.viewerA.render();
    state.viewerB.render();

    // Update Detail Panel
    updateResidueInspector(index);
}

// Select a contact cell and draw 3D connecting line
function selectContact(i, j) {
    state.selectedContact = { i, j };
    
    // Select residues associated with contact
    selectResidue(i);

    // Zoom camera on contact region in both viewers
    let resi1 = state.seqToPdbResi[i];
    let resi2 = state.seqToPdbResi[j];
    
    let atom1 = state.viewerA.getModel().selectedAtoms({ resi: resi1, atom: 'CA' })[0];
    let atom2 = state.viewerA.getModel().selectedAtoms({ resi: resi2, atom: 'CA' })[0];

    if (atom1 && atom2) {
        let center = {
            x: (atom1.x + atom2.x) / 2,
            y: (atom1.y + atom2.y) / 2,
            z: (atom1.z + atom2.z) / 2
        };
        
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

// Update Inspector Card
function updateResidueInspector(index) {
    $('#residue-inspector-empty').addClass('d-none');
    $('#residue-inspector-content').removeClass('d-none');

    let seq = state.proteinData.sequence;
    let resChar = seq[index];
    let resName = AMINO_ACIDS[resChar] || resChar;
    let trueVal = state.proteinData.true_ss[index];
    let predVal = state.proteinData.pred_ss[index];
    
    $('#insp-num').text(`${index + 1} (PDB: ${state.seqToPdbResi[index]})`);
    $('#insp-name').text(`${resName} [${resChar}]`);
    
    let trueText = CLASS_NAMES[trueVal] || 'Coil (C)';
    let predText = CLASS_NAMES[predVal] || 'Coil (C)';
    
    $('#insp-true-ss').text(trueText).css('color', SS_COLORS[trueVal]);
    $('#insp-pred-ss').text(predText).css('color', SS_COLORS[predVal]);
    
    let match = trueVal === predVal;
    let matchText = match ? 'CORRECT' : 'INCORRECT';
    let matchColor = match ? '#10b981' : '#ef4444';
    $('#insp-ss-match').text(matchText).css('color', matchColor);

    // Contact counts for this residue
    let nativeContacts = state.proteinData.true_contacts[index].reduce((a, b) => a + b, 0);
    let predContacts = state.proteinData.pred_contacts[index].filter(p => p >= state.threshold).length;

    $('#insp-true-contacts').text(nativeContacts);
    $('#insp-pred-contacts').text(predContacts);
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
