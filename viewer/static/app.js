document.addEventListener("DOMContentLoaded", () => {
    
    // --- DESIGN CONSTANTS ---
    const ACCENT_PRIMARY = '#38bdf8';
    const ACCENT_SECONDARY = '#818cf8';
    const TEXT_MUTED = '#94a3b8';
    const BG_NAVY = '#020617';

    // --- UTILS ---
    const classToChar = {0: 'H', 1: 'E', 2: 'C'};
    
    function showToast(message, type="success") {
        const toastEl = document.getElementById("action-toast");
        const toastBody = document.getElementById("action-toast-body");
        if (!toastEl) return;
        toastEl.className = `toast align-items-center text-bg-${type} border-0 glass-panel shadow-lg`;
        toastBody.textContent = message;
        const toast = new bootstrap.Toast(toastEl, {delay: 3000});
        toast.show();
    }
    
    function drawSSBar(elementId, ssArray) {
        const container = document.getElementById(elementId);
        container.innerHTML = "";
        let currentClass = -1, startIdx = 0;
        for (let i = 0; i <= ssArray.length; i++) {
            if (i === ssArray.length || ssArray[i] !== currentClass) {
                if (currentClass !== -1) {
                    const widthPct = ((i - startIdx) / ssArray.length) * 100;
                    const char = classToChar[currentClass];
                    const div = document.createElement("div");
                    div.className = `ss-item ss-${char}`;
                    div.style.width = `${widthPct}%`;
                    div.title = `Residues ${startIdx+1}-${i}: ${char}`;
                    div.style.borderRight = "1px solid rgba(255,255,255,0.1)";
                    container.appendChild(div);
                }
                currentClass = ssArray[i];
                startIdx = i;
            }
        }
    }
    
    function drawHeatmap(elementId, zMatrix, title, colorscale, zmin=null, zmax=null) {
        const data = [{
            z: zMatrix,
            type: 'heatmap',
            colorscale: colorscale,
            showscale: true,
            hoverongaps: false,
            colorbar: { 
                thickness: 8, 
                len: 0.8,
                outlinewidth: 0,
                tickfont: { color: TEXT_MUTED, family: 'Outfit', size: 9 }
            }
        }];
        if (zmin !== null) data[0].zmin = zmin;
        if (zmax !== null) data[0].zmax = zmax;

        const layout = {
            margin: { t: 5, r: 5, b: 25, l: 30 },
            xaxis: { 
                visible: true, 
                showgrid: false, 
                zeroline: false,
                tickfont: { color: TEXT_MUTED, family: 'Outfit', size: 9 },
                title: { text: "Residue Index", font: { size: 9, color: TEXT_MUTED } }
            },
            yaxis: { 
                visible: true, 
                autorange: 'reversed', 
                showgrid: false, 
                zeroline: false,
                tickfont: { color: TEXT_MUTED, family: 'Outfit', size: 9 },
                title: { text: "Residue Index", font: { size: 9, color: TEXT_MUTED } }
            },
            plot_bgcolor: "transparent",
            paper_bgcolor: "transparent",
            font: { family: 'Inter', color: '#f8fafc' }
        };

        const config = {
            responsive: true,
            displayModeBar: false,
            autosizable: true,
            scrollZoom: false
        };

        Plotly.newPlot(elementId, data, layout, config);
    }

    // --- REAL-TIME VALIDATION ---
    const seqInput = document.getElementById("seq-input");
    const valIndicator = document.getElementById("val-indicator");
    const valText = document.getElementById("val-text");
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;

    seqInput.addEventListener("input", () => {
        const seq = seqInput.value.trim();
        if (seq.length === 0) {
            valIndicator.className = "small mb-4 d-flex align-items-center gap-2 text-muted";
            valText.textContent = "Min 30 residues";
            return;
        }
        const isValidChars = validAminoAcids.test(seq);
        const isLongEnough = seq.length >= 30;

        if (isValidChars && isLongEnough) {
            valIndicator.className = "small mb-4 d-flex align-items-center gap-2 text-info fw-bold";
            valText.textContent = `Sequence Verified (${seq.length} residues)`;
        } else {
            valIndicator.className = "small mb-4 d-flex align-items-center gap-2 text-rose fw-bold";
            valText.textContent = !isValidChars ? "Invalid Residues Found" : `Length: ${seq.length}/30`;
        }
    });

    // --- PREDICTOR TAB ---
    let lastPredictionData = null;
    window.loadDemo = function(seqName) {
        const demos = {
            '1akeA': 'MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG',
            '2rn2': 'MVNPTVFFDIAVDGEPLGRVSFELFADKVPKTAENFRALSTGEKGFGYKGSCFHRIIPGFMCQGGDFTRHNGTGGKSIYGEKFEDENFILKHTGPGILSMANAGPNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERFGSRNGKTSKKITIADCGQLE'
        };
        seqInput.value = demos[seqName];
        seqInput.dispatchEvent(new Event('input'));
    };

    const colorPicker = document.getElementById("map-color-picker");
    colorPicker.addEventListener("change", () => {
        if (lastPredictionData) {
            drawHeatmap("predict-contact-map", lastPredictionData.pred_contacts, "", colorPicker.value, 0, 1);
        }
    });

    document.getElementById("btn-predict").addEventListener("click", async () => {
        const seq = seqInput.value.toUpperCase().trim();
        if (seq.length < 30 || !validAminoAcids.test(seq)) {
            showToast("Please enter a valid protein sequence (min 30 residues).", "danger");
            return;
        }
        
        const btn = document.getElementById("btn-predict");
        const loadingDiv = document.getElementById("predict-loading");
        const resDiv = document.getElementById("predict-results");
        const emptyDiv = document.getElementById("predict-empty");
        const progressBar = document.getElementById("predict-progress");
        const statusDesc = document.getElementById("predict-status-desc");
        const statusTitle = document.getElementById("predict-status-title");
        
        btn.disabled = true;
        emptyDiv.classList.add("d-none");
        resDiv.classList.add("d-none");
        loadingDiv.classList.remove("d-none");
        loadingDiv.style.opacity = '0';
        setTimeout(() => loadingDiv.style.opacity = '1', 10);
        
        // Progress Simulation
        progressBar.style.width = "10%";
        statusTitle.textContent = "Initializing Neural Pipeline...";
        statusDesc.textContent = "Step 1: Preparing ESM-2 Backbone";
        
        try {
            const fetchPromise = fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sequence: seq })
            });

            setTimeout(() => { 
                if (btn.disabled) { 
                    progressBar.style.width = "40%"; 
                    statusTitle.textContent = "ESM-2 Encoding...";
                    statusDesc.textContent = "Step 2: Generating 1280-dim Language Representations"; 
                } 
            }, 1500);
            
            setTimeout(() => { 
                if (btn.disabled) { 
                    progressBar.style.width = "75%"; 
                    statusTitle.textContent = "FoldNet Inference...";
                    statusDesc.textContent = "Step 3: BiLSTM Encoder & Dual-Head Decoding"; 
                } 
            }, 4000);

            const res = await fetchPromise;
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            lastPredictionData = data;
            
            progressBar.style.width = "100%";
            statusTitle.textContent = "Synthesis Complete";
            statusDesc.textContent = "Finalizing Visual Layers...";

            setTimeout(() => {
                drawSSBar("predict-ss-bar", data.pred_ss);
                document.getElementById("predict-ss-text").textContent = data.pred_ss.map(c => classToChar[c]).join("");
                drawHeatmap("predict-contact-map", data.pred_contacts, "", colorPicker.value, 0, 1);
                
                document.getElementById("pred-metric-len").textContent = data.length;
                document.getElementById("pred-metric-status").textContent = data.is_test_set ? data.protein_id : "User Sequence";
                
                loadingDiv.classList.add("d-none");
                resDiv.classList.remove("d-none");
                resDiv.style.opacity = '0';
                setTimeout(() => resDiv.style.opacity = '1', 10);
                showToast("Prediction synthesized successfully!");
            }, 800);
            
        } catch (err) {
            showToast("Inference Error: " + err.message, "danger");
            loadingDiv.classList.add("d-none");
            emptyDiv.classList.remove("d-none");
        } finally {
            btn.disabled = false;
        }
    });

    // --- TEST SET TAB ---
    async function loadTestProteins() {
        try {
            const res = await fetch("/api/test_proteins");
            if (!res.ok) return;
            const metadata = await res.json();
            const select = document.getElementById("protein-select");
            select.innerHTML = '<option value="">Select Target...</option>';
            metadata.sort((a,b) => a.protein_id.localeCompare(b.protein_id)).forEach(m => {
                const opt = document.createElement("option");
                opt.value = m.protein_id;
                opt.textContent = `${m.protein_id} [Q3: ${(m.Q3).toFixed(1)}%]`;
                select.appendChild(opt);
            });
            
            select.addEventListener("change", async (e) => {
                const pid = e.target.value;
                if (!pid) {
                    document.getElementById("testset-results").classList.add("d-none");
                    document.getElementById("testset-empty").classList.remove("d-none");
                    return;
                }
                document.getElementById("testset-empty").classList.add("d-none");
                document.getElementById("prot-len-badge").innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i>';
                
                try {
                    const pres = await fetch(`/api/test_protein/${pid}`);
                    const pdata = await pres.json();
                    
                    document.getElementById("prot-len-badge").textContent = pdata.length;
                    document.getElementById("ts-metric-q3").textContent = pdata.metrics.Q3.toFixed(1) + "%";
                    document.getElementById("ts-metric-mcc").textContent = pdata.metrics.MCC.toFixed(3);
                    document.getElementById("ts-metric-pl").textContent = pdata.metrics["Precision@L"].toFixed(3);
                    document.getElementById("ts-metric-lr").textContent = pdata.metrics.LongRangePrecision.toFixed(3);
                    
                    drawSSBar("ts-ss-true", pdata.true_ss);
                    drawSSBar("ts-ss-pred", pdata.pred_ss);
                    
                    const diffMap = [];
                    for(let i=0; i<pdata.length; i++) {
                        const row = [];
                        for(let j=0; j<pdata.length; j++) {
                            row.push(pdata.true_contacts[i][j] - pdata.pred_contacts[i][j]);
                        }
                        diffMap.push(row);
                    }
                    
                    drawHeatmap("ts-contact-true", pdata.true_contacts, "", "Greys", 0, 1);
                    drawHeatmap("ts-contact-pred", pdata.pred_contacts, "", "Blues", 0, 1);
                    drawHeatmap("ts-contact-diff", diffMap, "", "RdBu", -1, 1);
                    
                    const resArea = document.getElementById("testset-results");
                    resArea.classList.remove("d-none");
                    resArea.style.opacity = '0';
                    setTimeout(() => {
                        resArea.style.opacity = '1';
                        // Force resize to fix squashed alignment in 3-column layout
                        ['ts-contact-true', 'ts-contact-pred', 'ts-contact-diff'].forEach(id => {
                            const el = document.getElementById(id);
                            if (el) Plotly.Plots.resize(el);
                        });
                    }, 50);
                } catch(err) {
                    showToast("Error loading protein archives.", "danger");
                }
            });
        } catch(e) { console.error("Could not load test proteins", e); }
    }
    
    // --- BENCHMARK TAB ---
    function drawBenchmarkChart() {
        const trace1 = {
            x: ['CNN', 'BiLSTM', 'Transformer'],
            y: [83.66, 82.55, 82.93],
            name: 'Q3 Accuracy (%)', 
            type: 'bar', 
            marker: { color: ACCENT_PRIMARY, line: { width: 0 } },
            hovertemplate: '%{y}%'
        };
        const trace2 = {
            x: ['CNN', 'BiLSTM', 'Transformer'],
            y: [74.48, 72.70, 73.26],
            name: 'MCC (x100)', 
            type: 'bar', 
            marker: { color: ACCENT_SECONDARY, line: { width: 0 } },
            hovertemplate: '%{y}'
        };
        const data = [trace1, trace2];
        const layout = { 
            barmode: 'group', 
            margin: {t:20, b:60, l:40, r:20},
            plot_bgcolor: "transparent", 
            paper_bgcolor: "transparent",
            legend: { 
                orientation: "h", 
                y: -0.3, 
                font: { color: TEXT_MUTED, family: 'Outfit' } 
            },
            xaxis: { tickfont: { color: TEXT_MUTED, family: 'Outfit' }, showgrid: false },
            yaxis: { tickfont: { color: TEXT_MUTED, family: 'Outfit' }, gridcolor: 'rgba(255,255,255,0.05)' },
            font: { family: 'Outfit', color: '#f8fafc' }
        };
        Plotly.newPlot('benchmark-chart', data, layout, {responsive: true, displayModeBar: false});
    }

    // --- TAB SWITCH ANIMATION ---
    const tabBtns = document.querySelectorAll('.nav-link');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const panes = document.querySelectorAll('.tab-pane');
            panes.forEach(p => {
                p.style.opacity = '0';
                p.style.transform = 'translateY(10px)';
                setTimeout(() => {
                    p.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
                    p.style.opacity = '1';
                    p.style.transform = 'translateY(0)';
                }, 50);
            });
            // Recalculate Plotly sizes
            setTimeout(() => {
                const charts = ['predict-contact-map', 'ts-contact-true', 'ts-contact-pred', 'ts-contact-diff', 'benchmark-chart'];
                charts.forEach(c => {
                    const el = document.getElementById(c);
                    if (el && el.offsetHeight > 0) Plotly.Plots.resize(el);
                });
            }, 200);
        });
    });

    loadTestProteins();
    drawBenchmarkChart();
    
    // Smooth background parallax
    window.addEventListener('mousemove', (e) => {
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        document.body.style.backgroundPosition = `${x * 20}px ${y * 20}px`;
    });
});
