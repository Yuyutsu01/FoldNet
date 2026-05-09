document.addEventListener("DOMContentLoaded", () => {
    
    // --- THEME ENGINE ---
    const themeToggle = document.getElementById("theme-toggle");
    const body = document.body;
    
    function setTheme(theme) {
        body.setAttribute("data-theme", theme);
        localStorage.setItem("foldnet-theme", theme);
        const icon = themeToggle.querySelector("i");
        const text = themeToggle.querySelector("span");
        if (theme === "dark") {
            icon.className = "fa-solid fa-sun me-2";
            text.textContent = "Light Mode";
        } else {
            icon.className = "fa-solid fa-moon me-2";
            text.textContent = "Dark Mode";
        }
        // Redraw charts to match theme
        if (document.getElementById("benchmark-chart")) drawBenchmarkChart();
    }

    const savedTheme = localStorage.getItem("foldnet-theme") || "light";
    setTheme(savedTheme);

    themeToggle.addEventListener("click", () => {
        const newTheme = body.getAttribute("data-theme") === "light" ? "dark" : "light";
        setTheme(newTheme);
    });

    // --- UTILS ---
    const classToChar = {0: 'H', 1: 'E', 2: 'C'};
    
    function showToast(message, type="success") {
        const toastEl = document.getElementById("action-toast");
        const toastBody = document.getElementById("action-toast-body");
        if (!toastEl) return;
        toastEl.className = `toast align-items-center text-bg-${type} border-0`;
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
                    div.style.borderRight = "1px solid rgba(255,255,255,0.2)";
                    container.appendChild(div);
                }
                currentClass = ssArray[i];
                startIdx = i;
            }
        }
    }
    
    function drawHeatmap(elementId, zMatrix, title, colorscale, zmin=null, zmax=null) {
        const isDark = body.getAttribute("data-theme") === "dark";
        const data = [{
            z: zMatrix,
            type: 'heatmap',
            colorscale: colorscale,
            showscale: true,
            colorbar: { 
                thickness: 15, len: 0.8, 
                tickfont: { color: isDark ? "#94a3b8" : "#666" }
            }
        }];
        if (zmin !== null) data[0].zmin = zmin;
        if (zmax !== null) data[0].zmax = zmax;
        const layout = {
            margin: { t: 10, r: 10, b: 20, l: 30 },
            xaxis: { visible: true, showgrid: false, tickfont: { color: isDark ? "#94a3b8" : "#666" } },
            yaxis: { visible: true, autorange: 'reversed', showgrid: false, tickfont: { color: isDark ? "#94a3b8" : "#666" } },
            plot_bgcolor: "transparent",
            paper_bgcolor: "transparent"
        };
        Plotly.newPlot(elementId, data, layout, {responsive: true, displayModeBar: false});
    }

    // --- REAL-TIME VALIDATION ---
    const seqInput = document.getElementById("seq-input");
    const valIndicator = document.getElementById("val-indicator");
    const valText = document.getElementById("val-text");
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;

    seqInput.addEventListener("input", () => {
        const seq = seqInput.value.trim();
        if (seq.length === 0) {
            valIndicator.className = "small mb-3 text-muted";
            valText.textContent = "Ready for input (Min 30 chars)";
            return;
        }
        const isValidChars = validAminoAcids.test(seq);
        const isLongEnough = seq.length >= 30;

        if (isValidChars && isLongEnough) {
            valIndicator.className = "small mb-3 text-success fw-bold";
            valText.textContent = `Valid Sequence (${seq.length} residues)`;
        } else {
            valIndicator.className = "small mb-3 text-danger";
            valText.textContent = !isValidChars ? "Invalid characters detected!" : `Too short (${seq.length}/30)`;
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
        document.getElementById("seq-warning").classList.add("d-none");
    };

    const colorPicker = document.getElementById("map-color-picker");
    colorPicker.addEventListener("change", () => {
        if (lastPredictionData) {
            drawHeatmap("predict-contact-map", lastPredictionData.pred_contacts, "", colorPicker.value, 0, 1);
        }
    });

    document.getElementById("btn-predict").addEventListener("click", async () => {
        const seq = seqInput.value.toUpperCase().trim();
        const warning = document.getElementById("seq-warning");
        if (seq.length < 30 || !validAminoAcids.test(seq)) {
            warning.classList.remove("d-none");
            return;
        }
        warning.classList.add("d-none");
        
        const btn = document.getElementById("btn-predict");
        const loadingDiv = document.getElementById("predict-loading");
        const resDiv = document.getElementById("predict-results");
        const emptyDiv = document.getElementById("predict-empty");
        const progressBar = document.getElementById("predict-progress");
        const statusDesc = document.getElementById("predict-status-desc");
        
        btn.disabled = true;
        emptyDiv.classList.add("d-none");
        resDiv.classList.add("d-none");
        loadingDiv.classList.remove("d-none");
        
        // Progress Simulation
        progressBar.style.width = "10%";
        statusDesc.textContent = "Step 1: Initializing ESM-2 Model...";
        
        try {
            const fetchPromise = fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sequence: seq })
            });

            setTimeout(() => { 
                if (btn.disabled) { 
                    progressBar.style.width = "40%"; 
                    statusDesc.textContent = "Step 2: Calculating Language Embeddings (1280-dim)..."; 
                } 
            }, 1500);
            
            setTimeout(() => { 
                if (btn.disabled) { 
                    progressBar.style.width = "75%"; 
                    statusDesc.textContent = "Step 3: BiLSTM Sequential Processing..."; 
                } 
            }, 4000);

            const res = await fetchPromise;
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            lastPredictionData = data;
            
            progressBar.style.width = "100%";
            statusDesc.textContent = "Step 4: Rendering Results...";

            setTimeout(() => {
                drawSSBar("predict-ss-bar", data.pred_ss);
                document.getElementById("predict-ss-text").textContent = data.pred_ss.map(c => classToChar[c]).join("");
                drawHeatmap("predict-contact-map", data.pred_contacts, "", colorPicker.value, 0, 1);
                
                const statusEl = document.getElementById("pred-metric-status");
                const lenEl = document.getElementById("pred-metric-len");
                if (lenEl) lenEl.textContent = data.length;
                if (statusEl) {
                    statusEl.textContent = data.is_test_set ? data.protein_id : "New Sequence";
                    statusEl.style.fontSize = data.is_test_set ? "1.2rem" : "1.5rem";
                }
                
                loadingDiv.classList.add("d-none");
                resDiv.classList.remove("d-none");
                resDiv.classList.add("fade-in");
            }, 500);
            
        } catch (err) {
            alert("Error predicting: " + err.message);
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
            select.innerHTML = '<option value="">Select a protein...</option>';
            metadata.sort((a,b) => a.protein_id.localeCompare(b.protein_id)).forEach(m => {
                const opt = document.createElement("option");
                opt.value = m.protein_id;
                opt.textContent = `${m.protein_id} (L=${m.length}) | Q3: ${(m.Q3).toFixed(1)}%`;
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
                document.getElementById("prot-len-badge").innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
                
                try {
                    const pres = await fetch(`/api/test_protein/${pid}`);
                    const pdata = await pres.json();
                    document.getElementById("prot-len-badge").textContent = `${pdata.length} Residues`;
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
                    resArea.classList.add("fade-in");
                } catch(err) {
                    alert("Error loading protein data.");
                }
            });
        } catch(e) { console.error("Could not load test proteins", e); }
    }
    
    // --- BENCHMARK TAB ---
    function drawBenchmarkChart() {
        const isDark = body.getAttribute("data-theme") === "dark";
        const trace1 = {
            x: ['CNN (Residual)', 'BiLSTM', 'Transformer'],
            y: [83.66, 82.55, 82.93],
            name: 'Q3 Accuracy (%)', type: 'bar', marker: { color: '#2c7da0' }
        };
        const trace2 = {
            x: ['CNN (Residual)', 'BiLSTM', 'Transformer'],
            y: [74.48, 72.70, 73.26],
            name: 'MCC * 100', type: 'bar', marker: { color: '#a8dadc' }
        };
        const data = [trace1, trace2];
        const layout = { 
            barmode: 'group', margin: {t:10, b:40},
            plot_bgcolor: "transparent", paper_bgcolor: "transparent",
            legend: { orientation: "h", y: -0.2, font: { color: isDark ? "#94a3b8" : "#666" } },
            xaxis: { tickfont: { color: isDark ? "#94a3b8" : "#666" } },
            yaxis: { tickfont: { color: isDark ? "#94a3b8" : "#666" } }
        };
        Plotly.newPlot('benchmark-chart', data, layout, {responsive: true, displayModeBar: false});
    }

    loadTestProteins();
    drawBenchmarkChart();
});
