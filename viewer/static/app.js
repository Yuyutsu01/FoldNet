document.addEventListener("DOMContentLoaded", () => {
    
    // --- UTILS ---
    const classToChar = {0: 'H', 1: 'E', 2: 'C'};
    
    function showToast(message, type="success") {
        const toastEl = document.getElementById("action-toast");
        const toastBody = document.getElementById("action-toast-body");
        
        toastEl.className = `toast align-items-center text-bg-${type} border-0`;
        toastBody.textContent = message;
        
        const toast = new bootstrap.Toast(toastEl, {delay: 3000});
        toast.show();
    }
    
    function drawSSBar(elementId, ssArray) {
        const container = document.getElementById(elementId);
        container.innerHTML = "";
        
        let currentClass = -1;
        let startIdx = 0;
        
        for (let i = 0; i <= ssArray.length; i++) {
            if (i === ssArray.length || ssArray[i] !== currentClass) {
                if (currentClass !== -1) {
                    const widthPct = ((i - startIdx) / ssArray.length) * 100;
                    const char = classToChar[currentClass];
                    
                    const div = document.createElement("div");
                    div.className = `ss-item ss-${char}`;
                    div.style.width = `${widthPct}%`;
                    div.title = `Residues ${startIdx+1}-${i}: ${char}`;
                    
                    // Add tiny border to separate blocks slightly if desired
                    div.style.borderRight = "1px solid rgba(255,255,255,0.2)";
                    
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
            colorbar: { thickness: 15, len: 0.8 }
        }];
        
        if (zmin !== null) data[0].zmin = zmin;
        if (zmax !== null) data[0].zmax = zmax;
        
        const layout = {
            margin: { t: 10, r: 10, b: 20, l: 30 },
            xaxis: { visible: true, showgrid: false },
            yaxis: { visible: true, autorange: 'reversed', showgrid: false },
            plot_bgcolor: "#f8f9fa",
            paper_bgcolor: "transparent"
        };
        
        Plotly.newPlot(elementId, data, layout, {responsive: true, displayModeBar: false});
    }

    // --- PREDICTOR TAB ---
    window.loadDemo = function(seqName) {
        const demos = {
            '1akeA': 'MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG',
            '2rn2': 'MVNPTVFFDIAVDGEPLGRVSFELFADKVPKTAENFRALSTGEKGFGYKGSCFHRIIPGFMCQGGDFTRHNGTGGKSIYGEKFEDENFILKHTGPGILSMANAGPNTNGSQFFICTAKTEWLDGKHVVFGKVKEGMNIVEAMERFGSRNGKTSKKITIADCGQLE'
        };
        document.getElementById("seq-input").value = demos[seqName];
        document.getElementById("seq-warning").classList.add("d-none");
    };

    document.getElementById("btn-predict").addEventListener("click", async () => {
        const seq = document.getElementById("seq-input").value.toUpperCase().trim();
        const warning = document.getElementById("seq-warning");
        
        // Validation
        const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/;
        if (seq.length < 30 || !validAminoAcids.test(seq)) {
            warning.classList.remove("d-none");
            return;
        }
        warning.classList.add("d-none");
        
        const btn = document.getElementById("btn-predict");
        const loadingDiv = document.getElementById("predict-loading");
        const resDiv = document.getElementById("predict-results");
        const emptyDiv = document.getElementById("predict-empty");
        
        btn.disabled = true;
        emptyDiv.classList.add("d-none");
        resDiv.classList.add("d-none");
        loadingDiv.classList.remove("d-none");
        
        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sequence: seq })
            });
            
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            
            // Draw SS
            drawSSBar("predict-ss-bar", data.pred_ss);
            document.getElementById("predict-ss-text").textContent = data.pred_ss.map(c => classToChar[c]).join("");
            
            // Draw Contacts
            drawHeatmap("predict-contact-map", data.pred_contacts, "", "Viridis", 0, 1);
            
            loadingDiv.classList.add("d-none");
            resDiv.classList.remove("d-none");
            showToast("Prediction completed successfully!");
            
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
            
            // Sort by ID
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
                    document.getElementById("prot-len-badge").textContent = "-";
                    return;
                }
                
                document.getElementById("testset-empty").classList.add("d-none");
                
                // Show loading indicator in badge
                document.getElementById("prot-len-badge").innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
                
                try {
                    const pres = await fetch(`/api/test_protein/${pid}`);
                    const pdata = await pres.json();
                    
                    document.getElementById("prot-len-badge").textContent = `${pdata.length} Residues`;
                    
                    // Metrics
                    document.getElementById("ts-metric-q3").textContent = pdata.metrics.Q3.toFixed(1) + "%";
                    document.getElementById("ts-metric-mcc").textContent = pdata.metrics.MCC.toFixed(3);
                    document.getElementById("ts-metric-pl").textContent = pdata.metrics["Precision@L"].toFixed(3);
                    document.getElementById("ts-metric-lr").textContent = pdata.metrics.LongRangePrecision.toFixed(3);
                    
                    // Color code metrics based on generic thresholds
                    const q3El = document.getElementById("ts-metric-q3").parentElement;
                    q3El.style.borderLeftColor = pdata.metrics.Q3 > 82 ? "#2ecc71" : (pdata.metrics.Q3 > 78 ? "#f39c12" : "#e74c3c");
                    
                    const plEl = document.getElementById("ts-metric-pl").parentElement;
                    plEl.style.borderLeftColor = pdata.metrics["Precision@L"] > 0.05 ? "#2ecc71" : "#f39c12";

                    // Draw SS
                    drawSSBar("ts-ss-true", pdata.true_ss);
                    drawSSBar("ts-ss-pred", pdata.pred_ss);
                    
                    // Calculate Difference Map (True - Pred)
                    // True is 1/0, Pred is 0 to 1
                    const diffMap = [];
                    for(let i=0; i<pdata.length; i++) {
                        const row = [];
                        for(let j=0; j<pdata.length; j++) {
                            row.push(pdata.true_contacts[i][j] - pdata.pred_contacts[i][j]);
                        }
                        diffMap.push(row);
                    }
                    
                    // Draw Contacts
                    drawHeatmap("ts-contact-true", pdata.true_contacts, "", "Greys", 0, 1);
                    drawHeatmap("ts-contact-pred", pdata.pred_contacts, "", "Viridis", 0, 1);
                    drawHeatmap("ts-contact-diff", diffMap, "", "RdBu", -1, 1); // Divergent red to blue
                    
                    document.getElementById("testset-results").classList.remove("d-none");
                } catch(err) {
                    alert("Error loading protein data.");
                }
            });
            
        } catch(e) {
            console.error("Could not load test proteins", e);
        }
    }
    
    // --- BENCHMARK TAB ---
    window.exportTableToCSV = function() {
        const table = document.getElementById("benchmark-table");
        let csv = [];
        for (let i = 0; i < table.rows.length; i++) {
            let row = [], cols = table.rows[i].querySelectorAll("td, th");
            for (let j = 0; j < cols.length; j++) {
                row.push('"' + cols[j].innerText.trim() + '"');
            }
            csv.push(row.join(","));
        }
        
        const blob = new Blob([csv.join("\n")], { type: "text/csv" });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.setAttribute("hidden", "");
        a.setAttribute("href", url);
        a.setAttribute("download", "foldnet_benchmarks.csv");
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        showToast("CSV Downloaded!", "primary");
    };

    function drawBenchmarkChart() {
        const trace1 = {
            x: ['CNN (Residual)', 'BiLSTM', 'Transformer'],
            y: [83.66, 82.55, 82.93],
            name: 'Q3 Accuracy (%)',
            type: 'bar',
            marker: { color: '#2c7da0' }
        };
        const trace2 = {
            x: ['CNN (Residual)', 'BiLSTM', 'Transformer'],
            y: [74.48, 72.70, 73.26], // scaled up * 100
            name: 'MCC * 100',
            type: 'bar',
            marker: { color: '#a8dadc' }
        };
        const trace3 = {
            x: ['CNN (Residual)', 'BiLSTM', 'Transformer'],
            y: [2.58, 10.22, 2.26], // scaled up * 100
            name: 'Precision@L * 100',
            type: 'bar',
            marker: { color: '#e63946' }
        };
        
        const data = [trace1, trace2, trace3];
        const layout = { 
            barmode: 'group', 
            margin: {t:10, b:40},
            plot_bgcolor: "transparent",
            paper_bgcolor: "transparent",
            legend: { orientation: "h", y: -0.2 }
        };
        Plotly.newPlot('benchmark-chart', data, layout, {responsive: true, displayModeBar: false});
    }

    // Initialize
    loadTestProteins();
    drawBenchmarkChart();
});
