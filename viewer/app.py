import os
import sys
import json
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

# Add parent directory to path to import foldnet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from foldnet.utils.predict import load_model

app = FastAPI(title="FoldNet Dashboard API")

# Global variables to hold models
foldnet_model = None
esm_model = None
esm_batch_converter = None
startup_error = None

# Paths
DASHBOARD_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dashboard'))
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

# Mount static files
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.on_event("startup")
async def startup_event():
    """Load the model on startup to keep the API fast."""
    global foldnet_model, esm_model, esm_batch_converter, startup_error
    try:
        # Load the best FoldNet model
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', 'foldnet-epoch=08-val_loss=0.5015.ckpt')
        if os.path.exists(ckpt_path):
            foldnet_model = load_model(ckpt_path)
            foldnet_model.eval()
            if torch.cuda.is_available():
                foldnet_model = foldnet_model.cuda()
            print("FoldNet model loaded successfully.")
        else:
            print(f"Warning: Model checkpoint not found at {ckpt_path}")
            startup_error = "Checkpoint not found."
            
        print("Loading ESM-2 model for live predictions...")
        import esm
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm_batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
        # Keep ESM on CPU to prevent Out-Of-Memory errors on 6GB GPUs alongside FoldNet
        print("ESM-2 loaded on CPU.")
    except Exception as e:
        import traceback
        startup_error = str(e) + "\n" + traceback.format_exc()
        print(f"Error during startup: {startup_error}")

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/test_proteins")
async def get_test_proteins():
    """Returns the list of proteins and their metrics for the Test Set Visualisation."""
    metadata_path = os.path.join(DASHBOARD_DATA_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Dashboard data not generated yet.")
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    return data

@app.get("/api/test_protein/{protein_id}")
async def get_test_protein(protein_id: str):
    """Returns full data (SS and Contact Maps) for a specific test protein."""
    prot_path = os.path.join(DASHBOARD_DATA_DIR, "proteins", f"{protein_id}.json")
    if not os.path.exists(prot_path):
        raise HTTPException(status_code=404, detail="Protein data not found.")
        
    with open(prot_path, 'r') as f:
        data = json.load(f)
    return data

class PredictRequest(BaseModel):
    sequence: str

@app.post("/api/predict")
async def predict_sequence(req: PredictRequest):
    """Live inference for a new protein sequence."""
    if foldnet_model is None or esm_model is None:
        error_msg = startup_error if startup_error else "Models failed to load for an unknown reason."
        raise HTTPException(status_code=503, detail=f"Models are not loaded. Error: {error_msg}")
        
    seq = req.sequence.upper().strip()
    L = len(seq)
    if L == 0 or L > 1000:
        raise HTTPException(status_code=400, detail="Invalid sequence length (1-1000).")
        
    data = [("protein1", seq)]
    batch_labels, batch_strs, batch_tokens = esm_batch_converter(data)
    
    device = next(foldnet_model.parameters()).device
    batch_tokens = batch_tokens.to(device)
    
    with torch.no_grad():
        # Get ESM representations on CPU
        results = esm_model(batch_tokens.cpu(), repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        
        # Remove BOS and EOS tokens to get shape (1, L, 1280)
        features = token_representations[:, 1:L+1, :]
        features = features.to(device) # Move features to GPU for FoldNet
        
        # Pass to FoldNet
        ss_logits, contact_probs = foldnet_model(features)
        
        ss_pred = torch.argmax(ss_logits[0], dim=-1).cpu().numpy()
        c_probs = torch.sigmoid(contact_probs[0]).cpu().numpy()
        
    return {
        "sequence": seq,
        "length": L,
        "pred_ss": ss_pred.tolist(),
        "pred_contacts": np.round(c_probs, 3).tolist()
    }
