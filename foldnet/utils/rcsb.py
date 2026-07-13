import requests

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

def search_rcsb_sequence(sequence: str, identity: float = 0.95, rows: int = 3) -> list:
    """
    Query RCSB PDB for chains matching sequence at given identity.
    Returns list of dicts: [{'pdb_id': '1ABC', 'chain_id': 'A'}, ...]
    """
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1,
                "identity_cutoff": identity,
                "sequence_type": "protein",
                "value": sequence
            }
        },
        "return_type": "polymer_instance",
        "request_options": {
            "results_verbosity": "minimal",
            "results_content_type": ["experimental"],
            "paginate": {"start": 0, "rows": rows}
        }
    }
    try:
        r = requests.post(RCSB_SEARCH_URL, json=query, timeout=20)
        if r.status_code != 200:
            return []
        results = r.json().get("result_set", [])
        hits = []
        for item in results:
            parts = item.get("identifier", "").split(".")
            if len(parts) == 2:
                hits.append({"pdb_id": parts[0].upper(), "chain_id": parts[1]})
        return hits
    except Exception:
        return []
