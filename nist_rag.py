"""
nist_rag.py — RAG Pipeline for NIST SP 800-53 Rev. 5 Control Retrieval

Embeds NIST SP 800-53 control descriptions into ChromaDB using sentence-transformers
via chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction.
For each identified risk, retrieves the most relevant NIST control via semantic search.
"""

import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

COLLECTION_NAME = "nist_sp800_53_rev5"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Initializing SentenceTransformer embedding function...")
def get_embedding_function():
    """
    Get the embedding function using sentence-transformers.
    Cached across reruns so the model is loaded only once into memory.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """
    Get or create an in-memory ChromaDB client.
    Cached across reruns to preserve state without requiring filesystem persistence.
    """
    return chromadb.EphemeralClient()


@st.cache_resource(show_spinner="Embedding NIST controls into vector store...")
def build_nist_index(nist_controls):
    """
    Embed and index NIST SP 800-53 controls into ChromaDB.
    Cached with @st.cache_resource so 1,196 controls are indexed once and reused
    across Streamlit script reruns.
    """
    if not nist_controls:
        print("WARNING: No NIST controls to index.")
        return None

    ef = get_embedding_function()
    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "NIST SP 800-53 Rev. 5 Security Controls"},
    )

    # If already populated, reuse existing collection
    if collection.count() > 0:
        print(f"NIST index already populated with {collection.count()} controls.")
        return collection

    # Prepare documents
    ids = []
    documents = []
    metadatas = []

    for ctrl in nist_controls:
        ctrl_id = ctrl["control_id"]
        full_text = ctrl.get("full_text", "")
        if not full_text.strip():
            continue

        # Use hash-based ID to ensure deterministic unique IDs
        doc_id = hashlib.md5(ctrl_id.encode()).hexdigest()

        ids.append(doc_id)
        documents.append(full_text)
        metadatas.append({
            "control_id": ctrl_id,
            "title": ctrl.get("title", ""),
        })

    # Batch add to collection
    print(f"Embedding {len(documents)} NIST controls with {EMBEDDING_MODEL_NAME}...")
    BATCH_SIZE = 100
    for i in range(0, len(ids), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )

    print(f"Indexed {collection.count()} NIST controls into ChromaDB.")
    return collection


def query_nist_control(risk_context, collection=None, top_k=3):
    """
    Given a risk context string, retrieve the most relevant NIST SP 800-53 controls.

    Args:
        risk_context: A string describing the risk (vulnerability + asset context)
        collection: ChromaDB collection (if None, fetched or created from client)
        top_k: Number of controls to retrieve

    Returns:
        List of dicts with: control_id, title, full_text, distance
    """
    if collection is None:
        ef = get_embedding_function()
        client = get_chroma_client()
        try:
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
            )
        except Exception as e:
            print(f"WARNING: Could not access collection: {e}")
            return []

    if collection.count() == 0:
        return []

    # Query ChromaDB (embedding function automatically encodes query_texts)
    results = collection.query(
        query_texts=[risk_context],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    controls = []
    if results and results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            controls.append({
                "control_id": results["metadatas"][0][i].get("control_id", ""),
                "title": results["metadatas"][0][i].get("title", ""),
                "full_text": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

    return controls


def build_risk_query(risk_row):
    """
    Build a semantic query string from a risk row for NIST control retrieval.
    """
    parts = []

    # Vulnerability context
    vuln_name = risk_row.get("vulnerability_name", "")
    cve = risk_row.get("cve", "")
    if vuln_name:
        parts.append(f"Vulnerability: {vuln_name}")
    if cve:
        parts.append(f"CVE: {cve}")

    # Asset context
    asset_name = risk_row.get("asset_name", "")
    asset_type = risk_row.get("asset_type", "")
    if asset_name:
        parts.append(f"Asset: {asset_name} ({asset_type})")

    # Affected component
    affected = risk_row.get("affected_component", "")
    if affected:
        parts.append(f"Affected component: {affected}")

    # Access / network context
    exposure = str(risk_row.get("asset_exposure", risk_row.get("internet_exposed", ""))).strip().lower()
    if exposure in ("internet", "yes"):
        parts.append("Internet-facing system requiring access control and monitoring")

    patch_available = str(risk_row.get("patch_available", "")).strip().lower()
    if patch_available == "yes":
        parts.append("Patch available — flaw remediation and vulnerability scanning needed")
    else:
        parts.append("No patch available — compensating controls and system replacement needed")

    edr = str(risk_row.get("edr_installed", "")).strip().lower()
    if edr in ("no", "false"):
        parts.append("Missing endpoint detection and response — security monitoring gap")

    # Threat context
    ti_actor = risk_row.get("ti_threat_actor", "")
    if ti_actor:
        parts.append(f"Active threat actor: {ti_actor}")
        parts.append("Incident handling and threat response needed")

    ransomware = str(risk_row.get("ti_ransomware", "")).strip().lower() == "yes"
    if ransomware:
        parts.append("Ransomware risk — backup integrity and incident response critical")

    return " | ".join(parts)
