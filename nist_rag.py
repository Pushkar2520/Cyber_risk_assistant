"""
nist_rag.py — RAG Pipeline for NIST SP 800-53 Rev. 5 Control Retrieval

Embeds NIST SP 800-53 control descriptions into ChromaDB using sentence-transformers.
For each identified risk, retrieves the most relevant NIST control via semantic search.

This is the RAG component: the NIST controls are unstructured prose text that
benefits from embedding-based retrieval rather than keyword matching.
"""

import os
import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "nist_sp800_53_rev5"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    """Load the sentence-transformer model (cached across reruns)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def get_chroma_client():
    """Get or create a persistent ChromaDB client. Handles singleton conflicts."""
    try:
        return chromadb.PersistentClient(path=CHROMA_DIR)
    except KeyError:
        # SharedSystemClient singleton conflict — reset and retry
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient._identifier_to_system.clear()
        return chromadb.PersistentClient(path=CHROMA_DIR)


def build_nist_index(nist_controls):
    """
    Embed and index NIST SP 800-53 controls into ChromaDB.

    Each control becomes a document with:
      - id: control_id (e.g., "SI-2")
      - document: full text (title + description + discussion)
      - metadata: control_id, title
    """
    if not nist_controls:
        print("WARNING: No NIST controls to index.")
        return None

    model = get_embedding_model()
    client = get_chroma_client()

    # Check if collection already exists and is populated
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        if collection.count() > 0:
            print(f"NIST index already exists with {collection.count()} controls. Skipping rebuild.")
            return collection
    except Exception:
        pass

    # Create or recreate collection
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "NIST SP 800-53 Rev. 5 Security Controls"},
    )

    # Prepare documents
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for ctrl in nist_controls:
        ctrl_id = ctrl["control_id"]
        full_text = ctrl.get("full_text", "")
        if not full_text.strip():
            continue

        # Use a hash-based ID to avoid duplicates
        doc_id = hashlib.md5(ctrl_id.encode()).hexdigest()

        ids.append(doc_id)
        documents.append(full_text)
        metadatas.append({
            "control_id": ctrl_id,
            "title": ctrl.get("title", ""),
        })

    # Batch embed
    print(f"Embedding {len(documents)} NIST controls...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=64).tolist()

    # Add to collection in batches (ChromaDB has a limit)
    BATCH_SIZE = 500
    for i in range(0, len(ids), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(ids))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            embeddings=embeddings[i:batch_end],
        )

    print(f"Indexed {collection.count()} NIST controls into ChromaDB.")
    return collection


def query_nist_control(risk_context, collection=None, top_k=3):
    """
    Given a risk context string, retrieve the most relevant NIST SP 800-53 controls.

    Args:
        risk_context: A string describing the risk (vulnerability + asset context)
        collection: ChromaDB collection (if None, will get from client)
        top_k: Number of controls to retrieve

    Returns:
        List of dicts with: control_id, title, description, relevance_score
    """
    model = get_embedding_model()

    if collection is None:
        client = get_chroma_client()
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            print("WARNING: NIST index not found. Run build_nist_index first.")
            return []

    if collection.count() == 0:
        return []

    # Embed the query
    query_embedding = model.encode([risk_context]).tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    controls = []
    if results and results["ids"] and results["ids"][0]:
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

    This constructs a query that captures the essence of the risk so that
    the embedding search finds the most relevant NIST control.
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

    # What kind of remediation is needed
    affected = risk_row.get("affected_component", "")
    if affected:
        parts.append(f"Affected component: {affected}")

    # Add context about the type of security control needed
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
