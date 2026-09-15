"""
app.py — Streamlit Web Interface for TawasolPay AI Cyber Risk Assistant

Main entry point for the application. Orchestrates:
  1. Data ingestion (local CSVs + external CISA KEV + NIST SP 800-53)
  2. Multi-factor risk scoring
  3. NIST RAG retrieval for remediation guidance
  4. Batched LLM report generation (Executive Summary + 5 Risk Analyses in 1 call)
  5. Interactive dashboard display
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Streamlit page configuration must be first command
st.set_page_config(
    page_title="TawasolPay Cyber Risk Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_loader import load_local_data, fetch_cisa_kev, fetch_nist_controls, build_merged_dataset
from risk_scorer import rank_risks
from nist_rag import build_nist_index, query_nist_control, build_risk_query
from report_generator import (
    configure_gemini,
    generate_risk_narrative,
    generate_batched_report,
    get_active_model_name,
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .risk-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        border-left: 4px solid #ef4444;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .risk-score-badge {
        background: #ef4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Pipeline Stages ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_all_pipeline_data():
    """Load local datasets, CISA KEV, NIST controls, and build merged dataframe."""
    local_data = load_local_data()
    kev = fetch_cisa_kev()
    nist = fetch_nist_controls()
    merged_df = build_merged_dataset(local_data, kev)
    return local_data, kev, nist, merged_df


@st.cache_data(show_spinner=False)
def score_and_rank_risks(merged_df, top_n=5):
    """Compute 7-factor risk scores and return ranked dataframe."""
    return rank_risks(merged_df, top_n=top_n)


@st.cache_data(show_spinner="Generating AI Cyber Risk Assessment...")
def get_cached_report(_top_risks_dict, _nist_results_map, api_key):
    """
    Generate the executive summary and risk analyses in a single batched operation.
    Cached so widget interactions, expander clicks, or tab switching do NOT re-trigger API calls.
    """
    top_risks_df = pd.DataFrame(_top_risks_dict)
    initial_entries = []
    for idx, row in top_risks_df.iterrows():
        rank = int(row.get("risk_rank", idx + 1))
        nist_results = _nist_results_map.get(str(rank), [])
        entry = generate_risk_narrative(row, nist_results, rank)
        initial_entries.append(entry)

    exec_summary, finalized_entries = generate_batched_report(initial_entries, api_key=api_key)
    return exec_summary, finalized_entries


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("🛡️ Cyber Risk Assistant")
    st.caption("TawasolPay — AI-Powered Risk Analysis")

    st.divider()

    # Load API key silently from environment or Streamlit secrets
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key and hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]

    gemini_ready = configure_gemini(api_key)

    if gemini_ready:
        st.success(f"✅ Gemini Connected ({get_active_model_name()})")
    else:
        st.info("ℹ️ Using High-Fidelity Deterministic Engine")

    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("- 📋 Local Enterprise CSVs (assets, vulns, threats)")
    st.markdown("- 🏛️ CISA KEV Catalog (live feed)")
    st.markdown("- 📖 NIST SP 800-53 Rev. 5 (OSCAL vector RAG)")
    st.markdown("- ⚡ Batched Single-Call LLM Synthesis")


# ─── Main Page Header ────────────────────────────────────────────────────────

st.title("🛡️ TawasolPay Cyber Risk Assessment")
st.markdown(
    "**AI-powered risk analysis** combining asset inventory, vulnerability data, "
    "threat intelligence, and NIST SP 800-53 guidance to identify and prioritize "
    "the top cybersecurity risks."
)

st.divider()


# ─── Execution Pipeline ──────────────────────────────────────────────────────

with st.status("Executing cyber risk analysis pipeline...", expanded=False) as status:
    st.write("📂 Ingesting local CSVs, CISA KEV, and NIST SP 800-53 catalog...")
    data, kev_df, nist_controls, merged = load_all_pipeline_data()

    st.write("🧠 Ensuring NIST vector index is mounted in memory...")
    nist_collection = build_nist_index(nist_controls)

    st.write("📊 Computing multi-factor composite risk scores...")
    top_risks = score_and_rank_risks(merged, top_n=5)

    status.update(label="✅ Analysis pipeline complete!", state="complete")


# ─── Data Overview Metrics ────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Assets", len(data["assets"]))
with col2:
    st.metric("Vulnerabilities", len(data["vulnerabilities"]))
with col3:
    st.metric("Threat Intel Records", len(data["threat_intel"]))
with col4:
    kev_matches = merged["in_kev"].sum() if "in_kev" in merged.columns else 0
    st.metric("KEV Matches", int(kev_matches))
with col5:
    st.metric("NIST Controls Indexed", len(nist_controls))

st.divider()


# ─── Retrieve NIST Controls & Run Batched Generation ─────────────────────────

nist_results_map = {}
for idx, row in top_risks.iterrows():
    rank_key = str(int(row.get("risk_rank", idx + 1)))
    query = build_risk_query(row)
    controls = query_nist_control(query, nist_collection, top_k=3)
    nist_results_map[rank_key] = controls

# Execute single cached batched call
top_risks_dict = top_risks.to_dict(orient="records")
exec_summary, risk_entries = get_cached_report(
    top_risks_dict,
    nist_results_map,
    api_key=api_key,
)


# ─── Executive Summary ───────────────────────────────────────────────────────

st.markdown("## 📋 Executive Summary")
st.info(exec_summary)

st.divider()


# ─── Top 5 Prioritized Risks ─────────────────────────────────────────────────

st.markdown("## 🎯 Top 5 Prioritized Risks")
st.caption(
    "Ranked by composite risk score incorporating CVSS, internet exposure, "
    "active exploitation, threat campaign matches, business criticality, and missing controls."
)

for entry in risk_entries:
    rank = entry["rank"]
    score = entry["risk_score"]

    if score >= 80:
        score_color = "🔴"
    elif score >= 60:
        score_color = "🟠"
    else:
        score_color = "🟡"

    st.markdown(f"### {score_color} Risk #{rank} — {entry['vulnerability']}")
    st.markdown(f"**Risk Score: {score}/100**")

    # Factor breakdown
    with st.expander("📊 Risk Factor Breakdown", expanded=False):
        factors = entry["factor_scores"]
        factor_labels = {
            "cvss": ("CVSS Severity", "15%"),
            "exposure": ("Internet Exposure", "20%"),
            "exploit_kev": ("Active Exploit / KEV", "20%"),
            "threat_match": ("Threat Campaign Match", "15%"),
            "ransomware": ("Ransomware Association", "10%"),
            "business_criticality": ("Business Criticality", "10%"),
            "missing_controls": ("Missing Controls", "10%"),
        }
        for key, (label, weight) in factor_labels.items():
            val = factors.get(key, 0)
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(min(val, 1.0), text=f"{label} (weight: {weight})")
            with col_b:
                st.markdown(f"**{val:.0%}**")

    # Content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🖥️ Asset", "🔓 Vulnerability", "🎯 Threat Intel", "💼 Business Impact", "📖 NIST Guidance"
    ])

    with tab1:
        st.markdown(f"**{entry['asset']}**")
        st.markdown(entry["asset_detail"])

    with tab2:
        st.markdown(f"**{entry['vulnerability']}**")
        st.markdown(entry["vuln_detail"])

    with tab3:
        st.markdown(entry["threat_intel"])

    with tab4:
        st.markdown(entry["business_impact_text"])

    with tab5:
        if entry.get("nist_results"):
            best = entry["nist_results"][0]
            st.markdown(f"**{best['control_id']} — {best['title']}**")
            st.markdown(best["full_text"][:1500])
            if len(entry["nist_results"]) > 1:
                with st.expander("Additional relevant controls"):
                    for ctrl in entry["nist_results"][1:]:
                        st.markdown(f"**{ctrl['control_id']} — {ctrl['title']}**")
                        st.markdown(ctrl["full_text"][:500])
                        st.divider()
        else:
            st.warning("No NIST controls retrieved.")

    # Analysis
    st.markdown("**📝 Analysis**")
    st.markdown(entry["ranking_explanation"])

    st.divider()


# ─── Data Explorer ────────────────────────────────────────────────────────────

with st.expander("🔍 Data Explorer — Full Dataset", expanded=False):
    tab_assets, tab_vulns, tab_threats, tab_services = st.tabs([
        "Assets", "Vulnerabilities", "Threat Intel", "Business Services"
    ])

    with tab_assets:
        st.dataframe(data["assets"], use_container_width=True, height=300)
    with tab_vulns:
        st.dataframe(data["vulnerabilities"], use_container_width=True, height=300)
    with tab_threats:
        st.dataframe(data["threat_intel"], use_container_width=True, height=300)
    with tab_services:
        st.dataframe(data["business_services"], use_container_width=True, height=300)


# ─── Threat Report ────────────────────────────────────────────────────────────

with st.expander("📄 MDR Threat Advisory (Full Report)", expanded=False):
    st.markdown(data["threat_report"])


# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built for TawasolPay AI Engineer Assessment | "
    "Data: CISA KEV + NIST SP 800-53 Rev. 5 + Synthetic Threat Intel | "
    "RAG: ChromaDB + sentence-transformers | LLM: Batched Google Gemini Flash"
)
