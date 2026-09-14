"""
app.py — Streamlit Web Interface for TawasolPay AI Cyber Risk Assistant

Main entry point for the application. Orchestrates:
  1. Data ingestion (local CSVs + external CISA KEV + NIST SP 800-53)
  2. Multi-factor risk scoring
  3. NIST RAG retrieval for remediation guidance
  4. LLM-generated explanations
  5. Interactive dashboard display
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Must be first Streamlit call
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
    generate_llm_explanation,
    generate_executive_summary,
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
    .risk-card-high {
        border-left-color: #ef4444;
    }
    .risk-card-medium {
        border-left-color: #f59e0b;
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
    .section-header {
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    .factor-bar {
        background: #334155;
        border-radius: 4px;
        height: 8px;
        margin: 2px 0;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("🛡️ Cyber Risk Assistant")
    st.caption("TawasolPay — AI-Powered Risk Analysis")

    st.divider()

    # Load API key from .env (no UI input needed)
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    gemini_ready = configure_gemini(api_key)

    if gemini_ready:
        st.success("✅ Gemini API connected")
    else:
        st.warning("⚠️ GOOGLE_API_KEY not found in .env file")

    st.divider()
    st.markdown("**Data Sources**")
    st.markdown("- 📋 Local CSVs (assets, vulns, threats)")
    st.markdown("- 🏛️ CISA KEV Catalog (live)")
    st.markdown("- 📖 NIST SP 800-53 Rev. 5 (live + RAG)")


# ─── Main Page Header ────────────────────────────────────────────────────────

st.title("🛡️ TawasolPay Cyber Risk Assessment")
st.markdown(
    "**AI-powered risk analysis** combining asset inventory, vulnerability data, "
    "threat intelligence, and NIST SP 800-53 guidance to identify and prioritize "
    "the top cybersecurity risks."
)

st.divider()


# ─── Data Loading ────────────────────────────────────────────────────────────

with st.status("Loading and processing data...", expanded=True) as status:
    st.write("📂 Loading local datasets...")
    data = load_local_data()

    st.write("🌐 Fetching CISA KEV catalog...")
    kev_df = fetch_cisa_kev()

    st.write("📖 Fetching NIST SP 800-53 controls...")
    nist_controls = fetch_nist_controls()

    st.write("🔗 Building enriched dataset...")
    merged = build_merged_dataset(data, kev_df)

    st.write("🧠 Building NIST embedding index...")
    nist_collection = build_nist_index(nist_controls)

    st.write("📊 Computing risk scores...")
    top_risks = rank_risks(merged, top_n=5)

    status.update(label="✅ Data loaded and risks scored!", state="complete")


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


# ─── Retrieve NIST Controls for each risk ────────────────────────────────────

risk_entries = []

for idx, row in top_risks.iterrows():
    # Build semantic query for NIST retrieval
    query = build_risk_query(row)
    nist_results = query_nist_control(query, nist_collection, top_k=3)

    # Generate structured narrative
    entry = generate_risk_narrative(row, nist_results, int(row["risk_rank"]))

    # Generate LLM explanation if Gemini is available
    if gemini_ready:
        explanation = generate_llm_explanation(entry, row, nist_results)
        entry["ranking_explanation"] = explanation
    else:
        # Provide a basic explanation without LLM
        factors = entry["factor_scores"]
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
        factor_names = {
            "cvss": "CVSS severity",
            "exposure": "internet exposure",
            "exploit_kev": "active exploitation",
            "threat_match": "threat campaign match",
            "ransomware": "ransomware association",
            "business_criticality": "business criticality",
            "missing_controls": "missing compensating controls",
        }
        top_factor_str = ", ".join([f"{factor_names.get(f, f)} ({v:.0%})" for f, v in top_factors])
        entry["ranking_explanation"] = (
            f"This risk ranks #{entry['rank']} due to high scores in: {top_factor_str}. "
            f"Add a valid GOOGLE_API_KEY in the .env file for detailed AI-generated analysis."
        )

    entry["nist_results"] = nist_results
    risk_entries.append(entry)


# ─── Executive Summary ───────────────────────────────────────────────────────

st.markdown("## 📋 Executive Summary")

if gemini_ready:
    exec_summary = generate_executive_summary(risk_entries)
    st.info(exec_summary)
else:
    critical_count = sum(1 for e in risk_entries if e["risk_score"] > 75)
    st.info(
        f"**RISK LEVEL: HIGH** — {critical_count} of the top 5 risks score above 75/100. "
        f"Multiple internet-facing production systems have critical vulnerabilities with active "
        f"exploitation by threat actors targeting the Middle East fintech sector. "
        f"Immediate patching of VPN appliances and exposed API gateways is the top priority."
    )

st.divider()


# ─── Top 5 Risks ─────────────────────────────────────────────────────────────

st.markdown("## 🎯 Top 5 Prioritized Risks")
st.caption(
    "Ranked by composite risk score incorporating CVSS, internet exposure, "
    "active exploitation, threat campaign matches, business criticality, and missing controls."
)

for entry in risk_entries:
    rank = entry["rank"]
    score = entry["risk_score"]

    # Color coding
    if score >= 80:
        score_color = "🔴"
        border_color = "#ef4444"
    elif score >= 60:
        score_color = "🟠"
        border_color = "#f59e0b"
    else:
        score_color = "🟡"
        border_color = "#eab308"

    st.markdown(f"### {score_color} Risk #{rank} — {entry['vulnerability']}")
    st.markdown(f"**Risk Score: {score}/100**")

    # Factor score bars
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

    # Main content in tabs
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
            st.warning("No NIST controls retrieved. Ensure the NIST index is built.")

    # Ranking explanation
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
    "RAG: ChromaDB + sentence-transformers | LLM: Google Gemini 2.5 Flash"
)
