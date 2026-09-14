"""
report_generator.py — Risk Report Generation using Gemini LLM

Takes the top-5 ranked risks with their NIST control matches and generates
a human-readable, structured risk report.

Each risk entry includes:
  - Asset and context
  - Vulnerability and why it matters
  - Matched threat intelligence
  - Business service at risk
  - NIST remediation guidance (retrieved via RAG)
  - Plain-English explanation of ranking
"""

import os
import google.generativeai as genai
import streamlit as st

# ─── Gemini Configuration ────────────────────────────────────────────────────


def configure_gemini(api_key=None):
    """Configure the Gemini API with the provided key. Tests validity."""
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return False
    try:
        genai.configure(api_key=key)
        # Quick test to verify the key works
        model = genai.GenerativeModel("gemini-3.6-flash")
        model.generate_content("test", generation_config={"max_output_tokens": 5})
        return True
    except Exception as e:
        print(f"Gemini API key validation failed: {e}")
        return False


def generate_risk_narrative(risk_row, nist_controls, rank, total=5):
    """
    Generate a plain-English risk narrative for a single risk entry.

    Args:
        risk_row: pandas Series with all enriched risk data
        nist_controls: list of dicts from NIST RAG retrieval
        rank: 1-based rank of this risk
        total: total risks in the report

    Returns:
        dict with structured narrative fields
    """
    # Extract key fields
    asset_name = risk_row.get("asset_name", "Unknown")
    asset_type = risk_row.get("asset_type", "Unknown")
    environment = risk_row.get("environment", "Unknown")
    location = risk_row.get("location", "Unknown")
    owner_team = risk_row.get("owner_team", "Unknown")
    internet_exposed = risk_row.get("internet_exposed", risk_row.get("asset_exposure", "Unknown"))

    vuln_name = risk_row.get("vulnerability_name", "Unknown")
    cve = risk_row.get("cve", "Unknown")
    cvss = risk_row.get("cvss", "N/A")
    severity = risk_row.get("severity", "Unknown")
    days_open = risk_row.get("days_open", "Unknown")
    exploit_available = risk_row.get("exploit_available", "Unknown")
    patch_available = risk_row.get("patch_available", "Unknown")

    business_service = risk_row.get("business_service", "Unknown")
    business_impact = risk_row.get("business_impact", "")
    revenue_impact = risk_row.get("revenue_impact", "")
    compliance_scope = risk_row.get("compliance_scope", "")
    rto_hours = risk_row.get("rto_hours", "")
    customer_facing = risk_row.get("customer_facing", "")

    ti_actor = risk_row.get("ti_threat_actor", "")
    ti_campaign = risk_row.get("ti_campaign_name", "")
    ti_ransomware = risk_row.get("ti_ransomware", "No")
    ti_summary = risk_row.get("ti_summary", "")

    in_kev = risk_row.get("in_kev", False)
    kev_ransomware = risk_row.get("kev_ransomware", False)

    risk_score = risk_row.get("risk_score", 0)
    factor_scores = risk_row.get("factor_scores", {})

    edr_installed = risk_row.get("edr_installed", "Unknown")

    # Format NIST controls
    nist_text = ""
    if nist_controls:
        best = nist_controls[0]
        nist_text = f"**{best['control_id']} — {best['title']}**\n\n{best['full_text']}"

    # Build the structured entry (no LLM needed for basic structure)
    entry = {
        "rank": rank,
        "risk_score": round(risk_score * 100, 1),
        "asset": f"{asset_name} ({asset_type})",
        "asset_detail": f"Environment: {environment} | Location: {location} | Owner: {owner_team} | Internet-Exposed: {internet_exposed} | EDR: {edr_installed}",
        "vulnerability": f"{vuln_name} ({cve})",
        "vuln_detail": f"CVSS: {cvss} ({severity}) | Exploit Available: {exploit_available} | Patch Available: {patch_available} | Days Open: {days_open}",
        "threat_intel": "",
        "business_impact_text": "",
        "nist_guidance": nist_text,
        "ranking_explanation": "",
        "factor_scores": factor_scores,
    }

    # Threat intel
    if ti_actor:
        entry["threat_intel"] = f"**Threat Actor:** {ti_actor} | **Campaign:** {ti_campaign} | **Ransomware:** {ti_ransomware}"
        if ti_summary:
            entry["threat_intel"] += f"\n\n{ti_summary.split(' | ')[0]}"
    elif in_kev:
        entry["threat_intel"] = "Listed in CISA Known Exploited Vulnerabilities catalog"
        if kev_ransomware:
            entry["threat_intel"] += " — **associated with ransomware campaigns**"
    else:
        entry["threat_intel"] = "No active threat campaign match in current intelligence"

    # Business impact
    bi_parts = [f"**Service:** {business_service}"]
    if business_impact:
        bi_parts.append(f"**Impact:** {business_impact}")
    if revenue_impact:
        bi_parts.append(f"**Revenue Impact:** {revenue_impact}")
    if compliance_scope:
        bi_parts.append(f"**Compliance:** {compliance_scope}")
    if customer_facing:
        bi_parts.append(f"**Customer-Facing:** {customer_facing}")
    if rto_hours:
        bi_parts.append(f"**RTO:** {rto_hours} hours")
    entry["business_impact_text"] = " | ".join(bi_parts)

    return entry


def generate_llm_explanation(risk_entry, risk_row, nist_controls):
    """
    Use Gemini to generate a plain-English explanation of why this risk
    ranks where it does and what the NIST guidance recommends.

    Returns the explanation string.
    """
    nist_ctrl = nist_controls[0] if nist_controls else {"control_id": "N/A", "title": "N/A", "full_text": "No NIST control retrieved"}

    prompt = f"""You are a cybersecurity risk analyst writing a brief for a technical manager.

Given this risk entry, write TWO short paragraphs:

1. **Why this ranks #{risk_entry['rank']}**: Explain in plain English why this risk is ranked here. Reference specific factors: internet exposure, active exploitation, threat actor campaigns, business criticality, and missing controls. Be specific about this vulnerability and asset.

2. **What NIST recommends**: Based on the retrieved NIST SP 800-53 control below, explain in 2-3 sentences what the control recommends and how it applies to this specific risk. Do NOT make up control content — use only what is provided below.

--- RISK DATA ---
Asset: {risk_entry['asset']}
Detail: {risk_entry['asset_detail']}
Vulnerability: {risk_entry['vulnerability']}
Detail: {risk_entry['vuln_detail']}
Threat Intel: {risk_entry['threat_intel']}
Business Impact: {risk_entry['business_impact_text']}
Risk Score: {risk_entry['risk_score']}/100

Factor Scores:
- CVSS Factor: {risk_entry['factor_scores'].get('cvss', 0):.2f}
- Internet Exposure: {risk_entry['factor_scores'].get('exposure', 0):.2f}
- Active Exploit/KEV: {risk_entry['factor_scores'].get('exploit_kev', 0):.2f}
- Threat Campaign: {risk_entry['factor_scores'].get('threat_match', 0):.2f}
- Ransomware: {risk_entry['factor_scores'].get('ransomware', 0):.2f}
- Business Criticality: {risk_entry['factor_scores'].get('business_criticality', 0):.2f}
- Missing Controls: {risk_entry['factor_scores'].get('missing_controls', 0):.2f}

--- NIST CONTROL (RETRIEVED VIA RAG — USE ONLY THIS) ---
Control ID: {nist_ctrl['control_id']}
Title: {nist_ctrl['title']}
Full Text: {nist_ctrl['full_text'][:2000]}

--- INSTRUCTIONS ---
Write concisely. No headers. No bullet points. Just two clear paragraphs.
Do not start with "This risk..." — start with the specific asset or vulnerability name.
"""

    try:
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[LLM explanation unavailable: {e}]"


def generate_executive_summary(risk_entries):
    """
    Generate an executive summary for the full top-5 risk report.
    """
    risks_summary = ""
    for entry in risk_entries:
        risks_summary += f"- Risk #{entry['rank']}: {entry['vulnerability']} on {entry['asset']} (Score: {entry['risk_score']}/100)\n"

    prompt = f"""You are a cybersecurity risk analyst writing an executive summary for the CISO of TawasolPay, a fintech company in Dubai, UAE.

The MDR provider sent an urgent advisory about active ransomware campaigns targeting fintech firms in the Middle East. After analyzing 60 assets, 114 vulnerabilities, and 40 threat intel records, here are the top 5 risks:

{risks_summary}

Write a 3-4 sentence executive summary. Be direct. State the overall risk level, the most critical finding, and the recommended immediate action. Do not use bullet points. Write for a board-level audience.
"""

    try:
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"**HIGH RISK**: Multiple critical vulnerabilities with active exploitation detected across internet-facing production infrastructure. Immediate patching and compensating controls required. [Detailed summary unavailable: {e}]"
