"""
report_generator.py — Risk Report Generation using Gemini LLM with Batched Calling

Key Optimizations:
1. Batched Single LLM Call: Combines Executive Summary + all 5 risk analyses into
   ONE single structured JSON call, eliminating multi-request latency and 429 quota exhaustion.
2. Model Selection: Uses lightweight production Gemini flash models with graceful model fallback.
3. Deterministic Fallback: Never leaks raw API or JSON errors to users. If rate limits, network timeouts,
   or invalid keys occur, immediately produces a high-fidelity, rule-based report.
"""

import os
import json
import re
import google.generativeai as genai

# Models tried in order: high-quota flash-lite first, then standard flash
CANDIDATE_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]


def get_active_model_name():
    """Return the preferred Gemini model name."""
    return CANDIDATE_MODELS[0]


def configure_gemini(api_key=None):
    """
    Configure the Gemini API with the provided key.
    Tests model connectivity with minimal token generation.
    """
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return False
    try:
        genai.configure(api_key=key)
        # Verify connectivity using the lightweight candidate model
        for model_name in CANDIDATE_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                model.generate_content("ping", generation_config={"max_output_tokens": 3})
                return True
            except Exception:
                continue
        return False
    except Exception as e:
        print(f"Gemini configuration check failed: {e}")
        return False


def generate_risk_narrative(risk_row, nist_controls, rank, total=5):
    """
    Construct a base structured dictionary for a risk entry without calling LLMs.
    Used as the underlying data layer for rendering and template fallbacks.
    """
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

    nist_text = ""
    if nist_controls:
        best = nist_controls[0]
        nist_text = f"**{best['control_id']} — {best['title']}**\n\n{best['full_text']}"

    entry = {
        "rank": rank,
        "risk_score": round(risk_score * 100, 1) if risk_score <= 1.0 else round(risk_score, 1),
        "asset": f"{asset_name} ({asset_type})",
        "asset_name": asset_name,
        "asset_type": asset_type,
        "environment": environment,
        "location": location,
        "owner_team": owner_team,
        "internet_exposed": str(internet_exposed).strip(),
        "edr_installed": str(edr_installed).strip(),
        "asset_detail": f"Environment: {environment} | Location: {location} | Owner: {owner_team} | Internet-Exposed: {internet_exposed} | EDR: {edr_installed}",
        "vulnerability": f"{vuln_name} ({cve})",
        "vuln_name": vuln_name,
        "cve": cve,
        "cvss": cvss,
        "severity": severity,
        "days_open": days_open,
        "patch_available": patch_available,
        "vuln_detail": f"CVSS: {cvss} ({severity}) | Exploit Available: {exploit_available} | Patch Available: {patch_available} | Days Open: {days_open}",
        "threat_actor": ti_actor or ("CISA KEV" if in_kev else "Unknown"),
        "campaign_name": ti_campaign or ("Active KEV Exploitation" if in_kev else "Unspecified Campaign"),
        "threat_intel": "",
        "business_service": business_service,
        "business_impact_text": "",
        "nist_guidance": nist_text,
        "nist_results": nist_controls,
        "ranking_explanation": "",
        "factor_scores": factor_scores,
    }

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


def _build_deterministic_executive_summary(risk_entries):
    """Generate high-quality rule-based executive summary without API calls."""
    if not risk_entries:
        return "No high-priority cybersecurity risks currently detected."

    top_entry = risk_entries[0]
    top_asset = top_entry.get("asset_name", "perimeter systems")
    top_vuln = top_entry.get("vuln_name", "critical edge vulnerabilities")
    top_cve = top_entry.get("cve", "")
    top_campaign = top_entry.get("campaign_name", "targeted external intrusion")

    high_risk_count = sum(1 for e in risk_entries if e.get("risk_score", 0) >= 80)

    summary = (
        f"TawasolPay currently faces an elevated risk posture with {high_risk_count} of 5 top prioritized "
        f"vulnerabilities exhibiting active exploitation or immediate perimeter exposure. "
        f"The primary threat concerns {top_vuln} ({top_cve}) affecting {top_asset}, currently targeted under the "
        f"{top_campaign} campaign with confirmed adversary activity across regional financial systems. "
        f"Immediate operational priority requires an emergency remediation window to patch perimeter-facing assets, "
        f"revoke exposed session credentials, and verify endpoint visibility across all production gateways."
    )
    return summary


def _build_deterministic_risk_explanation(entry):
    """Generate concise 2-sentence deterministic analysis for a risk item."""
    asset = entry.get("asset_name", "The asset")
    cve = entry.get("cve", "the identified vulnerability")
    campaign = entry.get("campaign_name", "active threat campaigns")
    actor = entry.get("threat_actor", "external threat actors")
    exposure = entry.get("internet_exposed", "No")
    is_exposed = str(exposure).lower() in ("yes", "internet", "true")
    nist_controls = entry.get("nist_results", [])
    nist_id = nist_controls[0].get("control_id", "SI-2") if nist_controls else "SI-2"
    nist_title = nist_controls[0].get("title", "Flaw Remediation") if nist_controls else "Flaw Remediation"
    biz_svc = entry.get("business_service", "core infrastructure")

    p1 = (
        f"{asset} is prioritized at Rank #{entry.get('rank', 1)} because {cve} is "
        f"{'directly internet-exposed and' if is_exposed else 'deployed in production and'} subject to weaponized exploitation "
        f"by {actor} ({campaign}), directly threatening the {biz_svc} service."
    )
    p2 = (
        f"To address this exposure, NIST SP 800-53 Control {nist_id} ({nist_title}) specifies applying prompt "
        f"mitigation through authoritative security updates, restricting ingress vector access, and continuously verifying "
        f"compensating controls across surrounding network boundaries."
    )
    return f"{p1}\n\n{p2}"


def generate_batched_report(risk_entries, api_key=None):
    """
    Generate Executive Summary and all 5 risk analyses in ONE single LLM call.
    Uses structured JSON format. Automatically falls back deterministically on any failure.

    Returns:
        tuple: (executive_summary: str, updated_risk_entries: list)
    """
    # 1. Populate deterministic defaults for guaranteed safety
    fallback_exec_summary = _build_deterministic_executive_summary(risk_entries)
    for entry in risk_entries:
        entry["ranking_explanation"] = _build_deterministic_risk_explanation(entry)

    # If no API key configured, return deterministic output immediately
    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        return fallback_exec_summary, risk_entries

    # 2. Prepare payload for the batched prompt
    risks_payload = []
    for entry in risk_entries:
        nist_first = entry.get("nist_results", [{}])[0] if entry.get("nist_results") else {}
        risks_payload.append({
            "rank": entry.get("rank"),
            "risk_score": entry.get("risk_score"),
            "asset": entry.get("asset"),
            "asset_detail": entry.get("asset_detail"),
            "vulnerability": entry.get("vulnerability"),
            "vuln_detail": entry.get("vuln_detail"),
            "threat_intel": entry.get("threat_intel"),
            "business_impact": entry.get("business_impact_text"),
            "factors": entry.get("factor_scores", {}),
            "nist_control_id": nist_first.get("control_id", "SI-2"),
            "nist_control_title": nist_first.get("title", "Flaw Remediation"),
            "nist_control_text": nist_first.get("full_text", "")[:1000],
        })

    prompt = f"""You are an elite cybersecurity risk analyst writing an executive briefing for leadership at TawasolPay (fintech company in Dubai, UAE).

Review the top 5 cybersecurity risks identified across our systems:
{json.dumps(risks_payload, indent=2)}

TASK:
Write a high-quality, professional risk report in valid JSON format:
1. "executive_summary": A high-impact 3-4 sentence summary for executive leadership outlining the overall risk level, primary threat vectors (referencing top vulnerability and campaigns), and urgent operational directives.
2. "analyses": A list containing an object for each of the 5 risks (ranked 1 to 5). For each risk:
   - "rank": The integer rank (1 to 5).
   - "analysis": Write TWO detailed, substantive paragraphs (do NOT output placeholder words):
       * Paragraph 1: Why this risk ranks where it does. Detail the specific asset, vulnerability, internet exposure, threat actor/campaign, business criticality, and missing controls.
       * Paragraph 2: What NIST SP 800-53 recommends based on the retrieved control, and specific actionable steps for technical teams to remediate it.

Return ONLY a valid JSON object matching this structure:
{{
  "executive_summary": "TawasolPay faces an elevated...",
  "analyses": [
    {{"rank": 1, "analysis": "Detailed first paragraph explaining ranking...\\n\\nDetailed second paragraph explaining NIST remediation..."}},
    {{"rank": 2, "analysis": "Detailed first paragraph explaining ranking...\\n\\nDetailed second paragraph explaining NIST remediation..."}},
    {{"rank": 3, "analysis": "Detailed first paragraph explaining ranking...\\n\\nDetailed second paragraph explaining NIST remediation..."}},
    {{"rank": 4, "analysis": "Detailed first paragraph explaining ranking...\\n\\nDetailed second paragraph explaining NIST remediation..."}},
    {{"rank": 5, "analysis": "Detailed first paragraph explaining ranking...\\n\\nDetailed second paragraph explaining NIST remediation..."}}
  ]
}}
"""

    try:
        genai.configure(api_key=key)
        response_text = None

        # Try candidate models
        for model_name in CANDIDATE_MODELS:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                        "max_output_tokens": 2048,
                    },
                )
                resp = model.generate_content(prompt)
                if resp and resp.text:
                    response_text = resp.text.strip()
                    break
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue

        if not response_text:
            return fallback_exec_summary, risk_entries

        # Parse JSON
        # Clean any potential markdown wrapper
        cleaned_json = response_text
        if cleaned_json.startswith("```"):
            cleaned_json = re.sub(r"^```(?:json)?\n?", "", cleaned_json)
            cleaned_json = re.sub(r"\n?```$", "", cleaned_json)

        parsed = json.loads(cleaned_json)

        exec_summary = parsed.get("executive_summary", "").strip() or fallback_exec_summary
        analyses_map = {item.get("rank"): item.get("analysis", "") for item in parsed.get("analyses", [])}

        for entry in risk_entries:
            rank = entry.get("rank")
            if rank in analyses_map and analyses_map[rank]:
                entry["ranking_explanation"] = analyses_map[rank].strip()

        return exec_summary, risk_entries

    except Exception as err:
        print(f"Batched report generation error (fallback engaged): {err}")
        return fallback_exec_summary, risk_entries
