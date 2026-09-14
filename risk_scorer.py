"""
risk_scorer.py — Multi-Factor Risk Scoring Engine

Ranks vulnerabilities using a composite score that goes beyond CVSS alone.
Factors:
  1. CVSS Base Score (15%)
  2. Internet Exposure (20%)
  3. Active Exploit / CISA KEV Match (20%)
  4. Threat Campaign Match (15%)
  5. Ransomware Association (10%)
  6. Business Criticality (10%)
  7. Missing Compensating Controls (10%)

This ensures a CVSS-10 on an internal dev server ranks LOWER than a
CVSS-8 on an internet-exposed payment gateway with an active ransomware campaign.
"""

import pandas as pd


# ─── Scoring Weights ──────────────────────────────────────────────────────────

WEIGHTS = {
    "cvss": 0.15,
    "exposure": 0.20,
    "exploit_kev": 0.20,
    "threat_match": 0.15,
    "ransomware": 0.10,
    "business_criticality": 0.10,
    "missing_controls": 0.10,
}

# ─── Normalization Maps ──────────────────────────────────────────────────────

CRITICALITY_SCORES = {
    "Critical": 1.0,
    "High": 0.7,
    "Medium": 0.4,
    "Low": 0.2,
}

REVENUE_IMPACT_SCORES = {
    "Critical": 1.0,
    "High": 0.8,
    "Medium": 0.5,
    "Low": 0.2,
}

RISK_APPETITE_SCORES = {
    "Very Low": 1.0,   # Very low appetite = highest risk weight
    "Low": 0.8,
    "Medium": 0.5,
    "High": 0.3,
}


def score_cvss(row):
    """Normalize CVSS score to 0-1 range."""
    cvss = row.get("cvss", 0)
    try:
        cvss = float(cvss)
    except (ValueError, TypeError):
        cvss = 0
    return min(cvss / 10.0, 1.0)


def score_exposure(row):
    """Score based on internet exposure."""
    # Check multiple possible column names
    exposure = str(row.get("asset_exposure", row.get("internet_exposed", ""))).strip().lower()
    if exposure in ("internet", "yes"):
        return 1.0
    elif exposure in ("internal", "no"):
        return 0.3
    return 0.5


def score_exploit_kev(row):
    """Score based on active exploitation and CISA KEV presence."""
    in_kev = row.get("in_kev", False)
    exploit_available = str(row.get("exploit_available", "")).strip().lower() == "yes"

    if in_kev:
        return 1.0
    elif exploit_available:
        return 0.7
    return 0.0


def score_threat_match(row):
    """Score based on matching active threat campaigns."""
    has_match = row.get("has_threat_match", False)
    ti_confidence = str(row.get("ti_confidence", "")).lower()
    ti_exploit_maturity = str(row.get("ti_exploit_maturity", "")).lower()

    if not has_match:
        return 0.0

    score = 0.6  # base for having a match

    # Boost for high confidence
    if "high" in ti_confidence:
        score += 0.2

    # Boost for weaponized exploits
    if "weaponized" in ti_exploit_maturity:
        score += 0.2
    elif "active" in ti_exploit_maturity:
        score += 0.15
    elif "proof of concept" in ti_exploit_maturity:
        score += 0.05

    return min(score, 1.0)


def score_ransomware(row):
    """Score based on ransomware association."""
    # Check threat intel ransomware match
    ti_ransomware = str(row.get("ti_ransomware", "")).strip().lower() == "yes"
    # Check KEV ransomware flag
    kev_ransomware = row.get("kev_ransomware", False)

    if ti_ransomware or kev_ransomware:
        return 1.0
    return 0.0


def score_business_criticality(row):
    """Score based on business service criticality."""
    # Asset criticality
    asset_crit = str(row.get("criticality", "")).strip()
    asset_score = CRITICALITY_SCORES.get(asset_crit, 0.3)

    # Revenue impact from business service
    revenue = str(row.get("revenue_impact", "")).strip()
    revenue_score = REVENUE_IMPACT_SCORES.get(revenue, 0.3)

    # Customer facing bonus
    customer_facing = str(row.get("customer_facing", "")).strip().lower() == "yes"
    facing_bonus = 0.1 if customer_facing else 0.0

    # Risk appetite (lower appetite = higher risk priority)
    appetite = str(row.get("risk_appetite", "")).strip()
    appetite_score = RISK_APPETITE_SCORES.get(appetite, 0.5)

    # Compliance scope bonus
    compliance = str(row.get("compliance_scope", "")).strip().lower()
    compliance_bonus = 0.0
    if "pci dss" in compliance:
        compliance_bonus = 0.15
    elif "gdpr" in compliance or "pdpl" in compliance:
        compliance_bonus = 0.1
    elif "soc 2" in compliance or "iso 27001" in compliance:
        compliance_bonus = 0.05

    combined = (asset_score * 0.35 + revenue_score * 0.25 +
                appetite_score * 0.15 + facing_bonus + compliance_bonus)
    return min(combined, 1.0)


def score_missing_controls(row):
    """Score based on missing compensating controls."""
    penalties = 0.0

    # No EDR installed
    edr = str(row.get("edr_installed", "")).strip().lower()
    if edr in ("no", "false", ""):
        penalties += 0.35

    # No patch available
    patch = str(row.get("patch_available", "")).strip().lower()
    if patch in ("no", "false"):
        penalties += 0.30

    # Stale asset (not seen recently)
    try:
        last_seen = int(row.get("last_seen_days", 0))
    except (ValueError, TypeError):
        last_seen = 0
    if last_seen > 30:
        penalties += 0.15

    # Long days open
    try:
        days_open = int(row.get("days_open", 0))
    except (ValueError, TypeError):
        days_open = 0
    if days_open > 90:
        penalties += 0.20

    return min(penalties, 1.0)


def compute_risk_score(row):
    """Compute the composite risk score for a single vulnerability-asset pair."""
    scores = {
        "cvss": score_cvss(row),
        "exposure": score_exposure(row),
        "exploit_kev": score_exploit_kev(row),
        "threat_match": score_threat_match(row),
        "ransomware": score_ransomware(row),
        "business_criticality": score_business_criticality(row),
        "missing_controls": score_missing_controls(row),
    }

    # Weighted sum
    composite = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)

    return composite, scores


def rank_risks(merged_df, top_n=5):
    """
    Compute risk scores for all vulnerability-asset pairs and return
    the top N risks, sorted by composite risk score descending.

    Diversification strategy:
    - Keep the highest-scoring instance per CVE (avoid showing the same CVE
      on two nearly-identical assets, e.g., vpn-edge-01 and vpn-edge-02)
    - This ensures the top 5 covers different risk categories

    Returns a DataFrame with all original columns plus:
      - risk_score: composite score (0-1)
      - factor_scores: dict of individual factor scores
      - risk_rank: 1-based rank
    """
    results = []

    for idx, row in merged_df.iterrows():
        composite, factor_scores = compute_risk_score(row)
        results.append({
            "index": idx,
            "risk_score": composite,
            "factor_scores": factor_scores,
        })

    scores_df = pd.DataFrame(results).set_index("index")

    # Join back with merged_df
    ranked = merged_df.join(scores_df)

    # Sort by risk score descending
    ranked = ranked.sort_values("risk_score", ascending=False)

    # Deduplicate: keep the highest-scoring entry per CVE
    # This prevents showing CVE-2023-4966 on both load-balancer-prod-01 AND -02
    ranked = ranked.drop_duplicates(subset=["cve"], keep="first")

    # Take top N
    ranked = ranked.head(top_n).reset_index(drop=True)
    ranked["risk_rank"] = ranked.index + 1

    return ranked
