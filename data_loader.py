"""
data_loader.py — Data Ingestion Layer for TawasolPay Cyber Risk Assistant

Loads all local CSV files, the synthetic threat report, and fetches external
data sources (CISA KEV catalog and NIST SP 800-53 Rev. 5 controls).

Design Decision:
- Structured CSVs (assets, vulnerabilities, threat_intel, business_services, remediation)
  are loaded into pandas DataFrames for filtered/join queries.
- NIST SP 800-53 control prose descriptions are embedded into ChromaDB for semantic search (RAG).
- CISA KEV is loaded as structured data and cross-referenced by CVE ID.
"""

import os
import pandas as pd
import requests
import json
import streamlit as st

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")

# ─── CISA KEV Catalog ────────────────────────────────────────────────────────

CISA_KEV_JSON_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CISA_KEV_CSV_URL = "https://raw.githubusercontent.com/cisagov/kev-data/refs/heads/main/data/known_exploited_vulnerabilities.csv"

# ─── NIST SP 800-53 Rev. 5 ───────────────────────────────────────────────────

NIST_OSCAL_JSON_URL = "https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"


@st.cache_data(show_spinner="Loading local datasets...")
def load_local_data():
    """Load all local CSV files and the threat report."""
    assets = pd.read_csv(os.path.join(DATA_DIR, "assets.csv"))
    vulnerabilities = pd.read_csv(os.path.join(DATA_DIR, "vulnerabilities.csv"))
    threat_intel = pd.read_csv(os.path.join(DATA_DIR, "threat_intelligence.csv"))
    business_services = pd.read_csv(os.path.join(DATA_DIR, "business_services.csv"))
    remediation = pd.read_csv(os.path.join(DATA_DIR, "remediation_guidance.csv"))

    # Load threat report markdown
    with open(os.path.join(DATA_DIR, "synthetic_threat_report.md"), "r", encoding="utf-8") as f:
        threat_report = f.read()

    # Clean column names (strip whitespace)
    for df in [assets, vulnerabilities, threat_intel, business_services, remediation]:
        df.columns = df.columns.str.strip()

    return {
        "assets": assets,
        "vulnerabilities": vulnerabilities,
        "threat_intel": threat_intel,
        "business_services": business_services,
        "remediation": remediation,
        "threat_report": threat_report,
    }


@st.cache_data(show_spinner="Fetching CISA KEV catalog...")
def fetch_cisa_kev():
    """
    Fetch the CISA Known Exploited Vulnerabilities catalog.
    Returns a DataFrame with columns: cveID, knownRansomwareCampaignUse, dateAdded, etc.
    Falls back to CSV if JSON fails.
    """
    try:
        resp = requests.get(CISA_KEV_JSON_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        kev_df = pd.DataFrame(data.get("vulnerabilities", []))
        if not kev_df.empty:
            return kev_df
    except Exception as e:
        print(f"CISA KEV JSON fetch failed: {e}, trying CSV...")

    # Fallback to CSV
    try:
        resp = requests.get(CISA_KEV_CSV_URL, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        kev_df = pd.read_csv(StringIO(resp.text))
        return kev_df
    except Exception as e:
        print(f"CISA KEV CSV fetch also failed: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Fetching NIST SP 800-53 Rev. 5 controls...")
def fetch_nist_controls():
    """
    Fetch NIST SP 800-53 Rev. 5 control catalog from the OSCAL JSON.
    Returns a list of dicts with keys: control_id, title, description, discussion, full_text.
    """
    controls = []

    def extract_prose(parts, part_name):
        """Extract prose text from a named part in the OSCAL structure."""
        if not parts:
            return ""
        for part in parts:
            if part.get("name") == part_name:
                prose = part.get("prose", "")
                # Also check for nested parts (sub-items in statements)
                sub_parts = part.get("parts", [])
                if sub_parts:
                    sub_texts = []
                    for sp in sub_parts:
                        sp_prose = sp.get("prose", "")
                        if sp_prose:
                            sub_texts.append(sp_prose)
                        # Go one more level deep for (a)(1) style items
                        for ssp in sp.get("parts", []):
                            ssp_prose = ssp.get("prose", "")
                            if ssp_prose:
                                sub_texts.append("  " + ssp_prose)
                    if sub_texts:
                        prose = prose + "\n" + "\n".join(sub_texts) if prose else "\n".join(sub_texts)
                return prose
        return ""

    def process_control(ctrl):
        """Process a single OSCAL control into our schema."""
        ctrl_id = ctrl.get("id", "").upper().replace("-", "-")
        title = ctrl.get("title", "")
        parts = ctrl.get("parts", [])

        statement = extract_prose(parts, "statement")
        guidance = extract_prose(parts, "guidance")

        description = statement if statement else ""
        discussion = guidance if guidance else ""

        full_text = f"{ctrl_id} - {title}"
        if description:
            full_text += f"\n\nStatement: {description}"
        if discussion:
            full_text += f"\n\nGuidance: {discussion}"

        return {
            "control_id": ctrl_id,
            "title": title,
            "description": description,
            "discussion": discussion,
            "full_text": full_text,
        }

    try:
        resp = requests.get(NIST_OSCAL_JSON_URL, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        catalog = data.get("catalog", {})
        groups = catalog.get("groups", [])

        for group in groups:
            for ctrl in group.get("controls", []):
                control = process_control(ctrl)
                if control["control_id"]:
                    controls.append(control)

                # Also process control enhancements (sub-controls)
                for enhancement in ctrl.get("controls", []):
                    enh = process_control(enhancement)
                    if enh["control_id"]:
                        controls.append(enh)

        print(f"Loaded {len(controls)} NIST SP 800-53 Rev. 5 controls from OSCAL JSON")

    except Exception as e:
        print(f"NIST OSCAL JSON fetch failed: {e}")

        # Fallback: try the NIST CSRC download page
        try:
            fallback_url = "https://csrc.nist.gov/extensions/nudp/services/json/sp800-53/rev-5"
            resp = requests.get(fallback_url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            families = data if isinstance(data, list) else data.get("families", data.get("controls", []))
            for family in families:
                if isinstance(family, dict):
                    for c in family.get("controls", []):
                        control = {
                            "control_id": c.get("number", c.get("id", "")),
                            "title": c.get("title", c.get("name", "")),
                            "description": c.get("statement", c.get("description", "")),
                            "discussion": c.get("discussion", c.get("supplemental_guidance", "")),
                        }
                        control["full_text"] = f"{control['control_id']} - {control['title']}\n\n{control['description']}\n\n{control['discussion']}"
                        controls.append(control)
            if controls:
                print(f"Loaded {len(controls)} NIST controls from fallback JSON API")
        except Exception as e2:
            print(f"NIST fallback also failed: {e2}")

    if not controls:
        print("WARNING: Could not fetch NIST controls from any source.")

    return controls


def enrich_with_kev(vulnerabilities_df, kev_df):
    """
    Cross-reference vulnerabilities with CISA KEV catalog.
    Adds columns: in_kev, kev_ransomware, kev_date_added, kev_required_action
    """
    if kev_df.empty:
        vulnerabilities_df["in_kev"] = False
        vulnerabilities_df["kev_ransomware"] = False
        vulnerabilities_df["kev_date_added"] = None
        vulnerabilities_df["kev_required_action"] = None
        return vulnerabilities_df

    # Identify the CVE ID column in KEV
    kev_cve_col = None
    for candidate in ["cveID", "CVE ID", "cve_id", "cveId"]:
        if candidate in kev_df.columns:
            kev_cve_col = candidate
            break
    if kev_cve_col is None:
        kev_cve_col = kev_df.columns[0]

    kev_set = set(kev_df[kev_cve_col].dropna().str.strip())

    # Ransomware column
    ransomware_col = None
    for candidate in ["knownRansomwareCampaignUse", "ransomware", "Ransomware"]:
        if candidate in kev_df.columns:
            ransomware_col = candidate
            break

    # Date added column
    date_col = None
    for candidate in ["dateAdded", "date_added", "Date Added"]:
        if candidate in kev_df.columns:
            date_col = candidate
            break

    # Required action
    action_col = None
    for candidate in ["requiredAction", "required_action", "Required Action"]:
        if candidate in kev_df.columns:
            action_col = candidate
            break

    # Build lookup dict
    kev_lookup = {}
    for _, row in kev_df.iterrows():
        cve = str(row.get(kev_cve_col, "")).strip()
        if cve:
            kev_lookup[cve] = {
                "ransomware": str(row.get(ransomware_col, "Unknown")).strip().lower() == "known" if ransomware_col else False,
                "date_added": str(row.get(date_col, "")) if date_col else "",
                "required_action": str(row.get(action_col, "")) if action_col else "",
            }

    # Enrich
    vulnerabilities_df["in_kev"] = vulnerabilities_df["cve"].apply(
        lambda x: str(x).strip() in kev_set
    )
    vulnerabilities_df["kev_ransomware"] = vulnerabilities_df["cve"].apply(
        lambda x: kev_lookup.get(str(x).strip(), {}).get("ransomware", False)
    )
    vulnerabilities_df["kev_date_added"] = vulnerabilities_df["cve"].apply(
        lambda x: kev_lookup.get(str(x).strip(), {}).get("date_added", "")
    )
    vulnerabilities_df["kev_required_action"] = vulnerabilities_df["cve"].apply(
        lambda x: kev_lookup.get(str(x).strip(), {}).get("required_action", "")
    )

    return vulnerabilities_df


def build_merged_dataset(data, kev_df):
    """
    Merge vulnerabilities with assets, business services, threat intel, and KEV data
    to create a single enriched DataFrame for risk scoring.
    """
    vulns = data["vulnerabilities"].copy()
    assets = data["assets"].copy()
    threat_intel = data["threat_intel"].copy()
    biz_services = data["business_services"].copy()

    # Enrich with KEV
    vulns = enrich_with_kev(vulns, kev_df)

    # Merge with assets
    merged = vulns.merge(assets, on="asset_id", how="left", suffixes=("_vuln", "_asset"))

    # Merge with business services
    # The asset has business_service column which maps to business_services.business_service
    bs_col = "business_service" if "business_service" in merged.columns else "business_service_asset"
    merged = merged.merge(
        biz_services,
        left_on=bs_col,
        right_on="business_service",
        how="left",
        suffixes=("", "_biz"),
    )

    # Match threat intel by CVE
    # A vulnerability matches threat intel if its CVE appears in threat_intel.matched_cve_or_control
    threat_by_cve = threat_intel.groupby("matched_cve_or_control").agg({
        "threat_actor": lambda x: ", ".join(x.unique()),
        "campaign_name": lambda x: ", ".join(x.unique()),
        "ransomware_association": lambda x: "Yes" if "Yes" in x.values else "No",
        "confidence": lambda x: ", ".join(x.unique()),
        "exploit_maturity": lambda x: ", ".join(x.unique()),
        "target_sector": lambda x: ", ".join(x.unique()),
        "target_region": lambda x: ", ".join(x.unique()),
        "summary": lambda x: " | ".join(x.unique()),
    }).reset_index()

    threat_by_cve.columns = [
        "matched_cve", "ti_threat_actor", "ti_campaign_name",
        "ti_ransomware", "ti_confidence", "ti_exploit_maturity",
        "ti_target_sector", "ti_target_region", "ti_summary",
    ]

    merged = merged.merge(
        threat_by_cve,
        left_on="cve",
        right_on="matched_cve",
        how="left",
    )

    # Fill NaN for threat intel matches
    merged["ti_threat_actor"] = merged["ti_threat_actor"].fillna("")
    merged["ti_campaign_name"] = merged["ti_campaign_name"].fillna("")
    merged["ti_ransomware"] = merged["ti_ransomware"].fillna("No")
    merged["has_threat_match"] = merged["ti_threat_actor"] != ""

    return merged
