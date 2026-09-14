# 🛡️ TawasolPay AI-Powered Cyber Risk Assistant

An AI-powered cybersecurity risk assessment system that automatically ingests asset inventory, vulnerability data, threat intelligence, and business context to produce a prioritized, explainable risk picture with NIST SP 800-53 remediation guidance.

Built for the TawasolPay AI Engineer take-home assessment.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key ([get free key](https://makersuite.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-cyber-risk-assistant.git
cd ai-cyber-risk-assistant

# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key (or enter it in the app sidebar)
export GOOGLE_API_KEY="your-key-here"       # Linux/Mac
set GOOGLE_API_KEY=your-key-here            # Windows

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Web Interface                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │ Data Loader   │  │ Risk Scorer  │  │ Report Generator (LLM) ││
│  │              │  │              │  │                        ││
│  │ • Local CSVs │  │ • 7-Factor   │  │ • Gemini 3.6 Flash    ││
│  │ • CISA KEV   │  │   Weighted   │  │ • Grounded in RAG     ││
│  │ • NIST SP    │  │   Scoring    │  │ • Plain-English       ││
│  │   800-53     │  │ • Not just   │  │   explanations        ││
│  └──────┬───────┘  │   CVSS!      │  └─────────┬──────────────┘│
│         │          └──────┬───────┘            │               │
│         │                 │                    │               │
│  ┌──────▼─────────────────▼────────────────────▼──────────────┐│
│  │                   NIST RAG Pipeline                         ││
│  │  sentence-transformers (all-MiniLM-L6-v2) → ChromaDB       ││
│  │  1000+ NIST controls embedded for semantic retrieval        ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Streamlit |
| LLM | Google Gemini 3.6 Flash (free tier) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (persistent, local) |
| Data Processing | pandas |
| External Data | CISA KEV (live), NIST SP 800-53 Rev. 5 (live) |

---

## 📊 How Risk Scoring Works

The system uses a **7-factor weighted composite score** that goes beyond CVSS alone:

| Factor | Weight | Description |
|--------|--------|-------------|
| CVSS Base Score | 15% | Normalized vulnerability severity |
| Internet Exposure | 20% | Internet-facing assets score 1.0, internal score 0.3 |
| Active Exploit / KEV | 20% | CISA KEV listed = 1.0, exploit available = 0.7 |
| Threat Campaign Match | 15% | Matched to active campaign targeting sector/region |
| Ransomware Association | 10% | Campaign or KEV entry linked to ransomware |
| Business Criticality | 10% | Service criticality, revenue impact, compliance scope |
| Missing Controls | 10% | No EDR, no patch, stale asset penalties |

**Key design principle:** A CVSS-10 on an internal dev server ranks *lower* than a CVSS-8 on an internet-exposed payment gateway with an active ransomware campaign.

---

## 📁 Project Structure

```
├── app.py                    # Streamlit main application
├── data_loader.py            # Data ingestion (local + external)
├── risk_scorer.py            # Multi-factor risk scoring engine
├── nist_rag.py               # RAG pipeline (ChromaDB + embeddings)
├── report_generator.py       # LLM report generation (Gemini)
├── requirements.txt          # Python dependencies
├── .streamlit/config.toml    # Streamlit theme configuration
├── Dataset/                  # Provided data files
│   ├── assets.csv
│   ├── vulnerabilities.csv
│   ├── threat_intelligence.csv
│   ├── business_services.csv
│   ├── remediation_guidance.csv
│   └── synthetic_threat_report.md
├── chroma_db/                # ChromaDB persistent storage (auto-created)
└── README.md                 # This file
```

---

## Supporting Question 1: The Data Split

**What I embedded and why:** The NIST SP 800-53 Rev. 5 control catalog (~1,000+ controls) is embedded into ChromaDB using sentence-transformers. These controls are unstructured prose text (descriptions, discussions, supplemental guidance) where the relationship between a vulnerability and the right remediation control is *semantic*, not keyword-based. For example, a VPN authentication bypass vulnerability should match NIST IA-2 (Identification and Authentication) — a relationship that requires understanding meaning, not just matching CVE IDs. The synthetic threat report is also used as unstructured context.

**What I queried as structured data and why:** The CSV files (assets, vulnerabilities, threat intelligence, business services, CISA KEV) are all tabular data with well-defined schemas and explicit foreign keys (asset_id, CVE IDs, business_service names). These are best queried with exact filters and joins — for example, "find all vulnerabilities where `cve = 'CVE-2024-21762'` and the asset has `internet_exposed = 'Yes'`". Embedding these would lose the precision of exact matching and waste compute on data that doesn't benefit from fuzzy semantic search.

---

## Supporting Question 2: Where It Goes Wrong

### 1. Missing KEV entries for synthetic CVEs
If a CVE ID in `vulnerabilities.csv` is synthetic (e.g., `CVE-SYN-2026-0001`), my system will find no match in the CISA KEV catalog and will not flag it as actively exploited — even though the threat intelligence CSV confirms it *is* being weaponized. **Mitigation:** The system also checks the `exploit_available` field in the vulnerabilities CSV and the threat intelligence match, so it still scores these highly. But the `in_kev` flag will be `False`, which could mislead a reviewer into thinking it's not confirmed.

### 2. NIST control retrieval may return a related but not optimal control
The RAG retrieval uses cosine similarity between the risk context and NIST control descriptions. If the vulnerability description is too generic (e.g., "Privilege Escalation"), the system might retrieve AC-6 (Least Privilege) when the better answer is SI-2 (Flaw Remediation) or vice versa. **Mitigation:** The system retrieves the top 3 controls and presents all of them, so the analyst can see alternatives. A future version could use an LLM to re-rank the retrieved controls.

### 3. Deduplication may hide compounding risk
When two vulnerabilities on the same asset are both in the top 5, the system deduplicates by `(asset_id, cve)` but does not aggregate the *combined* risk of multiple vulnerabilities on one asset. For example, VPN appliances A-1005 and A-1006 both have CVE-2024-21762 *and* CVE-2024-55591 — a chained exploit. The system ranks them independently rather than recognizing the chain as a higher compound risk. **Mitigation:** A future version would implement "attack chain detection" that boosts scores when multiple vulnerabilities on the same asset form a known exploit chain.

---

## Supporting Question 3: One Thing I Would Change

If I had another day, the single most important improvement would be **attack chain awareness**. Right now, each vulnerability is scored independently, but the threat report explicitly describes multi-step exploit chains (e.g., CrimsonJackal uses CVE-2024-21762 → CVE-2024-55591; RedMantis chains CVE-SYN-2026-0004 → CVE-2023-22527 → CVE-2023-22515). A vulnerability that is part of a confirmed exploit chain is dramatically more dangerous than one that stands alone, because the attacker doesn't need to find a second entry point. I would parse the threat report and threat intelligence to identify these chains, then boost the composite score for any vulnerability that participates in a chain targeting assets in our inventory. This would make the ranking even more operationally relevant.

---

## 🔑 API Keys

The system uses **Google Gemini 3.6 Flash** (free tier) for generating plain-English explanations. You can:
1. Enter the key in the sidebar when the app loads
2. Set the `GOOGLE_API_KEY` environment variable
3. The app works without a key — you just won't get AI-generated narratives

---

## 📜 License

This project is built for assessment purposes. All threat data is synthetic.
