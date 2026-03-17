# Job Application Analysis

An ML pipeline for analyzing job application outcomes at scale. Built to answer a practical question: across 580+ applications over 22 months, what actually predicts whether you get screened, interviewed, or rejected, and what doesn't matter at all?

This isn't a toy project. It processes real application data and surfaces findings that contradict conventional job search wisdom (e.g., cover letters correlating with *lower* interview rates, referrals being 8x more effective than cold applications).

## What It Does

- **Rejection Prediction** — Random Forest and Gradient Boosting classifiers identify which application features most strongly predict rejection
- **Anomaly Detection** — Isolation Forest flags statistically unusual application patterns (outlier companies, atypical timelines)
- **Feature Importance Ranking** — Quantifies the relative impact of referrals, application source, timing, seniority level, resume version, and cover letter usage
- **Cross-Validation** — k-fold validation to ensure model reliability beyond a single train/test split
- **Automated Reporting** — Exports results to Excel with summary metrics, predictions, and anomaly flags

## Key Findings (from 580 applications)

| Metric | Value |
|--------|-------|
| Overall offer rate | 0.17% |
| Referral interview rate | 8.82% vs 1.10% without |
| Cover letter interview rate | 0% vs 1.65% without |
| Applications with referrals | 5.9% |
| Highest-converting seniority | Entry-level (4.17%) |

The data makes a clear case: fewer, higher-quality, referral-backed applications outperform volume.

## Versions

| File | Description |
|------|-------------|
| `jobappanalysis.py` | v1 — Core pipeline: Random Forest, Isolation Forest, feature importance, CLI interface |
| `jobappanalysis_v2.py` | v2 — Adds Gradient Boosting, cross-validation, multi-sheet analysis |
| `jobappanalysis_v3.py` | v3 — Refactored with CLI arguments, proper error handling, modular structure |

## Quick Start

```bash
pip install -r requirements.txt

# Basic usage
python jobappanalysis.py -i your_data.xlsx

# Specify output and skip plots
python jobappanalysis.py -i data.xlsx -o results/ --no-plots

# Use v2 with cross-validation
python jobappanalysis_v2.py -i data.xlsx -o results.xlsx
```

## Input Format

Excel file with yearly sheets (`2024`, `2025`, etc.) containing:

| Column | Description |
|--------|-------------|
| Company | Target company |
| Job Title | Position applied for |
| Application Date | When applied |
| Screening / Interview / Final Round | Pipeline stage dates |
| Rejected | Date rejected (or empty) |
| Interval | Days from application to outcome |
| Application Source | LinkedIn, Indeed, company site, referral, etc. |
| Referral | Yes/No |
| Resume Version | Version identifier (e.g., `24.3.1.3_BASE`) |
| Cover Letter | Yes/No |
| Industry | Sector classification |
| Seniority Level | Entry, Mid, Senior, Manager, Director |

## Output

- `JobAppResults_[timestamp].xlsx` — Summary metrics, per-application predictions, anomaly flags
- `plots/` — Feature importance charts, outcome distributions, industry breakdowns, timeline analysis

## CLI Options

```
-i, --input         Input Excel path (required)
-o, --output        Output path (default: JobAppResults.xlsx)
-s, --sheet         Specific sheet name (default: auto-detect)
--plots-dir         Directory for visualizations (default: plots/)
--no-plots          Skip plot generation
--contamination     Anomaly detection sensitivity (default: 0.1)
```

## Requirements

Python 3.10+

```
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

## License

MIT
