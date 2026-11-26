# RESEARCHNEXUS

**Interdisciplinary Research Intelligence & Collaboration Analytics**  
*Connecting Ideas. Discovering Possibilities.*  
**Institution:** CHRIST (Deemed to be University), Bengaluru

---

## Project Overview

**RESEARCHNEXUS** is an interactive academic research analytics dashboard built with **Python, Streamlit, Pandas, NumPy, Matplotlib, and Plotly**. It models CHRIST University's multidisciplinary research ecosystem across 10 academic departments and 10 research domains, enabling data-driven analysis of researcher profiles, technical skill supply-demand gaps, project readiness, interdepartmental collaboration networks, and explainable rule-based collaborator recommendations.

> **Scope Note:** RESEARCHNEXUS is a pure analytics and decision-support platform. All filtering, aggregation, dataset generation, and compatibility scoring are implemented locally using standard Python, Pandas, and NumPy—with zero external APIs, databases, or machine learning frameworks.

---

## Key Features & Dashboard Sections

1. **Research Overview:** Institutional KPI cards (total researchers, unique projects, departments, domains, open collaborations, average calculated match score), department distribution, domain donut, Student vs Faculty stacked bars, longitudinal activity trends, executive summary, and **Automated Research Insights**.
2. **Research Landscape:** Section-level department/domain filtering, hierarchical Research Interest treemap, Department-to-Domain Sankey flow, top research topics bar chart, cumulative project trends, underrepresented area detection, and cross-departmental topic overlap matrix.
3. **Department Intelligence:** Department vs Domain heatmap, Role vs Department breakdown, interactive interdepartmental collaboration network graph, deduplicated project participation bars, and interdisciplinary partnership opportunities.
4. **Skills Intelligence:** Most demanded skills (unique project requirements), Available vs Required skill supply-demand comparison, Department vs Technical Skills heatmap, interactive Researcher Skill Radar comparison (2–4 researchers), and scarcity gap table.
5. **Project Analytics:** Strictly deduplicated project portfolio analysis across `Open`, `Active`, `Ongoing`, `Matched`, and `Completed` statuses, Team Size vs Match Score bubble chart, Match Score by Department box plot, activity trends, and a searchable project explorer highlighting projects needing additional skills.
6. **Collaboration Intelligence:** Explainable rule-based matching engine excluding self-matches and duplicates, displaying Top 5 recommended collaborators with match rationales, Matplotlib score distribution histogram, stacked score breakdown bar chart, ego collaboration network graph, and compatibility box plot by department.
7. **Interactive Collaboration Simulator:** Dynamic what-if analysis allowing adjustments to collaboration domain, preferred department, required skills, target team size, and methodology—with live rank/score delta comparisons against baseline matches.

---

## Explainable Match Score Formula

Every compatibility score (0–100) in both `data/researchnexus_dataset.csv` and the live matching engine is calculated deterministically using five weighted rules totaling **100%**:

| Factor | Weight | Logic |
| --- | :---: | --- |
| **Research Interest Similarity** | **35%** | Jaccard overlap of pipe-separated `Research_Interests` (80%) + same `Research_Domain` alignment (20%) |
| **Technical Skill Compatibility** | **30%** | Required-skill coverage (50%) + complementary missing-skill contribution (30%) + shared baseline overlap (20%) |
| **Methodology Compatibility** | **15%** | Domain-informed compatibility matrix across 6 research methodologies (`1.0` for exact match, graded partial compatibility across compatible methods) |
| **Collaboration Domain Fit** | **10%** | Alignment between `Preferred_Collaboration_Domain` and candidate `Research_Domain` (mutual preference rewarded highest) |
| **Department Diversity** | **10%** | Rewards interdisciplinary cross-department connections (`1.0` when matching `Preferred_Department` across departments) |

---

## Dataset Structure

The consolidated dataset at `data/researchnexus_dataset.csv` is generated programmatically by `generate_dataset.py` (default **2,000 records**, configurable via `NUM_RECORDS`, reproducible via `RANDOM_SEED = 42`) with **exactly 20 columns**:

1. `Researcher_ID` (Unique, e.g., `RNX-0001`)
2. `Researcher_Name` (Unique fictional names)
3. `Role` (`UG Student`, `PG Student`, `PhD Scholar`, `Faculty`)
4. `Department` (10 departments)
5. `Programme`
6. `Year_of_Study`
7. `Research_Domain` (10 domains)
8. `Research_Interests` (`|`-separated)
9. `Technical_Skills` (`|`-separated)
10. `Research_Methodology`
11. `Experience_Level`
12. `Preferred_Collaboration_Domain`
13. `Project_Title` (Shared across co-researchers; deduplicated in project metrics)
14. `Project_Description` (Includes embedded `[Initiated: YYYY-MM-DD]` timeline tag)
15. `Required_Skills` (`|`-separated)
16. `Preferred_Department`
17. `Team_Size`
18. `Availability_Hours`
19. `Collaboration_Status` (`Open`, `Active`, `Ongoing`, `Matched`, `Completed`)
20. `Match_Score` (Calculated via `_compute_batch_match_scores` in `generate_dataset.py`)

---

## Project Structure

```text
RESEARCHNEXUS/
│
├── app.py                        # Self-contained Streamlit dashboard, analytics & matching
├── generate_dataset.py           # Reproducible synthetic dataset generator
├── data/
│   └── researchnexus_dataset.csv # Consolidated 20-column dataset (2,000 rows)
├── requirements.txt              # Permitted Python dependencies
└── README.md                     # Documentation & execution guide
```

---

## Installation & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate (or regenerate) the dataset:**
   ```bash
   python generate_dataset.py
   ```

3. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```
