"""
RESEARCHNEXUS — Synthetic Academic Dataset Generator
====================================================
Generates a reproducible, logically consistent 20-column synthetic dataset
representing CHRIST (Deemed to be University), Bengaluru's research ecosystem.

Usage:
    python generate_dataset.py
"""

from __future__ import annotations

import csv
import datetime
import random
from pathlib import Path

import numpy as np
import pandas as pd

METHODOLOGIES = [
    "Quantitative",
    "Qualitative",
    "Mixed Methods",
    "Experimental",
    "Computational & Simulation",
    "Empirical & Survey",
]

_METHOD_COMPAT = {
    ("Quantitative", "Computational & Simulation"): 0.85,
    ("Quantitative", "Experimental"): 0.80,
    ("Quantitative", "Empirical & Survey"): 0.82,
    ("Quantitative", "Mixed Methods"): 0.85,
    ("Quantitative", "Qualitative"): 0.45,
    ("Qualitative", "Mixed Methods"): 0.88,
    ("Qualitative", "Empirical & Survey"): 0.72,
    ("Qualitative", "Experimental"): 0.42,
    ("Qualitative", "Computational & Simulation"): 0.38,
    ("Mixed Methods", "Experimental"): 0.78,
    ("Mixed Methods", "Computational & Simulation"): 0.75,
    ("Mixed Methods", "Empirical & Survey"): 0.86,
    ("Experimental", "Computational & Simulation"): 0.78,
    ("Experimental", "Empirical & Survey"): 0.65,
    ("Computational & Simulation", "Empirical & Survey"): 0.68,
}


def _compute_batch_match_scores(df: pd.DataFrame, top_k: int = 5) -> np.ndarray:
    """Compute each researcher's mean Top-K peer score using the 35/30/15/10/10 rule."""
    n = len(df)
    if n <= 1:
        return np.zeros(n, dtype=float)

    def _split(val: object) -> list[str]:
        return [p.strip() for p in str(val).split("|") if p.strip()]

    def _matrix(series: pd.Series, vocab: list[str]) -> np.ndarray:
        idx = {k: i for i, k in enumerate(vocab)}
        mat = np.zeros((n, len(vocab)), dtype=np.float32)
        for r, val in enumerate(series):
            for tok in _split(val):
                if tok in idx:
                    mat[r, idx[tok]] = 1.0
        return mat

    int_vocab = sorted({t for v in df["Research_Interests"] for t in _split(v)})
    skl_vocab = sorted({t for c in ("Technical_Skills", "Required_Skills") for v in df[c] for t in _split(v)})
    m_int = _matrix(df["Research_Interests"], int_vocab)
    m_skl = _matrix(df["Technical_Skills"], skl_vocab)
    m_req = _matrix(df["Required_Skills"], skl_vocab)

    i_inter = m_int @ m_int.T
    i_cnt = m_int.sum(axis=1, keepdims=True)
    i_union = i_cnt + i_cnt.T - i_inter
    jaccard = np.divide(i_inter, i_union, out=np.zeros_like(i_inter), where=i_union > 0)
    domains = df["Research_Domain"].astype(str).to_numpy()
    same_dom = (domains[:, None] == domains[None, :]).astype(np.float32)
    int_mat = np.minimum(1.0, 0.80 * jaccard + 0.20 * same_dom) * 100.0

    req_cov = np.divide(m_req @ m_skl.T, np.maximum(1.0, m_req.sum(axis=1, keepdims=True)))
    missing = np.clip(m_req - m_skl, 0.0, 1.0)
    miss_cnt = missing.sum(axis=1, keepdims=True)
    skl_inter = m_skl @ m_skl.T
    skl_cnt = m_skl.sum(axis=1, keepdims=True)
    comp_ratio = np.where(
        miss_cnt > 0,
        np.divide(missing @ m_skl.T, np.maximum(1.0, miss_cnt)),
        np.minimum(1.0, np.divide(skl_cnt.T - skl_inter, np.maximum(1.0, skl_cnt.T))),
    )
    skl_union = skl_cnt + skl_cnt.T - skl_inter
    shared_skl = np.divide(skl_inter, skl_union, out=np.zeros_like(skl_inter), where=skl_union > 0)
    skl_mat = np.minimum(1.0, 0.50 * req_cov + 0.30 * comp_ratio + 0.20 * shared_skl) * 100.0

    methods = df["Research_Methodology"].astype(str).tolist()
    uniq_m = list(dict.fromkeys(methods))
    m_idx = {m: i for i, m in enumerate(uniq_m)}
    lookup = np.array(
        [[1.0 if a == b else _METHOD_COMPAT.get((a, b), _METHOD_COMPAT.get((b, a), 0.50)) for b in uniq_m] for a in uniq_m],
        dtype=np.float32,
    )
    ids = np.array([m_idx[m] for m in methods])
    meth_mat = lookup[ids[:, None], ids[None, :]] * 100.0

    pdom = df["Preferred_Collaboration_Domain"].astype(str).to_numpy()
    fwd = pdom[:, None] == domains[None, :]
    rev = domains[:, None] == pdom[None, :]
    dom_mat = np.where(fwd & rev, 1.0, np.where(fwd, 0.85, np.where(rev, 0.65, np.where(same_dom > 0, 0.45, 0.15)))) * 100.0

    depts = df["Department"].astype(str).to_numpy()
    pdept = df["Preferred_Department"].astype(str).to_numpy()
    diff_d = depts[:, None] != depts[None, :]
    match_pd = pdept[:, None] == depts[None, :]
    dept_mat = np.where(diff_d & match_pd, 1.0, np.where(diff_d, 0.85, np.where(match_pd, 0.55, 0.35))) * 100.0

    total = 0.35 * int_mat + 0.30 * skl_mat + 0.15 * meth_mat + 0.10 * dom_mat + 0.10 * dept_mat
    np.fill_diagonal(total, -1.0)
    k = min(top_k, n - 1)
    return np.round(np.partition(total, -k, axis=1)[:, -k:].mean(axis=1), 2)

# Configuration (adjust NUM_RECORDS to scale the dataset)
NUM_RECORDS = 2000
RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "researchnexus_dataset.csv"

DATASET_COLUMNS = [
    "Researcher_ID", "Researcher_Name", "Role", "Department", "Programme",
    "Year_of_Study", "Research_Domain", "Research_Interests", "Technical_Skills",
    "Research_Methodology", "Experience_Level", "Preferred_Collaboration_Domain",
    "Project_Title", "Project_Description", "Required_Skills",
    "Preferred_Department", "Team_Size", "Availability_Hours",
    "Collaboration_Status", "Match_Score",
]

DEPARTMENTS = {
    "Computer Science": {
        "weight": 0.18,
        "programmes": {"UG Student": ["B.Sc. Computer Science", "BCA", "B.Tech. CSE"], "PG Student": ["MCA", "M.Sc. Data Science"], "PhD Scholar": ["Ph.D. Computer Science"], "Faculty": ["Faculty - Computer Science"]},
        "domains": {"Artificial Intelligence": 0.34, "Data Analytics": 0.25, "Cybersecurity": 0.18, "Healthcare": 0.11, "Education": 0.07, "Sustainability": 0.05},
        "partners": ["Statistics", "Mathematics", "Life Sciences", "Management", "Psychology"],
        "skills": ["Python", "SQL", "Git", "Data Visualization"],
    },
    "Management": {
        "weight": 0.13,
        "programmes": {"UG Student": ["BBA", "BBA Business Analytics"], "PG Student": ["MBA", "MBA Business Analytics"], "PhD Scholar": ["Ph.D. Management"], "Faculty": ["Faculty - Management"]},
        "domains": {"Finance": 0.28, "Data Analytics": 0.26, "Sustainability": 0.22, "Social Sciences": 0.12, "Education": 0.07, "Artificial Intelligence": 0.05},
        "partners": ["Commerce", "Economics", "Computer Science", "Statistics", "Social Sciences"],
        "skills": ["Excel Analytics", "Business Strategy", "Data Visualization", "Academic Writing"],
    },
    "Life Sciences": {
        "weight": 0.12,
        "programmes": {"UG Student": ["B.Sc. Biotechnology", "B.Sc. Life Sciences"], "PG Student": ["M.Sc. Biotechnology", "M.Sc. Life Sciences"], "PhD Scholar": ["Ph.D. Life Sciences"], "Faculty": ["Faculty - Life Sciences"]},
        "domains": {"Biotechnology": 0.34, "Healthcare": 0.30, "Environmental Science": 0.20, "Sustainability": 0.10, "Data Analytics": 0.06},
        "partners": ["Computer Science", "Statistics", "Psychology", "Social Sciences"],
        "skills": ["Bioinformatics", "Experimental Design", "PCR & Wet Lab", "Academic Writing"],
    },
    "Commerce": {
        "weight": 0.11,
        "programmes": {"UG Student": ["B.Com.", "B.Com. Honours", "B.Com. FinTech"], "PG Student": ["M.Com.", "M.Com. International Finance"], "PhD Scholar": ["Ph.D. Commerce"], "Faculty": ["Faculty - Commerce"]},
        "domains": {"Finance": 0.42, "Sustainability": 0.22, "Data Analytics": 0.18, "Cybersecurity": 0.10, "Education": 0.08},
        "partners": ["Economics", "Management", "Statistics", "Computer Science"],
        "skills": ["Financial Modelling", "Excel Analytics", "ESG Auditing", "SQL"],
    },
    "Psychology": {
        "weight": 0.10,
        "programmes": {"UG Student": ["B.Sc. Psychology", "B.A. Psychology Honours"], "PG Student": ["M.Sc. Clinical Psychology", "M.Sc. Counselling Psychology"], "PhD Scholar": ["Ph.D. Psychology"], "Faculty": ["Faculty - Psychology"]},
        "domains": {"Healthcare": 0.34, "Social Sciences": 0.32, "Education": 0.22, "Artificial Intelligence": 0.07, "Data Analytics": 0.05},
        "partners": ["Life Sciences", "Social Sciences", "Computer Science", "Media Studies", "Statistics"],
        "skills": ["SPSS", "Survey Design", "Qualitative Coding (NVivo)", "Experimental Design"],
    },
    "Economics": {
        "weight": 0.09,
        "programmes": {"UG Student": ["B.A. Economics", "B.Sc. Economics & Analytics"], "PG Student": ["M.A. Applied Economics", "M.Sc. Quantitative Economics"], "PhD Scholar": ["Ph.D. Economics"], "Faculty": ["Faculty - Economics"]},
        "domains": {"Finance": 0.32, "Sustainability": 0.24, "Social Sciences": 0.22, "Data Analytics": 0.16, "Environmental Science": 0.06},
        "partners": ["Statistics", "Commerce", "Management", "Social Sciences", "Mathematics"],
        "skills": ["Econometrics", "R", "Stata", "Time Series Forecasting"],
    },
    "Statistics": {
        "weight": 0.08,
        "programmes": {"UG Student": ["B.Sc. Data Science & Statistics"], "PG Student": ["M.Sc. Statistics", "M.Sc. Biostatistics"], "PhD Scholar": ["Ph.D. Statistics"], "Faculty": ["Faculty - Statistics"]},
        "domains": {"Data Analytics": 0.35, "Artificial Intelligence": 0.22, "Healthcare": 0.18, "Finance": 0.15, "Environmental Science": 0.10},
        "partners": ["Computer Science", "Mathematics", "Economics", "Life Sciences", "Psychology"],
        "skills": ["R", "Statistical Modelling", "Python", "Bayesian Inference"],
    },
    "Mathematics": {
        "weight": 0.07,
        "programmes": {"UG Student": ["B.Sc. Mathematics"], "PG Student": ["M.Sc. Mathematics"], "PhD Scholar": ["Ph.D. Mathematics"], "Faculty": ["Faculty - Mathematics"]},
        "domains": {"Artificial Intelligence": 0.28, "Data Analytics": 0.26, "Cybersecurity": 0.20, "Finance": 0.16, "Education": 0.10},
        "partners": ["Computer Science", "Statistics", "Economics"],
        "skills": ["Optimization", "Python", "MATLAB", "Statistical Modelling"],
    },
    "Social Sciences": {
        "weight": 0.06,
        "programmes": {"UG Student": ["B.A. Sociology & Political Science"], "PG Student": ["M.A. Public Policy", "MSW"], "PhD Scholar": ["Ph.D. Social Sciences"], "Faculty": ["Faculty - Social Sciences"]},
        "domains": {"Social Sciences": 0.42, "Sustainability": 0.22, "Education": 0.20, "Healthcare": 0.10, "Environmental Science": 0.06},
        "partners": ["Psychology", "Media Studies", "Economics", "Management"],
        "skills": ["Qualitative Coding (NVivo)", "Survey Design", "Academic Writing", "Policy Analysis"],
    },
    "Media Studies": {
        "weight": 0.06,
        "programmes": {"UG Student": ["B.A. Media & Communication"], "PG Student": ["M.A. Media & Communication Studies"], "PhD Scholar": ["Ph.D. Media Studies"], "Faculty": ["Faculty - Media Studies"]},
        "domains": {"Social Sciences": 0.38, "Education": 0.25, "Data Analytics": 0.17, "Artificial Intelligence": 0.12, "Sustainability": 0.08},
        "partners": ["Psychology", "Social Sciences", "Computer Science"],
        "skills": ["Content Analysis", "Survey Design", "Data Visualization", "Digital Storytelling"],
    },
}

DOMAIN_CATALOG = {
    "Artificial Intelligence": {
        "interests": ["Natural Language Processing", "Computer Vision", "Explainable AI", "Generative AI Governance", "Predictive Modelling", "AI in Education"],
        "skills": ["Python", "Machine Learning", "Deep Learning", "NLP", "Statistical Modelling", "Data Visualization", "Cloud Computing"],
        "methods": ["Computational & Simulation", "Experimental", "Quantitative", "Mixed Methods"],
        "themes": ["Explainable Neural Diagnostics for Clinical Decision Support", "Multilingual NLP Framework for Regional Knowledge Retrieval", "Fairness-Aware Predictive Analytics for Campus Resource Optimization", "Automated Misinformation Detection in Digital News Ecosystems", "Adaptive Intelligent Tutoring Systems for STEM Cohorts"],
    },
    "Data Analytics": {
        "interests": ["Predictive Modelling", "Visual Analytics", "Time Series Forecasting", "Decision Intelligence", "Big Data Governance", "Behavioural Data Mining"],
        "skills": ["Python", "R", "SQL", "Power BI", "Tableau", "Statistical Modelling", "Time Series Forecasting", "Excel Analytics"],
        "methods": ["Quantitative", "Empirical & Survey", "Computational & Simulation", "Mixed Methods"],
        "themes": ["Unified Institutional Learning Analytics for Student Retention", "High-Frequency Demand Forecasting for Urban Sustainable Mobility", "Cross-Sector Spatial Data Observatory for Smart Cities", "Socio-Economic Vulnerability Index Using Longitudinal Data", "Real-Time Telemetry Analytics for Renewable Microgrid Efficiency"],
    },
    "Healthcare": {
        "interests": ["Public Health Analytics", "Mental Health Informatics", "Clinical Decision Support", "Behavioral Neuroscience", "Epidemiological Modelling", "Preventive Digital Health"],
        "skills": ["Statistical Modelling", "R", "SPSS", "Python", "Experimental Design", "Survey Design", "Bioinformatics"],
        "methods": ["Empirical & Survey", "Experimental", "Mixed Methods", "Quantitative"],
        "themes": ["Predictive Screening of Stress and Burnout Among University Cohorts", "Epidemiological Risk Mapping for Urban Vector-Borne Outbreaks", "Multimodal Mental Wellness Intervention Using Digital Biomarkers", "Hospital Readmission Risk Stratification via Interpretable Models", "Telehealth Accessibility Assessment in Semi-Urban Clinics"],
    },
    "Finance": {
        "interests": ["Financial Risk Modelling", "FinTech Inclusion", "ESG Investing Analytics", "Algorithmic Market Microstructure", "Behavioural Finance", "Digital Payments Security"],
        "skills": ["Financial Modelling", "Econometrics", "Python", "R", "Time Series Forecasting", "Excel Analytics", "SQL"],
        "methods": ["Quantitative", "Empirical & Survey", "Computational & Simulation"],
        "themes": ["ESG Disclosure Quality and Cost of Capital in Mid-Cap Firms", "Credit Scoring Frameworks for Micro-Enterprise Financial Inclusion", "Volatility Spillover Analysis Across Emerging Green Bond Markets", "Behavioral Biases in Retail FinTech Adoption Among Young Adults", "Fraud Anomaly Detection in Real-Time UPI Transaction Streams"],
    },
    "Sustainability": {
        "interests": ["ESG Analytics", "Circular Economy Models", "Sustainable Supply Chains", "Urban Climate Resilience", "Green Campus Operations", "Renewable Energy Policy"],
        "skills": ["ESG Auditing", "GIS & Spatial Analysis", "Python", "Excel Analytics", "Survey Design", "Policy Analysis", "Data Visualization"],
        "methods": ["Mixed Methods", "Empirical & Survey", "Quantitative", "Qualitative"],
        "themes": ["Carbon Footprint Accounting and Net-Zero Roadmap for Higher Education", "Circular Supply Chain Optimization in Regional Agri-Food Logistics", "Consumer Adoption Drivers for Zero-Waste Urban Retail", "Corporate ESG Greenwashing Detection Using Textual Mining", "Community Solar Microgrid Feasibility for Peri-Urban Settlements"],
    },
    "Social Sciences": {
        "interests": ["Public Policy Analytics", "Digital Media Literacy", "Urban Inclusion & Equity", "Civic Participation Studies", "Workplace Well-Being", "Gender & Technology"],
        "skills": ["Qualitative Coding (NVivo)", "Survey Design", "Policy Analysis", "Content Analysis", "SPSS", "Academic Writing", "Digital Storytelling"],
        "methods": ["Qualitative", "Mixed Methods", "Empirical & Survey"],
        "themes": ["Algorithmic Echo Chambers and Youth Civic Discourse", "Socio-Economic Impact Assessment of Gig Economy Labour Policies", "Participatory Evaluation of Urban Affordable Housing Interventions", "Digital Literacy Bridges for First-Generation University Entrants", "Visual Framing of Climate Justice in Indian Digital Journalism"],
    },
    "Education": {
        "interests": ["AI in Education", "Curriculum Innovation", "Student Mental Well-Being", "Assessment Analytics", "Digital Pedagogy", "Interdisciplinary Learning"],
        "skills": ["Survey Design", "SPSS", "Data Visualization", "Qualitative Coding (NVivo)", "Academic Writing", "Python", "Experimental Design"],
        "methods": ["Mixed Methods", "Empirical & Survey", "Qualitative", "Experimental"],
        "themes": ["Generative AI Literacy Frameworks for Postgraduate Capstones", "Longitudinal Study of Peer-Mentored Interdisciplinary Studio Learning", "Early Warning Indicators for Academic Disengagement in UG Cohorts", "Accessibility Audit of Multimodal Digital Textbooks", "Gamified Quantitative Reasoning Labs for Non-STEM Undergraduates"],
    },
    "Cybersecurity": {
        "interests": ["Network Intrusion Detection", "Privacy-Preserving Analytics", "Cloud Security Architecture", "Adversarial Machine Learning", "Digital Forensics", "Zero-Trust Governance"],
        "skills": ["Network Security", "Penetration Testing", "Python", "Cloud Computing", "Docker", "Machine Learning", "SQL"],
        "methods": ["Experimental", "Computational & Simulation", "Quantitative"],
        "themes": ["Federated Intrusion Detection for Privacy-Sensitive Campus Networks", "Adversarial Robustness Benchmarking of Fraud Classifiers", "Automated Vulnerability Triage in Containerized Microservices", "Phishing Resilience Training Efficacy Across University Departments", "Lightweight Cryptographic Protocols for Healthcare IoT Telemetry"],
    },
    "Biotechnology": {
        "interests": ["Genomic Data Mining", "Microbial Bioprocessing", "Drug Target Discovery", "Bioinformatics Pipelines", "Agricultural Biotech", "Molecular Diagnostics"],
        "skills": ["Bioinformatics", "PCR & Wet Lab", "Experimental Design", "R", "Python", "Statistical Modelling", "Academic Writing"],
        "methods": ["Experimental", "Computational & Simulation", "Quantitative"],
        "themes": ["In-Silico Screening of Phytochemical Inhibitors for Antimicrobial Targets", "Transcriptomic Biomarker Discovery Using Reproducible R Pipelines", "Bioremediation Potential of Native Microbial Consortia in Effluents", "CRISPR Off-Target Risk Scoring via Sequence Feature Modelling", "Metagenomic Profiling of Urban Wetland Soil Microbiomes"],
    },
    "Environmental Science": {
        "interests": ["Urban Heat Island Mapping", "Air Quality Modelling", "Watershed Conservation", "Biodiversity Informatics", "Climate Risk Forecasting", "Environmental Toxicology"],
        "skills": ["GIS & Spatial Analysis", "R", "Python", "Statistical Modelling", "Experimental Design", "Time Series Forecasting", "Data Visualization"],
        "methods": ["Quantitative", "Experimental", "Mixed Methods", "Computational & Simulation"],
        "themes": ["Spatial-Temporal Particulate Matter Forecasting in Bengaluru Corridors", "Urban Lake Eutrophication Monitoring via Remote Sensing Indices", "Microplastic Bioaccumulation Assessment in Freshwater Systems", "Canopy Cover Dynamics and Microclimate Cooling Across Campuses", "Hydrological Modelling of Rainwater Harvesting Potential"],
    },
}

FIRST_NAMES = [
    "Aarav", "Aditi", "Aditya", "Aishwarya", "Akash", "Akhil", "Amrita", "Ananya", "Anirudh", "Anjali",
    "Ankur", "Anushka", "Arjun", "Arnav", "Aryan", "Ashwin", "Avantika", "Ayush", "Bhavana", "Chaitanya",
    "Darshan", "Deepa", "Deepak", "Devika", "Dhanush", "Divya", "Diya", "Esha", "Gaurav", "Gayathri",
    "Gokul", "Harini", "Harish", "Harshita", "Ishaan", "Ishita", "Jahnvi", "Jayant", "Kajal", "Karan",
    "Karthik", "Kavya", "Keerthi", "Kiran", "Krish", "Kriti", "Lakshmi", "Lavanya", "Madhav", "Mahesh",
    "Malavika", "Manish", "Manoj", "Meera", "Meghana", "Mihir", "Mohit", "Nandini", "Nikhil", "Nisha",
    "Nithin", "Nivedita", "Pallavi", "Parth", "Pooja", "Pranav", "Prashanth", "Pratik", "Preethi", "Priya",
    "Priyanka", "Rahul", "Rajesh", "Rakesh", "Ramya", "Rashi", "Rashmi", "Rhea", "Rishabh", "Ritika",
    "Rohan", "Rohit", "Roshni", "Sagar", "Sahana", "Sakshi", "Samarth", "Sameer", "Sana", "Sandeep",
    "Sanjana", "Sanjay", "Sarthak", "Sathish", "Shalini", "Shashank", "Shikha", "Shilpa", "Shivam", "Shreya",
    "Shruti", "Siddharth", "Simran", "Sneha", "Soham", "Sonali", "Sonia", "Sooraj", "Soumya", "Sreeja",
    "Srinivas", "Srishti", "Subham", "Sudha", "Suhas", "Sujata", "Suman", "Sumit", "Sunitha", "Suraj",
    "Swathi", "Tanvi", "Tarun", "Tejas", "Trisha", "Tushar", "Uday", "Vaibhav", "Vaishnavi", "Varun",
    "Vedant", "Venkatesh", "Vidya", "Vignesh", "Vikram", "Vinay", "Vineeth", "Vishal", "Vivek", "Yash",
    "Clara", "Daniel", "Elena", "Hannah", "Jacob", "Maria", "Nathan", "Rachel", "Samuel", "Thomas",
]

LAST_NAMES = [
    "Agarwal", "Ahuja", "Anand", "Babu", "Balakrishnan", "Banerjee", "Bansal", "Bhat", "Bhattacharya", "Biswas",
    "Chakraborty", "Chatterjee", "Chauhan", "Chopra", "Das", "Desai", "Deshmukh", "Dutta", "Fernandes", "Gandhi",
    "Garg", "George", "Ghosh", "Gill", "Goel", "Gopal", "Gowda", "Gupta", "Hegde", "Iyer", "Jain", "Joseph",
    "Joshi", "Kulkarni", "Kumar", "Kurian", "Kapoor", "Kaushik", "Khan", "Khanna", "Kohli", "Krishnan",
    "Malhotra", "Mathew", "Mehta", "Menon", "Mishra", "Mittal", "Mohan", "Mukherjee", "Murthy", "Nair",
    "Nambiar", "Nanda", "Natarajan", "Nayak", "Negi", "Pai", "Panda", "Pandey", "Panicker", "Patel", "Patil",
    "Paul", "Pillai", "Prabhu", "Prakash", "Prasad", "Raghavan", "Raj", "Rajan", "Ramachandran", "Rao",
    "Reddy", "Roy", "Saha", "Saxena", "Sen", "Sengupta", "Sethi", "Shah", "Sharma", "Shetty", "Shukla",
    "Singh", "Sinha", "Srinivasan", "Subramanian", "Sundaram", "Swamy", "Thakur", "Thomas", "Tiwari",
    "Tripathi", "Unnikrishnan", "Upadhyay", "Varma", "Varghese", "Venkataraman", "Verma", "Vijayan", "Yadav",
]

ROLES = ["UG Student", "PG Student", "PhD Scholar", "Faculty"]
ROLE_WEIGHTS = [0.34, 0.30, 0.14, 0.22]
STATUSES = ["Open", "Active", "Ongoing", "Matched", "Completed"]
STATUS_WEIGHTS = [0.28, 0.25, 0.20, 0.15, 0.12]
COHORT_TAGS = ["Phase I", "Phase II", "Bengaluru Study", "Campus Pilot", "Cross-Faculty Track", "Regional Cohort", "Lab Track", "Field Study"]


def _weighted_pick(rng: random.Random, weights_map: dict[str, float]) -> str:
    return rng.choices(list(weights_map.keys()), weights=list(weights_map.values()), k=1)[0]


def _generate_unique_names(rng: random.Random, roles: list[str]) -> list[str]:
    used, names = set(), []
    initials = list("ABCDEFGHJKLMNPRSTV")
    for role in roles:
        for attempt in range(400):
            first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
            prefix = f"{rng.choice(['Dr.', 'Prof.'])} " if role == "Faculty" else ""
            mid = f"{rng.choice(initials)}. " if attempt >= 120 else ""
            candidate = f"{prefix}{first} {mid}{last}"
            if candidate not in used:
                used.add(candidate)
                names.append(candidate)
                break
        else:
            fallback = f"{first} {last} {len(names)+1}"
            used.add(fallback)
            names.append(fallback)
    return names


def _build_projects(rng: random.Random, total_researchers: int) -> list[dict]:
    dept_names = list(DEPARTMENTS.keys())
    dept_weights = [DEPARTMENTS[d]["weight"] for d in dept_names]
    start_date = datetime.date(2024, 1, 10)
    span_days = (datetime.date(2026, 10, 25) - start_date).days

    projects, used_titles = [], set()
    assigned_total, idx = 0, 1
    while assigned_total < total_researchers:
        lead_dept = rng.choices(dept_names, weights=dept_weights, k=1)[0]
        d_meta = DEPARTMENTS[lead_dept]
        domain = _weighted_pick(rng, d_meta["domains"])
        dom_meta = DOMAIN_CATALOG[domain]

        theme = rng.choice(dom_meta["themes"])
        title = theme if theme not in used_titles else f"{theme} ({rng.choice(COHORT_TAGS)})"
        if title in used_titles:
            title = f"{theme} — Track {idx:03d}"
        used_titles.add(title)

        pref_dept = rng.choice(d_meta["partners"]) if rng.random() < 0.65 else lead_dept
        team_size = rng.choices([2, 3, 4, 5, 6], weights=[0.14, 0.30, 0.30, 0.18, 0.08], k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        members = rng.randint(1, max(1, team_size - 1)) if status == "Open" else rng.randint(2, team_size)
        members = min(members, total_researchers - assigned_total)

        skill_pool = list(dict.fromkeys(dom_meta["skills"] + d_meta["skills"]))
        req_skills = rng.sample(skill_pool, k=min(rng.randint(3, 5), len(skill_pool)))
        init_date = start_date + datetime.timedelta(days=rng.randint(0, span_days))
        methodology = rng.choice(dom_meta["methods"])
        bridge = f"Bridges {lead_dept} and {pref_dept}." if pref_dept != lead_dept else f"Anchored in {lead_dept}."
        desc = f"{bridge} Uses {methodology.lower()} methods with {', '.join(req_skills[:3])}. [Initiated: {init_date:%Y-%m-%d}]"

        projects.append({
            "Project_Title": title, "Project_Description": desc, "Lead_Department": lead_dept,
            "Domain": domain, "Methodology": methodology, "Required_Skills": req_skills,
            "Preferred_Department": pref_dept, "Team_Size": team_size,
            "Collaboration_Status": status, "Actual_Members": members,
        })
        assigned_total += members
        idx += 1
    return projects


def generate_dataset(num_records: int = NUM_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)
    projects = _build_projects(rng, num_records)

    assignments, roles = [], []
    for proj in projects:
        for slot in range(proj["Actual_Members"]):
            role = rng.choices(["Faculty", "PhD Scholar"], weights=[0.6, 0.4], k=1)[0] if (slot == 0 and rng.random() < 0.42) else rng.choices(ROLES, weights=ROLE_WEIGHTS, k=1)[0]
            roles.append(role)
            assignments.append((proj, slot, role))

    names = _generate_unique_names(rng, roles)
    rows = []
    for i, ((proj, slot, role), name) in enumerate(zip(assignments, names), start=1):
        lead_dept, pref_dept = proj["Lead_Department"], proj["Preferred_Department"]
        dept = lead_dept if (slot == 0 or pref_dept == lead_dept or rng.random() < 0.55) else pref_dept
        d_meta = DEPARTMENTS[dept]
        prog = rng.choice(d_meta["programmes"][role])

        if role == "Faculty":
            yr, exp, hrs = "Faculty", rng.choices(["Advanced", "Expert"], weights=[0.35, 0.65], k=1)[0], rng.randint(4, 12)
        elif role == "PhD Scholar":
            yr, exp, hrs = rng.choice(["1st Year", "2nd Year", "3rd Year", "4th Year"]), rng.choices(["Intermediate", "Advanced", "Expert"], weights=[0.25, 0.55, 0.20], k=1)[0], rng.randint(10, 22)
        elif role == "PG Student":
            yr, exp, hrs = rng.choice(["1st Year", "2nd Year"]), rng.choices(["Beginner", "Intermediate", "Advanced"], weights=[0.20, 0.55, 0.25], k=1)[0], rng.randint(8, 18)
        else:
            yr = rng.choice(["1st Year", "2nd Year", "3rd Year", "4th Year"] if ("B.Tech" in prog or "Honours" in prog) else ["1st Year", "2nd Year", "3rd Year"])
            exp = rng.choices(["Beginner", "Intermediate", "Advanced"], weights=[0.45, 0.43, 0.12], k=1)[0]
            hrs = rng.randint(5, 14)

        domain = proj["Domain"] if rng.random() < 0.78 else _weighted_pick(rng, d_meta["domains"])
        dom_meta = DOMAIN_CATALOG[domain]

        int_pool = list(dict.fromkeys(dom_meta["interests"] + DOMAIN_CATALOG[proj["Domain"]]["interests"]))
        interests = rng.sample(int_pool, k=min(rng.randint(2, 4), len(int_pool)))

        req_skills = proj["Required_Skills"]
        cov_k = rng.randint(2, min(len(req_skills), 4)) if exp in ("Advanced", "Expert") else rng.randint(1, min(len(req_skills), 2))
        skills = rng.sample(req_skills, k=cov_k)
        extra = [s for s in list(dict.fromkeys(d_meta["skills"] + dom_meta["skills"])) if s not in skills]
        target_sk = rng.randint(3, 5)
        while len(skills) < target_sk and extra:
            pick = rng.choice(extra)
            skills.append(pick)
            extra.remove(pick)

        methodology = proj["Methodology"] if rng.random() < 0.70 else rng.choice(dom_meta["methods"] + [rng.choice(METHODOLOGIES)])
        other_doms = [d for d in DOMAIN_CATALOG if d != domain]
        pref_dom = rng.choice(other_doms) if rng.random() < 0.62 else domain

        rows.append({
            "Researcher_ID": f"RNX-{i:04d}", "Researcher_Name": name, "Role": role,
            "Department": dept, "Programme": prog, "Year_of_Study": yr,
            "Research_Domain": domain, "Research_Interests": "|".join(interests),
            "Technical_Skills": "|".join(skills), "Research_Methodology": methodology,
            "Experience_Level": exp, "Preferred_Collaboration_Domain": pref_dom,
            "Project_Title": proj["Project_Title"], "Project_Description": proj["Project_Description"],
            "Required_Skills": "|".join(req_skills), "Preferred_Department": pref_dept,
            "Team_Size": proj["Team_Size"], "Availability_Hours": hrs,
            "Collaboration_Status": proj["Collaboration_Status"], "Match_Score": 0.0,
        })

    df = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    df["Match_Score"] = _compute_batch_match_scores(df, top_k=5)
    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(NUM_RECORDS, RANDOM_SEED)
    df.to_csv(OUTPUT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Generated {len(df)} records ({len(df.columns)} cols, {df['Project_Title'].nunique()} unique projects) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
