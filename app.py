"""
RESEARCHNEXUS - Interdisciplinary Research Intelligence & Collaboration Analytics
CHRIST (Deemed to be University), Bengaluru
"""

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RESEARCHNEXUS | CHRIST University", page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).resolve().parent / "data" / "researchnexus_dataset.csv"
METHODOLOGIES = ["Quantitative", "Qualitative", "Mixed Methods", "Experimental", "Computational & Simulation", "Empirical & Survey"]
WEIGHTS = {"interest": 0.35, "skill": 0.30, "methodology": 0.15, "domain": 0.10, "department": 0.10}
METHOD_PAIRS = {
    ("Quantitative", "Computational & Simulation"): 0.85, ("Quantitative", "Experimental"): 0.80,
    ("Quantitative", "Empirical & Survey"): 0.82, ("Quantitative", "Mixed Methods"): 0.85,
    ("Quantitative", "Qualitative"): 0.45, ("Qualitative", "Mixed Methods"): 0.88,
    ("Qualitative", "Empirical & Survey"): 0.72, ("Qualitative", "Experimental"): 0.42,
    ("Qualitative", "Computational & Simulation"): 0.38, ("Mixed Methods", "Experimental"): 0.78,
    ("Mixed Methods", "Computational & Simulation"): 0.75, ("Mixed Methods", "Empirical & Survey"): 0.86,
    ("Experimental", "Computational & Simulation"): 0.78, ("Experimental", "Empirical & Survey"): 0.65,
    ("Computational & Simulation", "Empirical & Survey"): 0.68,
}
SECTIONS = ["Overview", "Landscape", "Departments", "Skills", "Projects", "Matching", "Simulator"]
PALETTE = ["#2E4A68", "#78998B", "#A99BC5", "#D5B77B", "#5B84AA", "#527466", "#7C6BA0", "#B89146", "#5E7B8C", "#9E8C7A"]
STATUS_COLORS = {"Open": "#D5B77B", "Active": "#78998B", "Ongoing": "#2E4A68", "Matched": "#A99BC5", "Completed": "#94A3B8"}


# --- Data Loading & Helper Functions ---
def split_pipes(val):
    if pd.isna(val) or str(val).strip().lower() in ("", "nan"):
        return []
    return list(dict.fromkeys(x.strip() for x in str(val).split("|") if x.strip()))


@st.cache_data
def load_data(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found at '{path}'. Run generate_dataset.py first.")
    df = pd.read_csv(path).drop_duplicates(subset=["Researcher_ID"]).reset_index(drop=True)
    for col in df.columns:
        if col not in ("Team_Size", "Availability_Hours", "Match_Score"):
            df[col] = df[col].fillna("Unspecified").astype(str).str.strip()
    df["Team_Size"] = pd.to_numeric(df["Team_Size"], errors="coerce").fillna(3).astype(int)
    df["Availability_Hours"] = pd.to_numeric(df["Availability_Hours"], errors="coerce").fillna(8).astype(int)
    df["Match_Score"] = pd.to_numeric(df["Match_Score"], errors="coerce").fillna(0.0).round(2)
    df["Role_Group"] = np.where(df["Role"] == "Faculty", "Faculty", "Student")
    df["Project_Date"] = pd.to_datetime(df["Project_Description"].str.extract(r"\[Initiated:\s*(\d{4}-\d{2}-\d{2})\]", expand=False), errors="coerce")
    df["Project_Month"] = df["Project_Date"].dt.to_period("M").astype(str).fillna("Unknown")
    return df


@st.cache_data
def get_unique_projects(df):
    if df.empty:
        return pd.DataFrame()
    rows = []
    for title, grp in df.groupby("Project_Title", sort=False):
        first = grp.iloc[0]
        depts = sorted(grp["Department"].unique().tolist())
        req = split_pipes(first["Required_Skills"])
        team_sk = {s for v in grp["Technical_Skills"] for s in split_pipes(v)}
        missing = [s for s in req if s not in team_sk]
        assigned, team_size = int(grp["Researcher_ID"].nunique()), int(first["Team_Size"])
        p_dept, pref_dept = grp["Department"].mode().iloc[0], str(first["Preferred_Department"])
        rows.append({
            "Project_Title": title, "Primary_Department": p_dept, "Departments_Involved": ", ".join(depts),
            "Primary_Domain": grp["Research_Domain"].mode().iloc[0], "Collaboration_Status": str(first["Collaboration_Status"]),
            "Preferred_Department": pref_dept, "Team_Size": team_size, "Assigned_Researchers": assigned,
            "Open_Slots": max(0, team_size - assigned), "Mean_Match_Score": round(float(grp["Match_Score"].mean()), 2),
            "Required_Skills": str(first["Required_Skills"]), "Required_Skills_Count": len(req),
            "Missing_Skills": ", ".join(missing) if missing else "Fully Covered",
            "Has_Skill_Gap": len(missing) > 0, "Project_Date": first["Project_Date"],
            "Project_Month": first["Project_Month"], "Is_Interdisciplinary": len(depts) > 1 or pref_dept != p_dept,
        })
    return pd.DataFrame(rows)


def skill_supply_demand(df, proj_df):
    sup = Counter(s for v in df["Technical_Skills"] for s in set(split_pipes(v)))
    dem = Counter(s for v in proj_df["Required_Skills"] for s in set(split_pipes(v)))
    n_res, n_proj = max(1, len(df)), max(1, len(proj_df))
    rows = []
    for sk in sorted(set(sup) | set(dem)):
        s_cnt, d_cnt = sup.get(sk, 0), dem.get(sk, 0)
        s_pct, d_pct = round(100 * s_cnt / n_res, 1), round(100 * d_cnt / n_proj, 1)
        rows.append({
            "Skill": sk, "Available_Researchers": s_cnt, "Required_by_Projects": d_cnt,
            "Availability_Pct": s_pct, "Project_Demand_Pct": d_pct,
            "Scarcity_Gap_Pct": round(d_pct - s_pct, 1), "Supply_to_Demand_Ratio": round(s_cnt / max(1, d_cnt), 2),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["Required_by_Projects", "Available_Researchers"], ascending=[False, True]).reset_index(drop=True) if not out.empty else out


def department_bridges(df, proj_df):
    depts = sorted(df["Department"].unique().tolist())
    if not depts:
        return pd.DataFrame(), pd.DataFrame()
    weights, shared_proj = Counter(), Counter()
    for _, grp in df.groupby("Project_Title"):
        p_depts = sorted(grp["Department"].unique().tolist())
        for i in range(len(p_depts)):
            for j in range(i + 1, len(p_depts)):
                weights[(p_depts[i], p_depts[j])] += 2
                shared_proj[(p_depts[i], p_depts[j])] += 1
    for _, proj in proj_df.iterrows():
        d1, d2 = str(proj["Primary_Department"]), str(proj["Preferred_Department"])
        if d1 in depts and d2 in depts and d1 != d2:
            weights[tuple(sorted((d1, d2)))] += 1

    strength, partners, edges = Counter(), {d: set() for d in depts}, []
    for (d1, d2), w in weights.items():
        edges.append({"Pair": f"{d1} x {d2}", "Weight": int(w), "Shared_Projects": int(shared_proj.get((d1, d2), 0))})
        strength[d1] += int(w); strength[d2] += int(w)
        partners[d1].add(d2); partners[d2].add(d1)
    edges_df = pd.DataFrame(edges).sort_values("Weight", ascending=False).reset_index(drop=True) if edges else pd.DataFrame()

    dept_rows = [{
        "Department": d, "Researchers": len(df[df["Department"] == d]),
        "Unique_Projects": int(df.loc[df["Department"] == d, "Project_Title"].nunique()),
        "Mean_Match_Score": round(float(df.loc[df["Department"] == d, "Match_Score"].mean()), 2),
        "Connection_Strength": int(strength.get(d, 0)), "Partner_Departments": len(partners[d]),
    } for d in depts]
    return pd.DataFrame(dept_rows), edges_df


def overlapping_interests(df, top_n=10):
    depts = sorted(df["Department"].unique().tolist())
    d_int = {d: Counter(t for v in df.loc[df["Department"] == d, "Research_Interests"] for t in set(split_pipes(v))) for d in depts}
    pairs = []
    for i in range(len(depts)):
        for j in range(i + 1, len(depts)):
            d1, d2 = depts[i], depts[j]
            shared = set(d_int[d1]) & set(d_int[d2])
            if shared:
                intensity = sum(min(d_int[d1][t], d_int[d2][t]) for t in shared)
                top_t = sorted(shared, key=lambda t: d_int[d1][t] + d_int[d2][t], reverse=True)[:4]
                pairs.append({"Department_A": d1, "Department_B": d2, "Shared_Topics_Count": len(shared), "Overlap_Intensity": intensity, "Top_Shared_Interests": ", ".join(top_t)})
    out = pd.DataFrame(pairs)
    return out.sort_values(["Overlap_Intensity", "Shared_Topics_Count"], ascending=False).head(top_n).reset_index(drop=True) if not out.empty else out


# --- Collaboration Matching & Simulation ---
def score_pair(src, cand, team_size=None):
    s_int, c_int = set(split_pipes(src.get("Research_Interests"))), set(split_pipes(cand.get("Research_Interests")))
    s_skl, c_skl = set(split_pipes(src.get("Technical_Skills"))), set(split_pipes(cand.get("Technical_Skills")))
    s_req = set(split_pipes(src.get("Required_Skills"))) or s_skl

    s_dom, c_dom = str(src.get("Research_Domain", "")).strip(), str(cand.get("Research_Domain", "")).strip()
    s_pdom, c_pdom = str(src.get("Preferred_Collaboration_Domain", "")).strip(), str(cand.get("Preferred_Collaboration_Domain", "")).strip()
    s_dept, c_dept = str(src.get("Department", "")).strip(), str(cand.get("Department", "")).strip()
    s_pdept = str(src.get("Preferred_Department", "")).strip()

    # 1. Research Interests (35%)
    shared_int = sorted(s_int & c_int)
    jaccard = len(shared_int) / len(s_int | c_int) if (s_int | c_int) else 0.0
    int_score = round(min(1.0, 0.80 * jaccard + 0.20 * (s_dom == c_dom)) * 100, 2)

    # 2. Technical Skills (30%)
    cov_req = sorted(s_req & c_skl)
    req_cov = len(cov_req) / len(s_req) if s_req else 0.0
    missing = s_req - s_skl
    comp_skl = sorted(missing & c_skl) if missing else sorted(c_skl - s_skl)
    comp_ratio = len(comp_skl) / len(missing) if missing else min(1.0, len(comp_skl) / max(1, len(c_skl)))
    shared_ratio = len(s_skl & c_skl) / len(s_skl | c_skl) if (s_skl | c_skl) else 0.0
    if team_size is not None and team_size >= 5:
        avail = min(1.0, float(cand.get("Availability_Hours", 10) or 10) / 16.0)
        sk_raw = 0.45 * req_cov + 0.35 * comp_ratio + 0.10 * shared_ratio + 0.10 * avail
    elif team_size is not None and team_size <= 2:
        sk_raw = 0.60 * req_cov + 0.20 * comp_ratio + 0.20 * shared_ratio
    else:
        sk_raw = 0.50 * req_cov + 0.30 * comp_ratio + 0.20 * shared_ratio
    skill_score = round(min(1.0, sk_raw) * 100, 2)

    # 3. Methodology (15%), 4. Domain (10%), 5. Department (10%)
    m1, m2 = str(src.get("Research_Methodology", "")).strip(), str(cand.get("Research_Methodology", "")).strip()
    meth_raw = 0.50 if (not m1 or not m2) else 1.0 if m1 == m2 else METHOD_PAIRS.get((m1, m2), METHOD_PAIRS.get((m2, m1), 0.50))
    meth_score = round(meth_raw * 100, 2)

    fwd, rev = (s_pdom == c_dom), (c_pdom == s_dom)
    dom_score = round((1.0 if (fwd and rev) else 0.85 if fwd else 0.65 if rev else 0.45 if s_dom == c_dom else 0.15) * 100, 2)

    is_inter = bool(s_dept and c_dept and s_dept != c_dept)
    dept_score = round((1.0 if (is_inter and c_dept == s_pdept) else 0.85 if is_inter else 0.55 if c_dept == s_pdept else 0.35) * 100, 2)

    total = round(WEIGHTS["interest"] * int_score + WEIGHTS["skill"] * skill_score + WEIGHTS["methodology"] * meth_score + WEIGHTS["domain"] * dom_score + WEIGHTS["department"] * dept_score, 2)

    reasons = []
    if shared_int: reasons.append(f"Shares {len(shared_int)} interest(s): {', '.join(shared_int[:3])}")
    elif s_dom == c_dom: reasons.append(f"Aligned in {c_dom}")
    if comp_skl: reasons.append(f"Complementary skills ({', '.join(comp_skl[:3])})")
    elif cov_req: reasons.append(f"Covers required skills ({', '.join(cov_req[:3])})")
    reasons.append(f"Interdisciplinary ({s_dept} x {c_dept})" if is_inter else f"Same-dept ({c_dept})")

    return {
        "total": total, "int": int_score, "skl": skill_score, "meth": meth_score, "dom": dom_score, "dept": dept_score,
        "shared_int": shared_int, "comp_skl": comp_skl, "inter": is_inter, "why": "; ".join(reasons) + ".",
    }


def score_pool(src, pool, team_size=None):
    rows = []
    for _, cand in pool.iterrows():
        r = score_pair(src, cand, team_size=team_size)
        rows.append({
            "Researcher_ID": cand["Researcher_ID"], "Researcher_Name": cand["Researcher_Name"],
            "Role": cand["Role"], "Department": cand["Department"],
            "Research_Domain": cand["Research_Domain"], "Research_Methodology": cand["Research_Methodology"],
            "Compatibility_Score": r["total"], "Interest_Score": r["int"], "Skill_Score": r["skl"],
            "Methodology_Score": r["meth"], "Domain_Score": r["dom"], "Department_Score": r["dept"],
            "Shared_Interests": ", ".join(r["shared_int"]) or "Domain alignment",
            "Complementary_Skills": ", ".join(r["comp_skl"][:4]) or "Shared technical baseline",
            "Is_Interdisciplinary": r["inter"], "Match_Explanation": r["why"],
        })
    return pd.DataFrame(rows)


def recommend_collaborators(df, target_id, pool_df, top_n=5, sort_by="Compatibility_Score", ascending=False):
    src = df[df["Researcher_ID"] == target_id]
    pool = pool_df[pool_df["Researcher_ID"] != target_id].drop_duplicates(subset=["Researcher_ID"])
    if src.empty or pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    scored = score_pool(src.iloc[0], pool).sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    scored["Rank"] = np.arange(1, len(scored) + 1)
    return scored.head(top_n).copy(), scored


def simulate_matches(df, base_id, sim_dom, sim_dept, sim_skills, sim_team, sim_meth, pool_df, top_n=5):
    src = df[df["Researcher_ID"] == base_id]
    pool = pool_df[pool_df["Researcher_ID"] != base_id].drop_duplicates(subset=["Researcher_ID"])
    if src.empty or pool.empty:
        return pd.DataFrame(), pd.DataFrame()
    base = src.iloc[0].to_dict()
    sim_prof = {**base, "Preferred_Collaboration_Domain": sim_dom, "Preferred_Department": sim_dept,
                "Required_Skills": "|".join(sim_skills) if sim_skills else base.get("Required_Skills", ""),
                "Research_Methodology": sim_meth, "Team_Size": sim_team}
    orig = score_pool(base, pool)[["Researcher_ID", "Compatibility_Score"]].rename(columns={"Compatibility_Score": "Original_Score"}).sort_values("Original_Score", ascending=False).reset_index(drop=True)
    orig["Original_Rank"] = np.arange(1, len(orig) + 1)
    sim = score_pool(sim_prof, pool, team_size=sim_team).rename(columns={"Compatibility_Score": "Simulated_Score"}).sort_values("Simulated_Score", ascending=False).reset_index(drop=True)
    sim["Simulated_Rank"] = np.arange(1, len(sim) + 1)
    merged = sim.merge(orig, on="Researcher_ID", how="left")
    merged["Score_Delta"] = (merged["Simulated_Score"] - merged["Original_Score"]).round(2)
    merged["Rank_Delta"] = merged["Original_Rank"] - merged["Simulated_Rank"]
    return merged.head(top_n).copy(), merged


# --- UI Helpers ---
def show_chart(title, fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=15), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend)
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.plotly_chart(fig, width="stretch")


def hbar(df, x, y):
    fig = px.bar(df, x=x, y=y, orientation="h", color=x, color_continuous_scale="Tealgrn", text=x)
    fig.update_traces(textposition="outside")
    fig.update_coloraxes(showscale=False)
    return fig


def match_card(title, meta, score_txt, tags, why, shared, comp):
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{title}**")
        c1.caption(meta)
        c2.markdown(f"**{score_txt}**")
        st.markdown(" ".join(f":blue-badge[{t}]" for t in tags))
        st.caption(f"Why it fits: {why} | Shared Interests: {shared} | Complementary Skills: {comp}")


def sidebar_filters(df):
    with st.sidebar:
        st.header("RESEARCHNEXUS Filters")
        if st.button("Reset Filters", width="stretch"):
            for k in ["f_dept", "f_role", "f_dom", "f_exp", "f_stat", "f_res", "f_proj"]:
                st.session_state.pop(k, None)
            st.rerun()
        depts = st.multiselect("Department", sorted(df["Department"].unique()), key="f_dept")
        roles = st.multiselect("Role", sorted(df["Role"].unique()), key="f_role")
        doms = st.multiselect("Research Domain", sorted(df["Research_Domain"].unique()), key="f_dom")
        exps = st.multiselect("Experience Level", ["Beginner", "Intermediate", "Advanced", "Expert"], key="f_exp")
        stats = st.multiselect("Collaboration Status", ["Open", "Active", "Ongoing", "Matched", "Completed"], key="f_stat")
        res_q = st.text_input("Search Researcher", key="f_res", placeholder="Name or ID")
        proj_q = st.text_input("Search Project", key="f_proj", placeholder="Title or keyword")

    fdf = df.copy()
    if depts: fdf = fdf[fdf["Department"].isin(depts)]
    if roles: fdf = fdf[fdf["Role"].isin(roles)]
    if doms: fdf = fdf[fdf["Research_Domain"].isin(doms)]
    if exps: fdf = fdf[fdf["Experience_Level"].isin(exps)]
    if stats: fdf = fdf[fdf["Collaboration_Status"].isin(stats)]
    if res_q.strip():
        q = res_q.strip().lower()
        fdf = fdf[fdf["Researcher_Name"].str.lower().str.contains(q) | fdf["Researcher_ID"].str.lower().str.contains(q)]
    if proj_q.strip():
        q = proj_q.strip().lower()
        fdf = fdf[fdf["Project_Title"].str.lower().str.contains(q) | fdf["Project_Description"].str.lower().str.contains(q)]

    with st.sidebar:
        st.divider()
        st.caption(f"Showing {len(fdf):,} / {len(df):,} researchers | {fdf['Project_Title'].nunique():,} projects")
        exp_df = fdf.drop(columns=["Role_Group", "Project_Date", "Project_Month"], errors="ignore")
        st.download_button("Download Filtered CSV", exp_df.to_csv(index=False).encode("utf-8"), "researchnexus_filtered.csv", "text/csv", width="stretch")
    return fdf


# --- Dashboard Sections ---
def render_overview(df, proj_df):
    st.subheader("1. Research Ecosystem Overview", divider=True)
    k = st.columns(6)
    k[0].metric("Researchers", f"{len(df):,}")
    k[1].metric("Unique Projects", f"{len(proj_df):,}")
    k[2].metric("Departments", f"{df['Department'].nunique()}")
    k[3].metric("Domains", f"{df['Research_Domain'].nunique()}")
    k[4].metric("Open Projects", f"{int((proj_df['Collaboration_Status'] == 'Open').sum()):,}")
    k[5].metric("Avg Match Score", f"{df['Match_Score'].mean():.1f}")

    top_dept, top_dom = df["Department"].value_counts(), df["Research_Domain"].value_counts()
    st.info(f"Summary: {top_dept.index[0]} leads with {int(top_dept.iloc[0])} researchers ({100*top_dept.iloc[0]/len(df):.1f}%), {top_dom.index[0]} is the top domain ({int(top_dom.iloc[0])} researchers), and {100*proj_df['Is_Interdisciplinary'].mean():.1f}% of projects are interdisciplinary.")

    c1, c2 = st.columns(2)
    with c1:
        dept_c = df["Department"].value_counts().reset_index()
        dept_c.columns = ["Department", "Researchers"]
        show_chart("Department-wise Researchers", hbar(dept_c.sort_values("Researchers"), "Researchers", "Department"), legend=False)
    with c2:
        dom_c = df["Research_Domain"].value_counts().reset_index()
        dom_c.columns = ["Domain", "Researchers"]
        show_chart("Research Domain Share", px.pie(dom_c, names="Domain", values="Researchers", hole=0.55, color_discrete_sequence=PALETTE))

    c3, c4 = st.columns(2)
    with c3:
        sf = df.groupby(["Department", "Role_Group"]).size().reset_index(name="Count")
        fig = px.bar(sf, x="Department", y="Count", color="Role_Group", barmode="stack", color_discrete_map={"Student": "#2E4A68", "Faculty": "#D5B77B"})
        fig.update_xaxes(tickangle=-25)
        show_chart("Student vs Faculty Participation", fig)
    with c4:
        vp = proj_df[proj_df["Project_Date"].notna()]
        if vp.empty:
            st.info("No dated project records available.")
        else:
            trend = vp.groupby("Project_Month").agg(Projects=("Project_Title", "nunique"), Researchers=("Assigned_Researchers", "sum")).reset_index().sort_values("Project_Month")
            fig = go.Figure([
                go.Scatter(x=trend["Project_Month"], y=trend["Projects"], name="Projects", mode="lines+markers"),
                go.Scatter(x=trend["Project_Month"], y=trend["Researchers"], name="Researchers", mode="lines+markers", yaxis="y2"),
            ])
            fig.update_layout(yaxis=dict(title="Projects"), yaxis2=dict(title="Researchers", overlaying="y", side="right", showgrid=False))
            fig.update_xaxes(nticks=10, tickangle=-25)
            show_chart("Monthly Research Activity Trends", fig)


def render_landscape(df):
    st.subheader("2. Research Landscape & Topic Overlap", divider=True)
    f1, f2 = st.columns(2)
    s_depts = f1.multiselect("Filter by Department", sorted(df["Department"].unique()))
    s_doms = f2.multiselect("Filter by Domain", sorted(df["Research_Domain"].unique()))
    sub = df.copy()
    if s_depts: sub = sub[sub["Department"].isin(s_depts)]
    if s_doms: sub = sub[sub["Research_Domain"].isin(s_doms)]
    if sub.empty:
        st.warning("No records match the section filters.")
        return

    sub_proj = get_unique_projects(sub)
    topics = pd.Series([t for v in sub["Research_Interests"] for t in split_pipes(v)]).value_counts()
    dom_c = sub["Research_Domain"].value_counts()

    m = st.columns(3)
    m[0].metric("Most Common Topic", topics.index[0] if not topics.empty else "N/A", f"{int(topics.iloc[0])} researchers" if not topics.empty else "0")
    m[1].metric("Underrepresented Topic", topics.index[-1] if not topics.empty else "N/A", f"{int(topics.iloc[-1])} researchers" if not topics.empty else "0")
    m[2].metric("Smallest Domain", dom_c.index[-1], f"{int(dom_c.iloc[-1])} researchers")

    c1, c2 = st.columns(2)
    with c1:
        dom_df = dom_c.reset_index(); dom_df.columns = ["Domain", "Researchers"]
        show_chart("Researchers by Domain", hbar(dom_df.sort_values("Researchers"), "Researchers", "Domain"), height=380, legend=False)
    with c2:
        flow = sub.groupby(["Department", "Research_Domain"]).size().reset_index(name="Researchers")
        fig = px.bar(flow, x="Department", y="Researchers", color="Research_Domain", barmode="stack", color_discrete_sequence=PALETTE)
        fig.update_xaxes(tickangle=-25)
        show_chart("Department vs Domain Distribution", fig, height=380)

    c3, c4 = st.columns(2)
    with c3:
        if topics.empty:
            with st.container(border=True):
                st.markdown("**Top 15 Research Topics**")
                st.info("No declared research interests in this selection.")
        else:
            top15 = topics.head(15).reset_index(); top15.columns = ["Topic", "Researchers"]
            show_chart("Top 15 Research Topics", hbar(top15.sort_values("Researchers"), "Researchers", "Topic"), height=380, legend=False)
    with c4:
        vp = sub_proj[sub_proj["Project_Date"].notna()]
        if vp.empty:
            with st.container(border=True):
                st.markdown("**Cumulative Projects (Top 5 Domains)**")
                st.info("No dated projects available for cumulative trends.")
        else:
            top5_d = vp["Primary_Domain"].value_counts().head(5).index.tolist()
            pt = vp[vp["Primary_Domain"].isin(top5_d)].groupby(["Project_Month", "Primary_Domain"])["Project_Title"].nunique().reset_index(name="Projects").sort_values("Project_Month")
            pt["Cumulative"] = pt.groupby("Primary_Domain")["Projects"].cumsum()
            fig = px.line(pt, x="Project_Month", y="Cumulative", color="Primary_Domain", markers=True, color_discrete_sequence=PALETTE)
            fig.update_xaxes(nticks=10, tickangle=-25)
            show_chart("Cumulative Projects (Top 5 Domains)", fig, height=380)

    with st.container(border=True):
        st.markdown("**Overlapping Research Interests Across Departments**")
        ov = overlapping_interests(sub, top_n=10)
        if not ov.empty:
            st.dataframe(ov, width="stretch", hide_index=True)
        else:
            st.info("Select at least two departments to view shared interests.")


def render_departments(df, proj_df):
    st.subheader("3. Department Intelligence & Bridges", divider=True)
    dept_df, edges_df = department_bridges(df, proj_df)
    if dept_df.empty:
        st.warning("No department records available.")
        return

    strongest = dept_df.sort_values("Researchers", ascending=False).iloc[0]
    weakest = dept_df.sort_values("Connection_Strength").iloc[0]
    m = st.columns(3)
    m[0].metric("Largest Department", strongest["Department"], f"{int(strongest['Researchers'])} researchers")
    m[1].metric("Fewest Connections", weakest["Department"], f"{int(weakest['Connection_Strength'])} weighted links")
    m[2].metric("Top Bridge", edges_df.iloc[0]["Pair"] if not edges_df.empty else "Single Dept", f"Weight {int(edges_df.iloc[0]['Weight'])}" if not edges_df.empty else "N/A")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(pd.crosstab(df["Department"], df["Research_Domain"]), text_auto=True, aspect="auto", color_continuous_scale="Teal")
        fig.update_xaxes(tickangle=-25)
        show_chart("Heatmap: Department vs Research Domain", fig, height=400, legend=False)
    with c2:
        role_df = df.groupby(["Department", "Role"]).size().reset_index(name="Count")
        fig = px.bar(role_df, x="Department", y="Count", color="Role", barmode="stack", color_discrete_sequence=PALETTE)
        fig.update_xaxes(tickangle=-25)
        show_chart("Role Distribution by Department", fig, height=400)

    c3, c4 = st.columns(2)
    with c3:
        if edges_df.empty:
            with st.container(border=True):
                st.markdown("**Top Interdepartmental Bridges**")
                st.info("Select at least two departments to view collaboration bridges.")
        else:
            show_chart("Top Interdepartmental Bridges", hbar(edges_df.head(12).sort_values("Weight"), "Weight", "Pair"), height=400, legend=False)
    with c4:
        dp = df.groupby("Department")["Project_Title"].nunique().reset_index(name="Unique_Projects").sort_values("Unique_Projects")
        show_chart("Unique Projects per Department", hbar(dp, "Unique_Projects", "Department"), height=400, legend=False)

    t1, t2 = st.columns(2)
    with t1, st.container(border=True):
        st.markdown("**Department Connectivity Scorecard**")
        st.dataframe(dept_df.sort_values("Connection_Strength", ascending=False), width="stretch", hide_index=True)
    with t2, st.container(border=True):
        st.markdown("**Recommended Interdisciplinary Partnerships**")
        ov = overlapping_interests(df, top_n=8)
        if not ov.empty:
            st.dataframe(ov, width="stretch", hide_index=True)
        else:
            st.info("Requires >= 2 departments.")


def render_skills(df, proj_df):
    st.subheader("4. Skills Intelligence & Researcher Radar", divider=True)
    sd = skill_supply_demand(df, proj_df)
    top_req, scarce, gap = sd.iloc[0], sd.sort_values("Available_Researchers").iloc[0], sd.sort_values("Scarcity_Gap_Pct", ascending=False).iloc[0]

    m = st.columns(3)
    m[0].metric("Most Demanded Skill", top_req["Skill"], f"{int(top_req['Required_by_Projects'])} projects")
    m[1].metric("Least Available Skill", scarce["Skill"], f"{int(scarce['Available_Researchers'])} researchers")
    m[2].metric("Largest Scarcity Gap", gap["Skill"], f"{gap['Scarcity_Gap_Pct']:+.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        show_chart("Top 15 Demanded Skills", hbar(sd.head(15).sort_values("Required_by_Projects"), "Required_by_Projects", "Skill"), height=400, legend=False)
    with c2:
        melt = sd.head(12).melt(id_vars="Skill", value_vars=["Availability_Pct", "Project_Demand_Pct"], var_name="Metric", value_name="Pct")
        melt["Metric"] = melt["Metric"].map({"Availability_Pct": "Supply (%)", "Project_Demand_Pct": "Demand (%)"})
        fig = px.bar(melt, x="Skill", y="Pct", color="Metric", barmode="group", color_discrete_map={"Supply (%)": "#78998B", "Demand (%)": "#2E4A68"})
        fig.update_xaxes(tickangle=-30)
        show_chart("Skill Supply vs Project Demand (%)", fig, height=400)

    c3, c4 = st.columns(2)
    with c3:
        sk_rows = df.assign(Skill=df["Technical_Skills"].apply(split_pipes)).explode("Skill")
        heat = sk_rows[sk_rows["Skill"].isin(sd.head(15)["Skill"].tolist())].groupby(["Department", "Skill"]).size().unstack(fill_value=0)
        fig = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Teal")
        fig.update_xaxes(tickangle=-30)
        show_chart("Heatmap: Department vs Top 15 Skills", fig, height=420, legend=False)
    with c4, st.container(border=True):
        st.markdown("**Researcher Skill Radar Comparison**")
        opts = (df["Researcher_ID"] + " - " + df["Researcher_Name"] + " (" + df["Department"] + ")").tolist()
        sel = st.multiselect("Compare 2-4 Researchers", opts, default=opts[:min(3, len(opts))], max_selections=4)
        if sel:
            sel_rows = df[df["Researcher_ID"].isin([x.split(" - ")[0] for x in sel])]
            axes = list(dict.fromkeys([s for v in sel_rows["Technical_Skills"] for s in split_pipes(v)] + sd["Skill"].tolist()))[:8]
            exp_w = {"Beginner": 45, "Intermediate": 68, "Advanced": 86, "Expert": 100}
            fig = go.Figure()
            for i, (_, r) in enumerate(sel_rows.iterrows()):
                r_sk, r_rq = set(split_pipes(r["Technical_Skills"])), set(split_pipes(r["Required_Skills"]))
                prof = exp_w.get(r["Experience_Level"], 70)
                vals = [prof if ax in r_sk else (25 if ax in r_rq else 5) for ax in axes]
                fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=axes + [axes[0]], fill="toself", name=r["Researcher_Name"], line=dict(color=PALETTE[i % len(PALETTE)]), opacity=0.65))
            fig.update_layout(height=340, margin=dict(l=15, r=15, t=25, b=15), paper_bgcolor="rgba(0,0,0,0)", polar=dict(radialaxis=dict(range=[0, 105]), bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, width="stretch")

    st.markdown("**Skill Supply-Demand Gap Table**")
    st.dataframe(sd, width="stretch", hide_index=True)


def render_projects(df, proj_df):
    st.subheader("5. Project Analytics & Skill Readiness", divider=True)
    st_counts, gaps_n = proj_df["Collaboration_Status"].value_counts(), int(proj_df["Has_Skill_Gap"].sum())
    m = st.columns(5)
    m[0].metric("Unique Projects", f"{len(proj_df):,}")
    m[1].metric("Open", f"{int(st_counts.get('Open', 0)):,}")
    m[2].metric("Active & Ongoing", f"{int(st_counts.get('Active', 0) + st_counts.get('Ongoing', 0)):,}")
    m[3].metric("Avg Team Size", f"{proj_df['Team_Size'].mean():.2f}")
    m[4].metric("Skill-Gap Projects", f"{gaps_n:,}", f"{100*gaps_n/len(proj_df):.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st_df = proj_df.groupby(["Primary_Department", "Collaboration_Status"]).size().reset_index(name="Projects")
        fig = px.bar(st_df, x="Primary_Department", y="Projects", color="Collaboration_Status", barmode="stack", color_discrete_map=STATUS_COLORS)
        fig.update_xaxes(tickangle=-25)
        show_chart("Collaboration Status by Department", fig)
    with c2:
        bub = proj_df.copy()
        bub["Size_X"] = bub["Team_Size"] + np.random.default_rng(42).uniform(-0.2, 0.2, len(bub))
        fig = px.scatter(bub, x="Size_X", y="Mean_Match_Score", size="Required_Skills_Count", color="Collaboration_Status", color_discrete_map=STATUS_COLORS, hover_name="Project_Title", hover_data={"Size_X": False, "Team_Size": True, "Missing_Skills": True}, opacity=0.78)
        fig.update_xaxes(title="Team Size", tickvals=[2, 3, 4, 5, 6])
        show_chart("Team Size vs Mean Match Score", fig)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(df, x="Department", y="Match_Score", color="Department", color_discrete_sequence=PALETTE)
        fig.update_xaxes(tickangle=-25)
        show_chart("Match Score Distribution by Department", fig, legend=False)
    with c4:
        vp = proj_df[proj_df["Project_Date"].notna()]
        if not vp.empty:
            pt = vp.groupby(["Project_Month", "Collaboration_Status"])["Project_Title"].nunique().reset_index(name="Projects").sort_values("Project_Month")
            fig = px.line(pt, x="Project_Month", y="Projects", color="Collaboration_Status", markers=True, color_discrete_map=STATUS_COLORS)
            fig.update_xaxes(nticks=10, tickangle=-25)
            show_chart("Project Trends by Status", fig)

    st.markdown("**Project Explorer**")
    s1, s2 = st.columns([3, 1])
    q = s1.text_input("Search project title, department, or missing skill")
    disp = proj_df[proj_df["Has_Skill_Gap"]] if s2.checkbox("Only skill-gap projects") else proj_df
    if q.strip():
        ql = q.strip().lower()
        disp = disp[disp["Project_Title"].str.lower().str.contains(ql) | disp["Departments_Involved"].str.lower().str.contains(ql) | disp["Missing_Skills"].str.lower().str.contains(ql)]
    st.dataframe(disp[["Project_Title", "Primary_Department", "Primary_Domain", "Collaboration_Status", "Team_Size", "Assigned_Researchers", "Open_Slots", "Mean_Match_Score", "Required_Skills", "Missing_Skills"]], width="stretch", hide_index=True)


def render_collaboration(full_df, filtered_df):
    st.subheader("6. Collaboration Matching Engine", divider=True)
    pool = filtered_df if len(filtered_df) >= 6 else full_df
    opts = (full_df["Researcher_ID"] + " - " + full_df["Researcher_Name"] + " (" + full_df["Department"] + ")").sort_values().tolist()

    c1, c2, c3 = st.columns([2, 1.2, 0.8])
    target_id = c1.selectbox("Select Target Researcher", opts).split(" - ")[0]
    sort_col = c2.selectbox("Sort By", ["Compatibility_Score", "Interest_Score", "Skill_Score", "Methodology_Score", "Domain_Score", "Department_Score"])
    sort_asc = c3.selectbox("Order", ["Descending", "Ascending"]) == "Ascending"

    src = full_df[full_df["Researcher_ID"] == target_id].iloc[0]
    with st.container(border=True):
        st.markdown(f"**Selected:** {src['Researcher_Name']} (`{src['Researcher_ID']}`) | {src['Role']} | {src['Department']} | Domain: **{src['Research_Domain']}** (Prefers: {src['Preferred_Collaboration_Domain']} / {src['Preferred_Department']})")
        st.caption(f"Interests: {', '.join(split_pipes(src['Research_Interests']))} | Skills: {', '.join(split_pipes(src['Technical_Skills']))} | Needs: {', '.join(split_pipes(src['Required_Skills']))}")

    top5, all_scored = recommend_collaborators(full_df, target_id, pool, top_n=5, sort_by=sort_col, ascending=sort_asc)
    if top5.empty:
        st.warning("No candidate collaborators available.")
        return

    for _, cand in top5.iterrows():
        dept_tag = f"Interdisciplinary: {src['Department']} x {cand['Department']}" if cand["Is_Interdisciplinary"] else f"Same Dept: {cand['Department']}"
        match_card(
            f"Rank #{int(cand['Rank'])} - {cand['Researcher_Name']} ({cand['Researcher_ID']})",
            f"{cand['Role']} | {cand['Department']} | {cand['Research_Domain']} | {cand['Research_Methodology']}",
            f"{cand['Compatibility_Score']:.1f} / 100",
            [dept_tag, f"Interests {cand['Interest_Score']:.0f}%", f"Skills {cand['Skill_Score']:.0f}%", f"Method {cand['Methodology_Score']:.0f}%"],
            cand["Match_Explanation"], cand["Shared_Interests"], cand["Complementary_Skills"],
        )

    st.download_button("Download Matching Results CSV", all_scored.to_csv(index=False).encode("utf-8"), f"matches_{target_id}.csv", "text/csv")

    v1, v2 = st.columns(2)
    with v1, st.container(border=True):
        st.markdown("**Candidate Score Distribution**")
        fig_m, ax = plt.subplots(figsize=(6, 3.5))
        scores = all_scored["Compatibility_Score"].to_numpy()
        ax.hist(scores, bins=24, color="#78998B", edgecolor="white")
        ax.axvline(scores.mean(), color="#17283B", ls="--", label=f"Mean ({scores.mean():.1f})")
        ax.axvline(top5["Compatibility_Score"].min(), color="#B89146", label=f"Top-5 ({top5['Compatibility_Score'].min():.1f})")
        ax.set_xlabel("Compatibility Score")
        ax.set_ylabel("Candidates")
        ax.legend(frameon=False)
        plt.tight_layout()
        st.pyplot(fig_m, width="stretch")
        plt.close(fig_m)
    with v2:
        bdf = top5.copy()
        bdf["Label"] = bdf["Researcher_Name"] + " (" + bdf["Department"] + ")"
        fmap = {"Interest_Score": ("Interests (35%)", "interest"), "Skill_Score": ("Skills (30%)", "skill"), "Methodology_Score": ("Method (15%)", "methodology"), "Domain_Score": ("Domain (10%)", "domain"), "Department_Score": ("Dept (10%)", "department")}
        for sc, (lbl, wk) in fmap.items():
            bdf[lbl] = (bdf[sc] * WEIGHTS[wk]).round(2)
        melt = bdf.melt(id_vars=["Label", "Compatibility_Score"], value_vars=[v[0] for v in fmap.values()], var_name="Factor", value_name="Points")
        fig = px.bar(melt, x="Points", y="Label", color="Factor", orientation="h", barmode="stack", category_orders={"Label": bdf.sort_values("Compatibility_Score")["Label"].tolist()}, color_discrete_sequence=PALETTE)
        fig.update_xaxes(range=[0, 102], title="Weighted Score")
        show_chart("Top 5 Score Breakdown", fig, height=370)


def render_simulator(full_df, filtered_df):
    st.subheader("7. Interactive Collaboration Simulator", divider=True)
    pool = filtered_df if len(filtered_df) >= 6 else full_df
    opts = (full_df["Researcher_ID"] + " - " + full_df["Researcher_Name"] + " (" + full_df["Department"] + ")").sort_values().tolist()
    all_doms, all_depts = sorted(full_df["Research_Domain"].unique()), sorted(full_df["Department"].unique())
    all_skills = sorted({s for v in full_df["Technical_Skills"] for s in split_pipes(v)})

    with st.container(border=True):
        r1c1, r1c2, r1c3 = st.columns([1.6, 1, 1])
        base_id = r1c1.selectbox("Baseline Researcher", opts, key="sim_base").split(" - ")[0]
        base = full_df[full_df["Researcher_ID"] == base_id].iloc[0]
        sim_dom = r1c2.selectbox("Target Domain", all_doms, index=all_doms.index(base["Preferred_Collaboration_Domain"]) if base["Preferred_Collaboration_Domain"] in all_doms else 0)
        sim_dept = r1c3.selectbox("Preferred Department", all_depts, index=all_depts.index(base["Preferred_Department"]) if base["Preferred_Department"] in all_depts else 0)

        r2c1, r2c2, r2c3 = st.columns([2, 1, 1])
        sim_skills = r2c1.multiselect("Required Project Skills", all_skills, default=[s for s in split_pipes(base["Required_Skills"]) if s in all_skills])
        sim_meth = r2c2.selectbox("Methodology Preference", METHODOLOGIES, index=METHODOLOGIES.index(base["Research_Methodology"]) if base["Research_Methodology"] in METHODOLOGIES else 0)
        sim_team = r2c3.slider("Target Team Size", 2, 6, int(base["Team_Size"]))

    sim_top5, sim_all = simulate_matches(full_df, base_id, sim_dom, sim_dept, sim_skills, sim_team, sim_meth, pool, top_n=5)
    if sim_top5.empty:
        st.warning("No candidates available to simulate.")
        return

    orig_mean = float(sim_all.sort_values("Original_Score", ascending=False).head(5)["Original_Score"].mean())
    sim_mean = float(sim_top5["Simulated_Score"].mean())
    m = st.columns(3)
    m[0].metric("Simulated Top-5 Mean", f"{sim_mean:.1f} / 100", f"Baseline: {orig_mean:.1f}")
    m[1].metric("Net Shift", f"{sim_mean - orig_mean:+.2f} pts")
    m[2].metric("Peak Uplift", f"{max(0.0, float(sim_top5['Score_Delta'].max())):+.2f} pts")

    c1, c2 = st.columns(2)
    with c1:
        cmp_df = sim_top5.copy()
        cmp_df["Label"] = cmp_df["Researcher_Name"] + " (" + cmp_df["Department"] + ")"
        melt = cmp_df.melt(id_vars="Label", value_vars=["Original_Score", "Simulated_Score"], var_name="Scenario", value_name="Score")
        melt["Scenario"] = melt["Scenario"].map({"Original_Score": "Original", "Simulated_Score": "Simulated"})
        fig = px.bar(melt, x="Score", y="Label", color="Scenario", barmode="group", orientation="h", category_orders={"Label": cmp_df.sort_values("Simulated_Score")["Label"].tolist()}, color_discrete_map={"Original": "#94A3B8", "Simulated": "#2E4A68"}, text="Score")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig.update_xaxes(range=[0, 108])
        show_chart("Original vs Simulated Compatibility", fig, height=370)
    with c2, st.container(border=True):
        st.markdown("**Rank & Score Delta Comparison**")
        st.dataframe(sim_top5[["Simulated_Rank", "Original_Rank", "Rank_Delta", "Researcher_Name", "Department", "Original_Score", "Simulated_Score", "Score_Delta"]], width="stretch", hide_index=True)
        st.download_button("Download Simulated Results CSV", sim_all.to_csv(index=False).encode("utf-8"), f"simulation_{base_id}.csv", "text/csv", width="stretch")

    for _, row in sim_top5.iterrows():
        match_card(
            f"Sim #{int(row['Simulated_Rank'])} - {row['Researcher_Name']} ({row['Department']})",
            f"{row['Role']} | {row['Research_Domain']} | {row['Research_Methodology']}",
            f"{row['Simulated_Score']:.1f} ({row['Score_Delta']:+.1f})",
            [f"Orig Rank #{int(row['Original_Rank'])}", f"Rank Delta {int(row['Rank_Delta']):+d}", f"Interests {row['Interest_Score']:.0f}%", f"Skills {row['Skill_Score']:.0f}%"],
            row["Match_Explanation"], row["Shared_Interests"], row["Complementary_Skills"],
        )


# --- Main App ---
def main():
    try:
        full_df = load_data(str(DATA_PATH))
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        st.stop()

    filtered_df = sidebar_filters(full_df)
    proj_df = get_unique_projects(filtered_df)

    st.title("RESEARCHNEXUS")
    st.caption("CHRIST (Deemed to be University), Bengaluru | Interdisciplinary Research Intelligence & Collaboration Analytics")
    st.markdown(
        f":blue-badge[{len(filtered_df):,} Researchers] :green-badge[{len(proj_df):,} Unique Projects] "
        f":violet-badge[{filtered_df['Department'].nunique()} Departments] :orange-badge[{filtered_df['Research_Domain'].nunique()} Domains]"
    )

    section = st.radio("Navigation", SECTIONS, horizontal=True, label_visibility="collapsed")
    if filtered_df.empty:
        st.warning("No records match your active filters. Click Reset Filters in the sidebar.")
        st.stop()

    pages = {
        "Overview": lambda: render_overview(filtered_df, proj_df),
        "Landscape": lambda: render_landscape(filtered_df),
        "Departments": lambda: render_departments(filtered_df, proj_df),
        "Skills": lambda: render_skills(filtered_df, proj_df),
        "Projects": lambda: render_projects(filtered_df, proj_df),
        "Matching": lambda: render_collaboration(full_df, filtered_df),
        "Simulator": lambda: render_simulator(full_df, filtered_df),
    }
    pages[section]()


if __name__ == "__main__":
    main()
