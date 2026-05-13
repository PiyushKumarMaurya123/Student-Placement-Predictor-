import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlacePredict — AI Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0f1a;
    color: #e8e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13152a !important;
    border-right: 1px solid #1e2240;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1a1d35 0%, #0d0f1a 60%);
    border: 1px solid #2a2d50;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #8b8fad;
    margin: 0;
}
.hero-accent {
    color: #818cf8;
}

/* Metric cards */
.metric-card {
    background: #13152a;
    border: 1px solid #1e2240;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
}
.metric-label {
    font-size: 0.78rem;
    color: #6b6f8d;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
}

/* Result box */
.result-placed {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-not-placed {
    background: linear-gradient(135deg, #4c1d24, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.5rem 0;
}
.result-prob {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
}

/* Section headers */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #c7c9e8;
    border-left: 3px solid #818cf8;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* Sidebar label override */
.stSlider label, .stSelectbox label, .stRadio label {
    color: #a0a3c0 !important;
    font-size: 0.85rem !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #818cf8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div.stButton > button:hover {
    opacity: 0.88;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #13152a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #6b6f8d;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: #1e2240 !important;
    color: #818cf8 !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODEL (cached so it runs only once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("student_placement_salary_elite_v2.csv")

    le = LabelEncoder()
    df['branch'] = le.fit_transform(df['branch'])
    branch_map = dict(zip(le.classes_, le.transform(le.classes_)))

    DROP = ['student_id', 'company_type', 'job_role', 'salary_lpa', 'placed']
    features = [c for c in df.columns if c not in DROP]

    X = df[features]
    y = df['placed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred       = model.predict(X_test_s)
    y_pred_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "roc_auc"  : roc_auc_score(y_test, y_pred_proba),
        "cm"       : confusion_matrix(y_test, y_pred),
        "report"   : classification_report(y_test, y_pred,
                         target_names=["Not Placed", "Placed"], output_dict=True),
        "coef"     : dict(zip(features, model.coef_[0])),
        "n_train"  : len(X_train),
        "n_test"   : len(X_test),
    }

    return model, scaler, branch_map, features, metrics, df

model, scaler, branch_map, features, metrics, df = train_model()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <span style='font-family:Syne; font-size:1.3rem; font-weight:800; color:#818cf8;'>🎓 PlacePredict</span><br>
        <span style='font-size:0.78rem; color:#6b6f8d;'>Enter student details below</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">📚 Academic Profile</div>', unsafe_allow_html=True)
    cgpa         = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
    branch       = st.selectbox("Branch", list(branch_map.keys()))
    college_tier = st.selectbox("College Tier", [1, 2, 3],
                                format_func=lambda x: f"Tier {x}")
    backlogs     = st.selectbox("Number of Backlogs", [0, 1, 2, 3])

    st.markdown('<div class="section-head">💻 Technical Skills</div>', unsafe_allow_html=True)
    python_skill  = st.radio("Python", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    dsa_skill     = st.radio("DSA", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    ml_skill      = st.radio("Machine Learning", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    web_dev_skill = st.radio("Web Development", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    skill_score   = st.slider("Overall Skill Score", 0, 4, 2)

    st.markdown('<div class="section-head">📊 Test Scores</div>', unsafe_allow_html=True)
    coding_score        = st.slider("Coding Score", 0.0, 100.0, 70.0, 0.5)
    communication_score = st.slider("Communication Score", 0.0, 100.0, 65.0, 0.5)
    aptitude_score      = st.slider("Aptitude Score", 0.0, 100.0, 70.0, 0.5)
    resume_score        = st.slider("Resume Score", 0.0, 100.0, 75.0, 0.5)

    st.markdown('<div class="section-head">🏆 Experience</div>', unsafe_allow_html=True)
    internships = st.selectbox("Internships", [0, 1, 2, 3])
    projects    = st.selectbox("Projects", [1, 2, 3, 4, 5, 6])

    predict_btn = st.button("🔮 Predict Placement")

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🎓 Place<span class="hero-accent">Predict</span></p>
    <p class="hero-sub">AI-powered student placement predictor · Logistic Regression · 9,000 students trained</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Data Insights"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════
with tab1:

    # Model stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value">{metrics['roc_auc']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Training Samples</div>
            <div class="metric-value">{metrics['n_train']:,}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Features Used</div>
            <div class="metric-value">{len(features)}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction result
    if predict_btn:
        input_data = pd.DataFrame([{
            'cgpa'               : cgpa,
            'branch'             : branch_map[branch],
            'college_tier'       : college_tier,
            'python_skill'       : python_skill,
            'dsa_skill'          : dsa_skill,
            'ml_skill'           : ml_skill,
            'web_dev_skill'      : web_dev_skill,
            'coding_score'       : coding_score,
            'communication_score': communication_score,
            'aptitude_score'     : aptitude_score,
            'internships'        : internships,
            'projects'           : projects,
            'backlogs'           : backlogs,
            'resume_score'       : resume_score,
            'skill_score'        : skill_score,
        }])

        scaled     = scaler.transform(input_data)
        pred_label = model.predict(scaled)[0]
        pred_proba = model.predict_proba(scaled)[0][1]

        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if pred_label == 1:
                st.markdown(f"""
                <div class="result-placed">
                    <div style='font-size:2.5rem;'>✅</div>
                    <div class="result-title" style='color:#34d399;'>LIKELY PLACED</div>
                    <div class="result-prob" style='color:#6ee7b7;'>{pred_proba*100:.1f}%</div>
                    <div style='color:#a7f3d0; font-size:0.85rem; margin-top:0.5rem;'>Placement Probability</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-not-placed">
                    <div style='font-size:2.5rem;'>⚠️</div>
                    <div class="result-title" style='color:#fca5a5;'>AT RISK</div>
                    <div class="result-prob" style='color:#fca5a5;'>{pred_proba*100:.1f}%</div>
                    <div style='color:#fecaca; font-size:0.85rem; margin-top:0.5rem;'>Placement Probability</div>
                </div>
                """, unsafe_allow_html=True)

        with col_gauge:
            # Probability bar chart
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor('#13152a')
            ax.set_facecolor('#13152a')

            bars = ax.barh(
                ['Not Placed', 'Placed'],
                [1 - pred_proba, pred_proba],
                color=['#ef4444', '#10b981'],
                height=0.5,
                edgecolor='none'
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability', color='#6b6f8d', fontsize=9)
            ax.tick_params(colors='#a0a3c0', labelsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.set_tick_params(color='#1e2240')
            ax.set_facecolor('#13152a')
            for bar, val in zip(bars, [1 - pred_proba, pred_proba]):
                ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val*100:.1f}%', va='center', color='#e8e8f0', fontsize=10, fontweight='bold')
            ax.set_title('Prediction Probabilities', color='#c7c9e8', fontsize=10, pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Key factors
        st.markdown('<div class="section-head">🔍 Key Factors Affecting This Prediction</div>', unsafe_allow_html=True)
        coef  = metrics['coef']
        input_dict = input_data.iloc[0].to_dict()
        impacts = {k: coef[k] * input_dict[k] for k in coef}
        top_pos = sorted(impacts.items(), key=lambda x: x[1], reverse=True)[:4]
        top_neg = sorted(impacts.items(), key=lambda x: x[1])[:3]

        col_p, col_n = st.columns(2)
        with col_p:
            st.markdown("**✅ Helping your placement:**")
            for feat, val in top_pos:
                st.markdown(f"- `{feat}` &nbsp; (+{val:.2f})")
        with col_n:
            st.markdown("**⚠️ Hurting your placement:**")
            for feat, val in top_neg:
                st.markdown(f"- `{feat}` &nbsp; ({val:.2f})")

    else:
        st.info("👈 Fill in the student details in the sidebar and click **Predict Placement**")

# ══════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    # Confusion Matrix
    with col_a:
        st.markdown('<div class="section-head">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#13152a')
        ax.set_facecolor('#13152a')
        cm = metrics['cm']
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Not Placed', 'Placed'], color='#a0a3c0')
        ax.set_yticklabels(['Not Placed', 'Placed'], color='#a0a3c0')
        ax.set_xlabel('Predicted', color='#6b6f8d')
        ax.set_ylabel('Actual', color='#6b6f8d')
        ax.set_title('Confusion Matrix', color='#c7c9e8', pad=12)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white', fontsize=16, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <small style='color:#6b6f8d;'>
        ▪ <b style='color:#a0a3c0'>Top-left</b>: Correctly predicted "Not Placed"<br>
        ▪ <b style='color:#a0a3c0'>Bottom-right</b>: Correctly predicted "Placed"<br>
        ▪ <b style='color:#a0a3c0'>Off-diagonal</b>: Prediction errors
        </small>
        """, unsafe_allow_html=True)

    # Feature Coefficients
    with col_b:
        st.markdown('<div class="section-head">Feature Importance (Coefficients)</div>', unsafe_allow_html=True)
        coef_df = pd.DataFrame(list(metrics['coef'].items()), columns=['Feature', 'Coeff'])
        coef_df = coef_df.reindex(coef_df['Coeff'].abs().sort_values(ascending=True).index)

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#13152a')
        ax.set_facecolor('#13152a')
        colors = ['#10b981' if v > 0 else '#ef4444' for v in coef_df['Coeff']]
        ax.barh(coef_df['Feature'], coef_df['Coeff'], color=colors, edgecolor='none', height=0.6)
        ax.axvline(0, color='#2a2d50', linewidth=1)
        ax.set_xlabel('Coefficient', color='#6b6f8d', fontsize=9)
        ax.tick_params(colors='#a0a3c0', labelsize=8)
        ax.set_title('Feature Coefficients', color='#c7c9e8', pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        green_patch = mpatches.Patch(color='#10b981', label='Boosts placement')
        red_patch   = mpatches.Patch(color='#ef4444', label='Hurts placement')
        ax.legend(handles=[green_patch, red_patch], facecolor='#13152a',
                  labelcolor='#a0a3c0', fontsize=8, framealpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Classification report
    st.markdown('<div class="section-head">Classification Report</div>', unsafe_allow_html=True)
    report = metrics['report']
    report_df = pd.DataFrame({
        'Class'    : ['Not Placed', 'Placed', 'Macro Avg'],
        'Precision': [report['Not Placed']['precision'], report['Placed']['precision'], report['macro avg']['precision']],
        'Recall'   : [report['Not Placed']['recall'],    report['Placed']['recall'],    report['macro avg']['recall']],
        'F1-Score' : [report['Not Placed']['f1-score'],  report['Placed']['f1-score'],  report['macro avg']['f1-score']],
        'Support'  : [int(report['Not Placed']['support']), int(report['Placed']['support']), int(report['macro avg']['support'])],
    }).set_index('Class')
    st.dataframe(report_df.style.format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-Score': '{:.3f}'}),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        # CGPA distribution by placement
        st.markdown('<div class="section-head">CGPA Distribution by Placement</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#13152a')
        ax.set_facecolor('#13152a')
        placed     = df[df['placed'] == 1]['cgpa']
        not_placed = df[df['placed'] == 0]['cgpa']
        ax.hist(placed, bins=20, alpha=0.7, color='#10b981', label='Placed', edgecolor='none')
        ax.hist(not_placed, bins=20, alpha=0.7, color='#ef4444', label='Not Placed', edgecolor='none')
        ax.set_xlabel('CGPA', color='#6b6f8d'); ax.set_ylabel('Count', color='#6b6f8d')
        ax.tick_params(colors='#a0a3c0', labelsize=8)
        ax.legend(facecolor='#13152a', labelcolor='#a0a3c0', fontsize=8)
        ax.set_title('CGPA vs Placement', color='#c7c9e8', pad=10)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        # Placement by branch
        st.markdown('<div class="section-head">Placement Rate by Branch</div>', unsafe_allow_html=True)
        branch_stats = df.copy()
        le2 = LabelEncoder()
        branch_stats['branch_name'] = le2.fit_transform(df['branch'])
        # Reload original for display
        orig = pd.read_csv("student_placement_salary_elite_v2.csv")
        branch_rate = orig.groupby('branch')['placed'].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#13152a')
        ax.set_facecolor('#13152a')
        bars = ax.barh(branch_rate.index, branch_rate.values * 100, color='#818cf8', edgecolor='none', height=0.5)
        ax.set_xlabel('Placement Rate (%)', color='#6b6f8d')
        ax.tick_params(colors='#a0a3c0', labelsize=8)
        ax.set_title('Placement Rate by Branch', color='#c7c9e8', pad=10)
        for bar, val in zip(bars, branch_rate.values):
            ax.text(val * 100 + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', color='#c7c9e8', fontsize=8)
        ax.set_xlim(0, 105)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        # Internships vs placement
        st.markdown('<div class="section-head">Internships vs Placement Rate</div>', unsafe_allow_html=True)
        int_rate = orig.groupby('internships')['placed'].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#13152a')
        ax.set_facecolor('#13152a')
        ax.plot(int_rate.index, int_rate.values, color='#818cf8', marker='o',
                linewidth=2.5, markersize=8, markerfacecolor='#4f46e5')
        ax.fill_between(int_rate.index, int_rate.values, alpha=0.15, color='#818cf8')
        ax.set_xlabel('Number of Internships', color='#6b6f8d')
        ax.set_ylabel('Placement Rate (%)', color='#6b6f8d')
        ax.tick_params(colors='#a0a3c0', labelsize=8)
        ax.set_title('More Internships → Higher Placement', color='#c7c9e8', pad=10)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.grid(True, alpha=0.1, color='#2a2d50')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        # College tier vs placement
        st.markdown('<div class="section-head">College Tier vs Placement Rate</div>', unsafe_allow_html=True)
        tier_rate = orig.groupby('college_tier')['placed'].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#13152a')
        ax.set_facecolor('#13152a')
        colors_tier = ['#10b981', '#818cf8', '#f59e0b']
        bars = ax.bar([f'Tier {t}' for t in tier_rate.index], tier_rate.values,
                      color=colors_tier, edgecolor='none', width=0.5)
        ax.set_ylabel('Placement Rate (%)', color='#6b6f8d')
        ax.tick_params(colors='#a0a3c0', labelsize=9)
        ax.set_title('College Tier vs Placement', color='#c7c9e8', pad=10)
        for bar, val in zip(bars, tier_rate.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:.1f}%', ha='center', color='#e8e8f0', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 115)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Raw data preview
    st.markdown('<div class="section-head">📋 Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(orig.head(10), use_container_width=True, height=300)

# Footer
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem; color:#3a3d5c; font-size:0.8rem;'>
    Built with Logistic Regression · scikit-learn · Streamlit &nbsp;|&nbsp; 9,000 student records
</div>
""", unsafe_allow_html=True)
