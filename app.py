import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from fpdf import FPDF
import time

# ------------------------------
# Load Model & Scaler
# ------------------------------
MODEL_PATH = "outputs/models/"
MODEL_FILE = MODEL_PATH + "FD004_RandomForest.pkl"
SCALER_FILE = MODEL_PATH + "FD004_scaler.pkl"

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

feature_cols = ["operational_setting_1", "operational_setting_2", "operational_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]

# ------------------------------
# Prediction Function
# ------------------------------
def make_predictions(df):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]
    return preds, probs

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# ------------------------------
# Custom CSS (Black / Gray theme)
# ------------------------------
st.markdown("""
<style>
body {
    background-color: #ECECEC;
    color: #000000;
}
.stSidebar {
    width: 28% !important;
    background-color: #F8F8F8;  
    padding:15px;
    border-radius:10px;
}

/* Benefit Cards */
.card {
    background-color: #000000;
    color: #FFFFFF;
    padding:20px;
    border-radius:12px;
    text-align:center;
    margin:10px;
    font-weight:bold;
    transition: transform 0.4s, box-shadow 0.4s;
}
.card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 10px 20px rgba(0,0,0,0.4);
}

/* Section Titles */
.section-title {
    background-color:#000000;
    color:#FFFFFF;
    font-size:20px;
    font-weight:bold;
    padding:12px;
    border-radius:6px;
    margin-bottom:12px;
}
.section-title:hover {
    transform: scale(1.03);
    box-shadow: 0 5px 15px rgba(0,0,0,0.4);
}

/* Alerts & Recommendations */
.alerts {
    border-left:5px solid #000000;
    padding-left:10px;
    margin-bottom:10px;
}
.recommendations {
    border-left:5px solid #000000;
    padding-left:10px;
    margin-bottom:10px;
}

/* Table Styling */
.stDataFrame table {
    background-color:#FDFDFD;
    color:#000000;
}
.stDataFrame th {
    background-color:#D9D9D9;
    color:#000000;
}
.stDataFrame td:hover {
    background-color:#CFCFCF;
    color:#000000;
    transition: all 0.2s;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Dashboard Header / Benefits
# ------------------------------
st.title("AI Predictive Maintenance Dashboard")
st.markdown('<div class="section"></div>', unsafe_allow_html=True)

benefits = ["🔍 Detect issues early", "⏳ Extend engine lifetime",
            "💸 Cut maintenance costs", "📊 Make data-driven decisions"]
col1, col2 = st.columns(2)
for i, col in enumerate([col1, col2]*2):
    if i < len(benefits):
        col.markdown(f'<div class="card">{benefits[i]}</div>', unsafe_allow_html=True)

# ------------------------------
# Sidebar Input & PDF Download
# ------------------------------
st.sidebar.markdown(
    '<div style="font-size:20px; font-weight:bold; color:#000000">Choose Input Mode:</div>',
    unsafe_allow_html=True
)
mode = st.sidebar.radio("", ["📂 Upload CSV", "⌨️ Manual Input"])

df = None

if mode == "📂 Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with engine data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.sidebar.markdown("### 📄 Preview Uploaded Data")
        st.sidebar.dataframe(df.head())
elif mode == "⌨️ Manual Input":
    st.sidebar.subheader("Enter Engine Sensor Readings")
    manual_input = {}
    for col in ["operational_setting_1", "operational_setting_2", "operational_setting_3",
                "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5"]:
        manual_input[col] = st.sidebar.number_input(f"{col}", value=0.0)
    df = pd.DataFrame([manual_input])

# ------------------------------
# Run Predictions
# ------------------------------
if df is not None:
    preds, probs = make_predictions(df)
    df["Prediction"] = preds
    df["Failure_Probability"] = np.round(probs, 3)

    st.markdown(
    '<div style="background-color:#000000; color:#FFFFFF; display:inline-block; padding:12px 18px; border-radius:8px; font-size:22px; font-weight:bold; margin-bottom:10px; margin-top:40px;">'
    '🔍 Prediction Results'
    '</div>',
    unsafe_allow_html=True
    )
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)

    # Animate metrics count up
    engines_at_risk = int((preds == 1).sum())
    healthy_engines = int((preds == 0).sum())

    risk_placeholder = col1.empty()
    healthy_placeholder = col2.empty()

    for i in range(0, engines_at_risk+1):
        risk_placeholder.metric("⚠️ Engines at Risk", i)
        time.sleep(0.02)
    for i in range(0, healthy_engines+1):
        healthy_placeholder.metric("✅ Healthy Engines", i)
        time.sleep(0.02)

    st.markdown(
    '<div style="background-color:#000000; color:#FFFFFF; display:inline-block; padding:12px 18px; border-radius:8px; font-size:22px; font-weight:bold; margin-bottom:10px; margin-top:10px;">'
    '📈 Engine Health Insights'
    '</div>',
    unsafe_allow_html=True
    )

    # Charts in black/gray
    hist_data = df["Failure_Probability"]
    fig1 = px.histogram(hist_data, nbins=20,
                        title="Failure Probability Distribution",
                        color_discrete_sequence=["black"])
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(df, names="Prediction",
                  title="Engine Health Status (0=Healthy,1=At Risk)",
                  color_discrete_sequence=["gray","black"])
    st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------
    # Alerts & Recommendations
    # ------------------------------
    col_alert, col_recom = st.columns(2)
    at_risk = df[df["Prediction"] == 1]

    with col_alert:
        st.markdown('<div class="section-title">🚨 Alerts</div>', unsafe_allow_html=True)
        if not at_risk.empty:
            for i, row in at_risk.iterrows():
                st.error(f"Engine {i} predicted to fail soon ⚠️ (Prob: {row['Failure_Probability']})")
        else:
            st.success("✅ All engines currently healthy.")

    with col_recom:
        st.markdown('<div class="section-title">💡 Recommendations</div>', unsafe_allow_html=True)
        if not at_risk.empty:
            st.info("Schedule inspection & preventive maintenance within next 10 cycles.")
        else:
            st.info("Continue monitoring engines regularly.")

    st.sidebar.markdown('<div style="font-size:22px; font-weight:bold; margin-top:15px;">📄 Full Report Preview</div>', unsafe_allow_html=True)

    # ------------------------------
    # PDF Generation
    # ------------------------------
    def create_pdf_with_plots(dataframe, hist_fig, pie_fig, filename="maintenance_report.pdf"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "AI Predictive Maintenance Report", ln=True, align="C")
        pdf.ln(10)

        # Save Plotly figures as PNG
        hist_file, pie_file = "hist_plot.png", "pie_plot.png"
        hist_fig.write_image(hist_file)
        pie_fig.write_image(pie_file)
        pdf.image(hist_file, x=15, w=180)
        pdf.ln(5)
        pdf.image(pie_file, x=40, w=120)
        pdf.ln(10)

        # Table with multiple pages if many cols
        pdf.set_font("Arial", "B", 10)
        pdf.set_fill_color(230, 230, 230)

        page_width = pdf.w - 2 * pdf.l_margin
        cols_per_page = 8
        row_height = pdf.font_size * 1.5

        for start in range(0, len(dataframe.columns), cols_per_page):
            subset_cols = dataframe.columns[start:start+cols_per_page]

            # Header
            col_width = page_width / len(subset_cols)
            for col_name in subset_cols:
                pdf.cell(col_width, row_height, col_name, border=1, fill=True, align="C")
            pdf.ln(row_height)

            # Rows
            pdf.set_font("Arial", "", 9)
            for row in dataframe.itertuples(index=False):
                for item in row[start:start+cols_per_page]:
                    pdf.cell(col_width, row_height, str(item), border=1)
                pdf.ln(row_height)

            pdf.ln(5)

        # Summary
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(
            0, 10,
            f"Total Engines: {len(dataframe)}\n"
            f"Engines at Risk: {(dataframe['Prediction']==1).sum()}\n"
            f"Healthy Engines: {(dataframe['Prediction']==0).sum()}\n"
            f"Avg Failure Probability: {dataframe['Failure_Probability'].mean():.2f}"
        )

        pdf.output(filename)

    create_pdf_with_plots(df, fig1, fig2)

    with open("maintenance_report.pdf", "rb") as f:
        st.sidebar.download_button("📥 Download Full Report PDF", f, file_name="maintenance_report.pdf", mime="application/pdf")
