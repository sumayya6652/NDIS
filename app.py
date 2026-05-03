import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

from src.rules import (
    classify_access_risk_from_input,
    explain_access_risk,
    rule_based_recommendation
)

DATA_PATH = Path("data/processed/ndis_model_data.csv")
RISK_MODEL_PATH = Path("models/risk_model.pkl")
SUPPORT_MODEL_PATH = Path("models/support_model.pkl")
RESULTS_PATH = Path("results/model_results.csv")
RISK_IMPORTANCE_PATH = Path("results/risk_feature_importance.csv")
SUPPORT_IMPORTANCE_PATH = Path("results/support_feature_importance.csv")
ASSETS_DIR = Path("assets")

st.set_page_config(
    page_title="NDIS Recommender AI",
    page_icon="♿",
    layout="wide"
)


# ============================================================
# CLEANING HELPERS
# ============================================================

def clean_disability_label(value):
    if pd.isna(value):
        return "Other / Unknown"

    text = str(value).strip()

    if text == "":
        return "Other / Unknown"

    lower = text.lower().strip()

    missing_values = [
        "unknown",
        "nan",
        "none",
        "missing",
        "dsbltygrp_missing",
        "dsbltygrp missing",
        "not stated",
        "not specified",
        "other / unknown"
    ]

    if lower in missing_values:
        return "Other / Unknown"

    mapping = {
        "autism": "Autism",
        "autism spectrum disorder": "Autism",
        "intellectual disability": "Intellectual Disability",
        "developmental delay": "Developmental Delay",
        "global developmental delay": "Global Developmental Delay",
        "down syndrome": "Down Syndrome",
        "psychosocial disability": "Psychosocial Disability",
        "physical disability": "Physical Disability",
        "other physical": "Other Physical",
        "neurological disability": "Neurological Disability",
        "other neurological": "Other Neurological",
        "multiple sclerosis": "Multiple Sclerosis",
        "stroke": "Stroke",
        "spinal cord injury": "Spinal Cord Injury",
        "acquired brain injury": "Acquired Brain Injury",
        "hearing impairment": "Hearing Impairment",
        "visual impairment": "Visual Impairment",
        "other sensory/speech": "Other Sensory/Speech",
        "other": "Other"
    }

    if lower in mapping:
        return mapping[lower]

    return text.title()


def risk_colour(risk):
    risk = str(risk).lower()

    if "low" in risk:
        return "#2E8B57"
    if "medium" in risk:
        return "#F4A261"
    if "high" in risk:
        return "#D62828"

    return "#6C757D"


def risk_numeric_value(risk):
    risk = str(risk).lower()

    if "low" in risk:
        return 25
    if "medium" in risk:
        return 55
    if "high" in risk:
        return 85

    return 50


def calculate_risk_components(profile):
    participant_count = float(profile.get("participant_count", 0))
    active_provider_count = float(profile.get("active_provider_count", 1))
    utilisation_rate = float(profile.get("utilisation_rate", 0))
    average_support_budget = float(profile.get("average_support_budget", 0))

    if active_provider_count <= 0:
        active_provider_count = 1

    participant_to_provider_ratio = participant_count / active_provider_count
    utilisation_gap = 100 - utilisation_rate

    demand_pressure = min(participant_to_provider_ratio / 100 * 100, 100)
    utilisation_pressure = min(utilisation_gap, 100)

    if active_provider_count <= 5:
        provider_scarcity = 100
    elif active_provider_count <= 15:
        provider_scarcity = 70
    elif active_provider_count <= 30:
        provider_scarcity = 45
    else:
        provider_scarcity = 15

    budget_complexity = min(average_support_budget / 70000 * 100, 100)

    return pd.DataFrame({
        "Risk Factor": [
            "Demand Pressure",
            "Utilisation Gap",
            "Provider Scarcity",
            "Budget Complexity"
        ],
        "Score": [
            demand_pressure,
            utilisation_pressure,
            provider_scarcity,
            budget_complexity
        ]
    })


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("Processed dataset not found. Run: python src/prepare_data.py")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    if "disability_type" not in df.columns:
        df["disability_type"] = "Other / Unknown"

    df["disability_type"] = df["disability_type"].apply(clean_disability_label)

    numeric_cols = [
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "participant_to_provider_ratio",
        "utilisation_gap",
        "payment_per_participant",
        "payment_amount"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_models():
    if not RISK_MODEL_PATH.exists() or not SUPPORT_MODEL_PATH.exists():
        st.error("Model files not found. Run: python src/train_models.py")
        st.stop()

    risk_model = joblib.load(RISK_MODEL_PATH)
    support_model = joblib.load(SUPPORT_MODEL_PATH)

    return risk_model, support_model


def make_input_dataframe(profile):
    return pd.DataFrame([profile])


def get_top_support_predictions(model, input_df, top_n=3):
    try:
        probabilities = model.predict_proba(input_df)[0]

        if hasattr(model, "named_steps"):
            classes = model.named_steps["model"].classes_
        else:
            classes = model.classes_

        top_indices = np.argsort(probabilities)[::-1][:top_n]

        return pd.DataFrame({
            "AI Recommended Support": classes[top_indices],
            "Confidence Score": probabilities[top_indices]
        })

    except Exception:
        try:
            prediction = model.predict(input_df)[0]

            return pd.DataFrame({
                "AI Recommended Support": [prediction],
                "Confidence Score": ["Not available"]
            })

        except Exception:
            return pd.DataFrame({
                "AI Recommended Support": ["Model recommendation unavailable"],
                "Confidence Score": ["Not available"]
            })


def find_similar_groups(df, profile, top_n=5):
    temp = df.copy()

    required_cols = [
        "state",
        "age_group",
        "disability_type",
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "access_risk"
    ]

    for col in required_cols:
        if col not in temp.columns:
            temp[col] = "Unknown"

    temp["similarity_score"] = 0.0

    temp["similarity_score"] += (
        temp["state"].astype(str) == str(profile["state"])
    ).astype(int) * 2

    temp["similarity_score"] += (
        temp["age_group"].astype(str) == str(profile["age_group"])
    ).astype(int) * 2

    temp["similarity_score"] += (
        temp["disability_type"].astype(str).str.lower()
        == str(profile["disability_type"]).lower()
    ).astype(int) * 3

    temp["average_support_budget"] = pd.to_numeric(
        temp["average_support_budget"], errors="coerce"
    ).fillna(0)

    temp["utilisation_rate"] = pd.to_numeric(
        temp["utilisation_rate"], errors="coerce"
    ).fillna(0)

    temp["participant_count"] = pd.to_numeric(
        temp["participant_count"], errors="coerce"
    ).fillna(0)

    temp["active_provider_count"] = pd.to_numeric(
        temp["active_provider_count"], errors="coerce"
    ).fillna(1)

    temp["budget_difference"] = abs(
        temp["average_support_budget"] - profile["average_support_budget"]
    )

    temp["utilisation_difference"] = abs(
        temp["utilisation_rate"] - profile["utilisation_rate"]
    )

    temp["participant_difference"] = abs(
        temp["participant_count"] - profile["participant_count"]
    )

    temp["similarity_score"] -= temp["budget_difference"].rank(pct=True)
    temp["similarity_score"] -= temp["utilisation_difference"].rank(pct=True)
    temp["similarity_score"] -= temp["participant_difference"].rank(pct=True)

    display_cols = [
        "state",
        "age_group",
        "disability_type",
        "participant_count",
        "active_provider_count",
        "utilisation_rate",
        "access_risk",
        "similarity_score"
    ]

    if "support_class" in temp.columns:
        display_cols.insert(3, "support_class")

    return temp.sort_values("similarity_score", ascending=False)[display_cols].head(top_n)


def show_banner():
    banner_files = [
        ASSETS_DIR / "ndis_banner.png",
        ASSETS_DIR / "ndis_banner.jpeg",
        ASSETS_DIR / "banner.png",
        ASSETS_DIR / "banner.jpg"
    ]

    shown = False

    for file in banner_files:
        if file.exists():
            st.image(str(file), use_container_width=True)
            shown = True
            break

    if not shown:
        st.markdown(
            """
            <div style="
                padding: 35px;
                border-radius: 18px;
                background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
                color: white;
                margin-bottom: 20px;
            ">
                <h1 style="margin-bottom: 0;">NDIS Recommender AI</h1>
                <p style="font-size: 18px;">
                    Explainable recommendation and decision-support system for disability service access.
                </p>
                <p style="font-size: 14px;">
                    Public aggregated data • Rule-based reasoning • Machine learning comparison • Responsible AI
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


def create_risk_gauge(risk):
    value = risk_numeric_value(risk)
    colour = risk_colour(risk)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "Access Risk Gauge"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": colour},
            "steps": [
                {"range": [0, 35], "color": "#D8F3DC"},
                {"range": [35, 70], "color": "#FFE8A3"},
                {"range": [70, 100], "color": "#FFCCD5"}
            ],
            "threshold": {
                "line": {"color": colour, "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

    return fig


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

df = load_data()
risk_model, support_model = load_models()

for col in ["state", "age_group", "disability_type"]:
    if col not in df.columns:
        df[col] = "Unknown"

df["state"] = df["state"].astype(str)
df["age_group"] = df["age_group"].astype(str)
df["disability_type"] = df["disability_type"].apply(clean_disability_label)


# ============================================================
# HEADER
# ============================================================

show_banner()

st.info(
    "This app is a prototype for group-level NDIS decision support. "
    "It does not make individual funding decisions. "
    "It combines public aggregated data, rule-based reasoning, visual analytics, and machine learning comparison."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Project Overview",
    "Data Explorer",
    "Visual Analytics",
    "Recommendation System",
    "Model Evaluation",
    "Ethics and Limitations"
])


# ============================================================
# TAB 1: PROJECT OVERVIEW
# ============================================================

with tab1:
    st.header("Project Overview")

    col_img, col_text = st.columns([1, 2])

    with col_img:
        image_files = [
            ASSETS_DIR / "accessibility.png",
            ASSETS_DIR / "accessibility.jpg",
            ASSETS_DIR / "support.png",
            ASSETS_DIR / "support.jpg"
        ]

        image_found = False

        for image_file in image_files:
            if image_file.exists():
                st.image(str(image_file), caption="Disability service access and support planning", use_container_width=True)
                image_found = True
                break

        if not image_found:
            st.markdown(
                """
                <div style="
                    height: 260px;
                    border-radius: 18px;
                    background: linear-gradient(135deg, #1d3557, #457b9d);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 80px;
                ">
                    ♿
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_text:
        st.write("""
        This project addresses a real-world disability service access problem. 
        NDIS participants may have approved support funding but still experience difficulty accessing services due to:
        
        - provider availability
        - low plan utilisation
        - location and regional service gaps
        - participant support needs
        - complex disability support categories
        """)

        st.write("""
        The system integrates:
        
        1. Machine learning for access-risk comparison  
        2. Rule-based reasoning for transparent risk classification  
        3. Recommendation logic for support suggestions  
        4. Visual analytics for service planning  
        """)

    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows in Dataset", f"{len(df):,}")
    col2.metric("States/Territories", df["state"].nunique())
    col3.metric("Age Groups", df["age_group"].nunique())
    col4.metric("Disability Groups", df["disability_type"].nunique())

    st.subheader("System Workflow")

    workflow = pd.DataFrame({
        "Step": [
            "1. Data Collection",
            "2. Data Cleaning",
            "3. Feature Engineering",
            "4. Risk Classification",
            "5. Recommendation",
            "6. Explanation"
        ],
        "Description": [
            "NDIS participant, provider, utilisation and payment data are collected.",
            "Raw columns are standardised and merged into one modelling dataset.",
            "Features such as utilisation gap and participant-to-provider ratio are created.",
            "The system predicts low, medium or high service access risk.",
            "The system recommends support planning actions.",
            "Rule-based reasoning explains why the recommendation was made."
        ]
    })

    st.dataframe(workflow, use_container_width=True)


# ============================================================
# TAB 2: DATA EXPLORER
# ============================================================

with tab2:
    st.header("Data Explorer")

    st.write("Preview of the processed modelling dataset:")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Dataset Columns")
    st.write(list(df.columns))

    col_a, col_b = st.columns(2)

    with col_a:
        if "access_risk" in df.columns:
            st.subheader("Access Risk Counts")
            st.dataframe(df["access_risk"].value_counts().reset_index().rename(
                columns={"index": "Access Risk", "access_risk": "Count"}
            ))

    with col_b:
        if "disability_type" in df.columns:
            st.subheader("Top Disability Groups")
            st.dataframe(df["disability_type"].value_counts().head(15).reset_index().rename(
                columns={"index": "Disability Group", "disability_type": "Count"}
            ))


# ============================================================
# TAB 3: VISUAL ANALYTICS
# ============================================================

with tab3:
    st.header("Visual Analytics Dashboard")

    st.write("These visualisations help explain service access risk patterns in the processed dataset.")

    if "access_risk" in df.columns:
        risk_counts = df["access_risk"].value_counts().reset_index()
        risk_counts.columns = ["Access Risk", "Count"]

        fig_risk_pie = px.pie(
            risk_counts,
            values="Count",
            names="Access Risk",
            title="Access Risk Distribution",
            hole=0.45
        )

        st.plotly_chart(fig_risk_pie, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if "disability_type" in df.columns:
            disability_counts = df["disability_type"].value_counts().head(12).reset_index()
            disability_counts.columns = ["Disability Group", "Count"]

            fig_disability = px.bar(
                disability_counts,
                x="Count",
                y="Disability Group",
                orientation="h",
                title="Top Disability Groups in Dataset"
            )

            fig_disability.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_disability, use_container_width=True)

    with col2:
        if "state" in df.columns and "participant_count" in df.columns:
            state_participants = (
                df.groupby("state")["participant_count"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig_state = px.bar(
                state_participants,
                x="state",
                y="participant_count",
                title="Participant Count by State/Territory",
                labels={"state": "State", "participant_count": "Participant Count"}
            )

            st.plotly_chart(fig_state, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if "state" in df.columns and "participant_to_provider_ratio" in df.columns:
            ratio_by_state = (
                df.groupby("state")["participant_to_provider_ratio"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig_ratio = px.bar(
                ratio_by_state,
                x="state",
                y="participant_to_provider_ratio",
                title="Average Participant-to-Provider Ratio by State",
                labels={
                    "state": "State",
                    "participant_to_provider_ratio": "Participant-to-Provider Ratio"
                }
            )

            st.plotly_chart(fig_ratio, use_container_width=True)

    with col4:
        if "state" in df.columns and "utilisation_gap" in df.columns:
            gap_by_state = (
                df.groupby("state")["utilisation_gap"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig_gap = px.bar(
                gap_by_state,
                x="state",
                y="utilisation_gap",
                title="Average Utilisation Gap by State",
                labels={
                    "state": "State",
                    "utilisation_gap": "Utilisation Gap (%)"
                }
            )

            st.plotly_chart(fig_gap, use_container_width=True)

    if all(col in df.columns for col in ["participant_count", "active_provider_count", "utilisation_rate", "access_risk"]):
        scatter_df = df.copy()
        scatter_df["participant_count"] = pd.to_numeric(scatter_df["participant_count"], errors="coerce")
        scatter_df["active_provider_count"] = pd.to_numeric(scatter_df["active_provider_count"], errors="coerce")
        scatter_df["utilisation_rate"] = pd.to_numeric(scatter_df["utilisation_rate"], errors="coerce")

        scatter_df = scatter_df.dropna(subset=["participant_count", "active_provider_count", "utilisation_rate"])

        fig_scatter = px.scatter(
            scatter_df,
            x="active_provider_count",
            y="participant_count",
            size="utilisation_rate",
            color="access_risk",
            hover_data=["state", "age_group", "disability_type"],
            title="Participant Demand vs Provider Availability",
            labels={
                "active_provider_count": "Active Provider Count",
                "participant_count": "Participant Count",
                "utilisation_rate": "Utilisation Rate"
            }
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    if all(col in df.columns for col in ["access_risk", "utilisation_rate"]):
        fig_box = px.box(
            df,
            x="access_risk",
            y="utilisation_rate",
            title="Plan Utilisation Rate by Access Risk",
            labels={
                "access_risk": "Access Risk",
                "utilisation_rate": "Utilisation Rate (%)"
            }
        )

        st.plotly_chart(fig_box, use_container_width=True)


# ============================================================
# TAB 4: RECOMMENDATION SYSTEM
# ============================================================

with tab4:
    st.header("AI Recommendation System")

    st.write(
        "Enter a participant-group profile. "
        "The system will recommend support options and predict access risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        state_options = sorted(df["state"].dropna().unique())
        age_options = sorted(df["age_group"].dropna().unique())

        if len(state_options) == 0:
            state_options = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]

        if len(age_options) == 0:
            age_options = [
                "0 to 8",
                "9 to 14",
                "15 to 24",
                "25 to 34",
                "35 to 44",
                "45 to 54",
                "55 to 64",
                "65+"
            ]

        dataset_disabilities = (
            df["disability_type"]
            .dropna()
            .astype(str)
            .apply(clean_disability_label)
            .unique()
            .tolist()
        )

        manual_disabilities = [
            "Autism",
            "Intellectual Disability",
            "Psychosocial Disability",
            "Physical Disability",
            "Developmental Delay",
            "Global Developmental Delay",
            "Down Syndrome",
            "Acquired Brain Injury",
            "Hearing Impairment",
            "Visual Impairment",
            "Neurological Disability",
            "Multiple Sclerosis",
            "Stroke",
            "Spinal Cord Injury",
            "Other Neurological",
            "Other Physical",
            "Other Sensory/Speech",
            "Other / Unknown"
        ]

        disability_options = sorted(set(dataset_disabilities + manual_disabilities))

        if "Other / Unknown" in disability_options:
            disability_options.remove("Other / Unknown")
            disability_options.append("Other / Unknown")

        state = st.selectbox("State", state_options)
        age_group = st.selectbox("Age Group", age_options)
        disability_type = st.selectbox("Disability Type", disability_options)

    with col2:
        participant_count = st.number_input(
            "Participant Count",
            min_value=1,
            value=200,
            step=50
        )

        average_support_budget = st.number_input(
            "Average Support Budget",
            min_value=0,
            value=25000,
            step=1000
        )

        utilisation_rate = st.slider(
            "Plan Utilisation Rate (%)",
            min_value=0,
            max_value=100,
            value=85
        )

        active_provider_count = st.number_input(
            "Active Provider Count",
            min_value=1,
            value=80,
            step=5
        )

        payment_per_participant = st.number_input(
            "Payment Per Participant",
            min_value=0,
            value=20000,
            step=1000
        )

    participant_to_provider_ratio = participant_count / active_provider_count
    utilisation_gap = 100 - utilisation_rate

    profile = {
        "state": state,
        "age_group": age_group,
        "disability_type": disability_type,
        "participant_count": participant_count,
        "average_support_budget": average_support_budget,
        "utilisation_rate": utilisation_rate,
        "active_provider_count": active_provider_count,
        "participant_to_provider_ratio": participant_to_provider_ratio,
        "utilisation_gap": utilisation_gap,
        "payment_per_participant": payment_per_participant
    }

    input_df = make_input_dataframe(profile)

    st.subheader("Calculated Indicators")

    ind1, ind2, ind3 = st.columns(3)

    ind1.metric("Participant-to-Provider Ratio", round(participant_to_provider_ratio, 2))
    ind2.metric("Utilisation Gap", f"{round(utilisation_gap, 2)}%")
    ind3.metric("Payment Per Participant", f"${payment_per_participant:,.0f}")

    if st.button("Generate AI Recommendation"):
        st.subheader("Service Access Risk Prediction")

        final_rule_risk = classify_access_risk_from_input(profile)

        try:
            ml_risk = risk_model.predict(input_df)[0]
        except Exception:
            ml_risk = "Model prediction unavailable"

        risk_col1, risk_col2 = st.columns([1, 1])

        with risk_col1:
            st.metric("Final Explainable Access Risk", final_rule_risk)
            st.metric("ML Model Prediction", ml_risk)

        with risk_col2:
            st.plotly_chart(create_risk_gauge(final_rule_risk), use_container_width=True)

        st.write("Explanation:")
        st.write(explain_access_risk(profile))

        st.caption(
            "The final access-risk result uses transparent rule-based reasoning. "
            "The ML prediction is shown as a comparison because the training label is project-defined."
        )

        st.subheader("Risk Factor Breakdown")

        risk_components = calculate_risk_components(profile)

        fig_components = px.bar(
            risk_components,
            x="Risk Factor",
            y="Score",
            title="Risk Factor Contribution",
            range_y=[0, 100]
        )

        st.plotly_chart(fig_components, use_container_width=True)

        st.subheader("Rule-Based Support Suggestions")

        suggestions = rule_based_recommendation(profile)

        for suggestion in suggestions:
            st.write("- " + suggestion)

        st.subheader("AI Support Recommendation Model")

        support_predictions = get_top_support_predictions(
            support_model,
            input_df,
            top_n=3
        )

        st.dataframe(support_predictions, use_container_width=True)

        if "Confidence Score" in support_predictions.columns:
            try:
                chart_df = support_predictions.copy()
                chart_df["Confidence Score"] = pd.to_numeric(chart_df["Confidence Score"], errors="coerce")

                fig_support = px.bar(
                    chart_df,
                    x="Confidence Score",
                    y="AI Recommended Support",
                    orientation="h",
                    title="Top AI Support Recommendations"
                )

                fig_support.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_support, use_container_width=True)
            except Exception:
                pass

        st.subheader("Similar Participant Groups from Dataset")

        similar_groups = find_similar_groups(df, profile, top_n=5)
        st.dataframe(similar_groups, use_container_width=True)


# ============================================================
# TAB 5: MODEL EVALUATION
# ============================================================

with tab5:
    st.header("Model Evaluation")

    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)

        st.subheader("Model Results")
        st.dataframe(results, use_container_width=True)

        if "macro_f1" in results.columns:
            st.subheader("Macro-F1 by Model")

            chart_data = results.pivot_table(
                index="model",
                columns="task",
                values="macro_f1",
                aggfunc="mean"
            )

            st.bar_chart(chart_data)

            fig_eval = px.bar(
                results,
                x="model",
                y="macro_f1",
                color="task",
                barmode="group",
                title="Model Comparison by Macro-F1"
            )

            st.plotly_chart(fig_eval, use_container_width=True)

        st.write("""
        Macro-F1 is included because risk categories may be imbalanced. 
        Accuracy alone can be misleading if one category dominates the dataset.
        """)

    else:
        st.warning("Model results file not found. Run: python src/train_models.py")

    st.subheader("Feature Importance")

    importance_files = {
        "Risk Model Feature Importance": RISK_IMPORTANCE_PATH,
        "Support Model Feature Importance": SUPPORT_IMPORTANCE_PATH
    }

    for title, path in importance_files.items():
        if path.exists():
            importance_df = pd.read_csv(path).head(15)

            st.write(title)
            st.dataframe(importance_df, use_container_width=True)

            if "feature" in importance_df.columns and "importance" in importance_df.columns:
                fig_importance = px.bar(
                    importance_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=title
                )

                fig_importance.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_importance, use_container_width=True)


# ============================================================
# TAB 6: ETHICS AND LIMITATIONS
# ============================================================

with tab6:
    st.header("Ethics and Limitations")

    col_ethics, col_icon = st.columns([2, 1])

    with col_ethics:
        st.write("""
        This system is a prototype and should be treated as a decision-support tool only.
        It should not be used to make individual NDIS funding or eligibility decisions.
        """)

        st.subheader("Limitations")

        st.write("""
        - The model uses public and aggregated data, not individual participant records.
        - The access-risk label is project-defined, not an official NDIS outcome.
        - The ML prediction may differ from the rule-based result because the training label is project-defined.
        - The system cannot fully capture lived experience, informal care, provider quality, cultural needs, transport barriers, or participant preference.
        """)

        st.subheader("Responsible AI Safeguards")

        st.write("""
        - The app explains why a risk level is assigned.
        - The final risk output uses transparent rule-based reasoning.
        - The ML result is shown only as a comparison.
        - The system is designed to support human review, not replace planners, providers, or disability advocates.
        - Recommendations should be interpreted carefully and ethically.
        """)

    with col_icon:
        st.markdown(
            """
            <div style="
                height: 300px;
                border-radius: 18px;
                background: linear-gradient(135deg, #283618, #606C38);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 90px;
            ">
                ⚖️
            </div>
            """,
            unsafe_allow_html=True
        )