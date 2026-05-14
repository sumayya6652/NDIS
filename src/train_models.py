# import sys
# import pandas as pd
# import numpy as np
# import joblib
# from pathlib import Path

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier

# # Make sure Python can import src.rules when running from project root
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# from src.rules import classify_access_risk_from_input


# DATA_PATH = Path("data/processed/ndis_model_data.csv")
# MODEL_DIR = Path("models")
# RESULTS_DIR = Path("results")

# MODEL_DIR.mkdir(exist_ok=True)
# RESULTS_DIR.mkdir(exist_ok=True)


# # ============================================================
# # SUPPORT RECOMMENDATION TARGET
# # ============================================================

# def create_recommended_support(row):
#     """
#     Project-defined support recommendation.
#     This is not an official NDIS decision.
#     It is used for explainable AI prototype/demo purposes.
#     """

#     disability = str(row.get("disability_type", "")).lower()
#     utilisation_gap = float(row.get("utilisation_gap", 0))
#     participant_to_provider_ratio = float(row.get("participant_to_provider_ratio", 0))
#     active_provider_count = float(row.get("active_provider_count", 1))
#     average_support_budget = float(row.get("average_support_budget", 0))

#     if utilisation_gap >= 40 and participant_to_provider_ratio >= 50:
#         return "Support Coordination / Plan Management Review"

#     if active_provider_count <= 5 or participant_to_provider_ratio >= 80:
#         return "Provider Availability Review"

#     if "autism" in disability or "development" in disability:
#         return "Capacity Building - Improved Daily Living"

#     if "intellectual" in disability:
#         return "Capacity Building - Social and Community Participation"

#     if "psychosocial" in disability:
#         return "Psychosocial Recovery and Community Participation"

#     if "physical" in disability or "mobility" in disability:
#         return "Assistive Technology / Transport Support"

#     if "hearing" in disability:
#         return "Assistive Technology and Communication Support"

#     if "visual" in disability or "vision" in disability:
#         return "Assistive Technology and Orientation Support"

#     if "brain" in disability:
#         return "Capacity Building and Daily Living Support Review"

#     if "neurological" in disability:
#         return "Therapy, Assistive Technology, and Daily Living Support Review"

#     if average_support_budget >= 60000:
#         return "High-Intensity Support Review"

#     if utilisation_gap >= 25:
#         return "Plan Utilisation Support"

#     return "General Core Support Monitoring"


# # ============================================================
# # SYNTHETIC DEMO CASES
# # ============================================================

# def create_synthetic_demo_cases():
#     """
#     Adds scenario-based examples so the ML model learns low, medium, and high-risk patterns.
#     This is useful because the public NDIS dataset may be aggregated and incomplete.
#     """

#     states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
#     age_groups = ["0 to 8", "9 to 14", "15 to 24", "25 to 34", "35 to 44", "45 to 54", "55 to 64"]
#     disabilities = [
#         "Autism",
#         "Intellectual Disability",
#         "Psychosocial Disability",
#         "Physical Disability",
#         "Developmental Delay",
#         "Acquired Brain Injury",
#         "Hearing Impairment",
#         "Visual Impairment",
#         "Neurological Disability",
#         "Other / Unknown"
#     ]

#     rows = []

#     for state in states:
#         for age in age_groups:
#             for disability in disabilities:

#                 # Low-risk scenario
#                 rows.append({
#                     "state": state,
#                     "age_group": age,
#                     "disability_type": disability,
#                     "participant_count": 200,
#                     "average_support_budget": 25000,
#                     "utilisation_rate": 85,
#                     "active_provider_count": 80,
#                     "participant_to_provider_ratio": 200 / 80,
#                     "utilisation_gap": 15,
#                     "payment_per_participant": 20000
#                 })

#                 # Medium-risk scenario
#                 rows.append({
#                     "state": state,
#                     "age_group": age,
#                     "disability_type": disability,
#                     "participant_count": 500,
#                     "average_support_budget": 40000,
#                     "utilisation_rate": 65,
#                     "active_provider_count": 25,
#                     "participant_to_provider_ratio": 500 / 25,
#                     "utilisation_gap": 35,
#                     "payment_per_participant": 15000
#                 })

#                 # High-risk scenario
#                 rows.append({
#                     "state": state,
#                     "age_group": age,
#                     "disability_type": disability,
#                     "participant_count": 1000,
#                     "average_support_budget": 60000,
#                     "utilisation_rate": 30,
#                     "active_provider_count": 8,
#                     "participant_to_provider_ratio": 1000 / 8,
#                     "utilisation_gap": 70,
#                     "payment_per_participant": 8000
#                 })

#     synthetic_df = pd.DataFrame(rows)

#     synthetic_df["access_risk"] = synthetic_df.apply(
#         classify_access_risk_from_input,
#         axis=1
#     )

#     synthetic_df["recommended_support"] = synthetic_df.apply(
#         create_recommended_support,
#         axis=1
#     )

#     return synthetic_df


# # ============================================================
# # DATA PREPARATION FOR TRAINING
# # ============================================================

# def prepare_training_data(df):
#     df = df.copy()

#     required_columns = [
#         "state",
#         "age_group",
#         "disability_type",
#         "participant_count",
#         "average_support_budget",
#         "utilisation_rate",
#         "active_provider_count",
#         "participant_to_provider_ratio",
#         "utilisation_gap",
#         "payment_per_participant"
#     ]

#     for col in required_columns:
#         if col not in df.columns:
#             if col in ["state", "age_group", "disability_type"]:
#                 df[col] = "Other / Unknown"
#             else:
#                 df[col] = 0

#     for col in [
#         "participant_count",
#         "average_support_budget",
#         "utilisation_rate",
#         "active_provider_count",
#         "participant_to_provider_ratio",
#         "utilisation_gap",
#         "payment_per_participant"
#     ]:
#         df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

#     df["state"] = df["state"].astype(str)
#     df["age_group"] = df["age_group"].astype(str)

#     # If disability type is missing or only Unknown, keep it but synthetic cases will add useful categories.
#     df["disability_type"] = df["disability_type"].astype(str)
#     df.loc[df["disability_type"].str.lower().isin(["unknown", "nan", "none", ""]), "disability_type"] = "Other / Unknown"

#     # Recreate access risk using the same explainable rule as the app.
#     df["access_risk"] = df.apply(classify_access_risk_from_input, axis=1)

#     # Recreate recommendation label.
#     df["recommended_support"] = df.apply(create_recommended_support, axis=1)

#     synthetic_df = create_synthetic_demo_cases()

#     combined_df = pd.concat([df, synthetic_df], ignore_index=True)

#     combined_df = combined_df.dropna(subset=required_columns)

#     return combined_df


# # ============================================================
# # SAFE SPLIT
# # ============================================================

# def safe_train_test_split(X, y):
#     class_counts = y.value_counts()

#     if y.nunique() < 2:
#         raise ValueError(f"Target has only one class: {y.unique()}")

#     if class_counts.min() >= 2:
#         return train_test_split(
#             X,
#             y,
#             test_size=0.2,
#             random_state=42,
#             stratify=y
#         )

#     return train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42
#     )


# # ============================================================
# # TRAIN MODEL
# # ============================================================

# def train_and_evaluate_model(df, target, task_name):
#     features = [
#         "state",
#         "age_group",
#         "disability_type",
#         "participant_count",
#         "average_support_budget",
#         "utilisation_rate",
#         "active_provider_count",
#         "participant_to_provider_ratio",
#         "utilisation_gap",
#         "payment_per_participant"
#     ]

#     model_df = df[features + [target]].dropna()

#     X = model_df[features]
#     y = model_df[target].astype(str)

#     print("\n" + "=" * 80)
#     print(f"Target distribution for {task_name}")
#     print("=" * 80)
#     print(y.value_counts())

#     categorical_features = ["state", "age_group", "disability_type"]

#     numeric_features = [
#         "participant_count",
#         "average_support_budget",
#         "utilisation_rate",
#         "active_provider_count",
#         "participant_to_provider_ratio",
#         "utilisation_gap",
#         "payment_per_participant"
#     ]

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
#             ("num", "passthrough", numeric_features)
#         ]
#     )

#     models = {
#         "Logistic Regression": LogisticRegression(max_iter=3000, solver="lbfgs"),
#         "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
#         "Gradient Boosting": GradientBoostingClassifier(random_state=42),
#         "Neural Network": MLPClassifier(
#             hidden_layer_sizes=(64, 32),
#             max_iter=600,
#             random_state=42
#         )
#     }

#     X_train, X_test, y_train, y_test = safe_train_test_split(X, y)

#     results = []
#     trained_models = {}

#     for model_name, model in models.items():
#         pipeline = Pipeline([
#             ("preprocessor", preprocessor),
#             ("model", model)
#         ])

#         try:
#             pipeline.fit(X_train, y_train)
#             predictions = pipeline.predict(X_test)

#             accuracy = accuracy_score(y_test, predictions)
#             macro_f1 = f1_score(y_test, predictions, average="macro")

#             results.append({
#                 "task": task_name,
#                 "model": model_name,
#                 "accuracy": accuracy,
#                 "macro_f1": macro_f1
#             })

#             print("\n" + "=" * 80)
#             print(f"{task_name} - {model_name}")
#             print("=" * 80)
#             print("Accuracy:", accuracy)
#             print("Macro-F1:", macro_f1)
#             print(classification_report(y_test, predictions, zero_division=0))

#             trained_models[model_name] = pipeline

#         except Exception as e:
#             print(f"\n{task_name} - {model_name} failed:")
#             print(e)

#     results_df = pd.DataFrame(results)

#     if results_df.empty:
#         raise ValueError(f"No model trained successfully for {task_name}")

#     best_row = results_df.sort_values("macro_f1", ascending=False).iloc[0]
#     best_model_name = best_row["model"]
#     best_model = trained_models[best_model_name]

#     print(f"\nBest model for {task_name}: {best_model_name}")

#     return results_df, best_model


# # ============================================================
# # FEATURE IMPORTANCE
# # ============================================================

# def save_feature_importance(model, output_path):
#     try:
#         final_model = model.named_steps["model"]
#         preprocessor = model.named_steps["preprocessor"]

#         if not hasattr(final_model, "feature_importances_"):
#             print("Feature importance not available for this model.")
#             return

#         feature_names = preprocessor.get_feature_names_out()
#         importances = final_model.feature_importances_

#         importance_df = pd.DataFrame({
#             "feature": feature_names,
#             "importance": importances
#         }).sort_values("importance", ascending=False)

#         importance_df.to_csv(output_path, index=False)
#         print("Feature importance saved to:", output_path)

#     except Exception as e:
#         print("Could not save feature importance:", e)


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     if not DATA_PATH.exists():
#         raise FileNotFoundError(
#             "Processed dataset not found. Run this first:\n"
#             "python src/prepare_data.py"
#         )

#     raw_df = pd.read_csv(DATA_PATH)

#     print("\nLoaded processed data:")
#     print("Original shape:", raw_df.shape)

#     training_df = prepare_training_data(raw_df)

#     print("\nTraining data after adding scenario examples:")
#     print("Training shape:", training_df.shape)

#     training_df.to_csv("data/processed/ndis_training_data_with_scenarios.csv", index=False)

#     print("\nAccess risk distribution:")
#     print(training_df["access_risk"].value_counts())

#     print("\nRecommended support distribution:")
#     print(training_df["recommended_support"].value_counts())

#     risk_results, risk_model = train_and_evaluate_model(
#         training_df,
#         target="access_risk",
#         task_name="Access Risk Prediction"
#     )

#     support_results, support_model = train_and_evaluate_model(
#         training_df,
#         target="recommended_support",
#         task_name="Support Recommendation"
#     )

#     all_results = pd.concat([risk_results, support_results], ignore_index=True)
#     all_results.to_csv(RESULTS_DIR / "model_results.csv", index=False)

#     joblib.dump(risk_model, MODEL_DIR / "risk_model.pkl")
#     joblib.dump(support_model, MODEL_DIR / "support_model.pkl")

#     save_feature_importance(
#         risk_model,
#         RESULTS_DIR / "risk_feature_importance.csv"
#     )

#     save_feature_importance(
#         support_model,
#         RESULTS_DIR / "support_feature_importance.csv"
#     )

#     print("\nSaved files:")
#     print("- models/risk_model.pkl")
#     print("- models/support_model.pkl")
#     print("- results/model_results.csv")
#     print("- data/processed/ndis_training_data_with_scenarios.csv")

#     print("\nTraining completed successfully.")


# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

DATA_PATH = Path("data/processed/ndis_model_data.csv")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def classify_access_risk(row):
    participant_count = float(row.get("participant_count", 0))
    active_provider_count = float(row.get("active_provider_count", 1))
    utilisation_rate = float(row.get("utilisation_rate", 0))
    average_support_budget = float(row.get("average_support_budget", 0))

    if active_provider_count <= 0:
        active_provider_count = 1

    ratio = participant_count / active_provider_count
    utilisation_gap = 100 - utilisation_rate

    score = 0

    if ratio >= 80:
        score += 3
    elif ratio >= 40:
        score += 2
    elif ratio >= 20:
        score += 1

    if utilisation_gap >= 50:
        score += 3
    elif utilisation_gap >= 30:
        score += 2
    elif utilisation_gap >= 20:
        score += 1

    if active_provider_count <= 5:
        score += 2
    elif active_provider_count <= 15:
        score += 1

    if average_support_budget >= 60000:
        score += 1

    if score >= 5:
        return "High Risk"
    elif score >= 2:
        return "Medium Risk"
    return "Low Risk"


def create_recommended_support(row):
    disability = str(row.get("disability_type", "")).lower()
    support_class = str(row.get("support_class", "")).lower()
    utilisation_gap = float(row.get("utilisation_gap", 0))
    ratio = float(row.get("participant_to_provider_ratio", 0))
    active_provider_count = float(row.get("active_provider_count", 1))
    budget = float(row.get("average_support_budget", 0))

    if utilisation_gap >= 40 and ratio >= 50:
        return "Support Coordination / Plan Management Review"

    if active_provider_count <= 5 or ratio >= 80:
        return "Provider Availability Review"

    if "autism" in disability or "developmental" in disability:
        return "Capacity Building - Improved Daily Living"

    if "intellectual" in disability:
        return "Capacity Building - Social and Community Participation"

    if "psychosocial" in disability:
        return "Psychosocial Recovery and Community Participation"

    if "physical" in disability or "spinal" in disability:
        return "Assistive Technology / Transport Support"

    if "hearing" in disability or "visual" in disability or "sensory" in disability:
        return "Assistive Technology and Communication Support"

    if "brain" in disability or "neurological" in disability or "stroke" in disability:
        return "Therapy and Daily Living Support Review"

    if "capacity" in support_class:
        return "Capacity Building Support Review"

    if "core" in support_class:
        return "Core Support Monitoring"

    if budget >= 60000:
        return "High-Intensity Support Review"

    if utilisation_gap >= 25:
        return "Plan Utilisation Support"

    return "General Core Support Monitoring"


def create_random_scenarios(n=1200, seed=42):
    rng = np.random.default_rng(seed)

    states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "ACT", "NT"]
    ages = ["0 to 8", "9 to 14", "15 to 24", "25 to 34", "35 to 44", "45 to 54", "55 to 64", "65+"]
    disabilities = [
        "Autism",
        "Intellectual Disability",
        "Psychosocial Disability",
        "Physical Disability",
        "Developmental Delay",
        "Acquired Brain Injury",
        "Hearing Impairment",
        "Visual Impairment",
        "Neurological Disability",
        "Other / Unknown"
    ]
    support_classes = [
        "Core Support",
        "Capacity Building",
        "Therapeutic Support",
        "Assistive Technology",
        "Support Coordination",
        "Community Participation"
    ]

    rows = []

    for _ in range(n):
        participant_count = int(rng.integers(50, 1600))
        active_provider_count = int(rng.integers(3, 130))
        utilisation_rate = float(rng.uniform(25, 95))
        budget = float(rng.integers(12000, 85000))
        payment = float(rng.integers(5000, 60000))

        row = {
            "state": rng.choice(states),
            "age_group": rng.choice(ages),
            "disability_type": rng.choice(disabilities),
            "support_class": rng.choice(support_classes),
            "participant_count": participant_count,
            "average_support_budget": budget,
            "utilisation_rate": utilisation_rate,
            "active_provider_count": active_provider_count,
            "participant_to_provider_ratio": participant_count / active_provider_count,
            "utilisation_gap": 100 - utilisation_rate,
            "payment_per_participant": payment
        }

        row["access_risk"] = classify_access_risk(row)
        row["recommended_support"] = create_recommended_support(row)
        rows.append(row)

    return pd.DataFrame(rows)


def prepare_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run python src/prepare_data.py first.")

    real_df = pd.read_csv(DATA_PATH)

    required_cols = [
        "state",
        "age_group",
        "disability_type",
        "support_class",
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "participant_to_provider_ratio",
        "utilisation_gap",
        "payment_per_participant"
    ]

    for col in required_cols:
        if col not in real_df.columns:
            real_df[col] = "Other / Unknown" if col in ["state", "age_group", "disability_type", "support_class"] else 0

    for col in [
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "participant_to_provider_ratio",
        "utilisation_gap",
        "payment_per_participant"
    ]:
        real_df[col] = pd.to_numeric(real_df[col], errors="coerce").fillna(0)

    real_df["access_risk"] = real_df.apply(classify_access_risk, axis=1)
    real_df["recommended_support"] = real_df.apply(create_recommended_support, axis=1)

    scenario_df = create_random_scenarios()

    training_df = pd.concat([real_df[required_cols + ["access_risk", "recommended_support"]], scenario_df], ignore_index=True)

    training_df.to_csv("data/processed/ndis_training_data_with_scenarios.csv", index=False)

    return training_df


def train_model(df, target, task_name):
    features = [
        "state",
        "age_group",
        "disability_type",
        "support_class",
        "participant_count",
        "average_support_budget",
        "utilisation_rate",
        "active_provider_count",
        "participant_to_provider_ratio",
        "utilisation_gap",
        "payment_per_participant"
    ]

    X = df[features]
    y = df[target].astype(str)

    categorical = ["state", "age_group", "disability_type", "support_class"]
    numeric = [col for col in features if col not in categorical]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    results = []
    trained = {}

    print("\n" + "=" * 80)
    print(task_name)
    print("=" * 80)
    print(y.value_counts())

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")

        results.append({
            "task": task_name,
            "model": name,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4)
        })

        trained[name] = pipe

        print("\n", name)
        print("Accuracy:", round(acc, 4))
        print("Macro-F1:", round(f1, 4))
        print(classification_report(y_test, pred, zero_division=0))

    selected = trained["Random Forest"]
    return pd.DataFrame(results), selected


def save_feature_importance(model, output_path):
    try:
        final_model = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]

        if not hasattr(final_model, "feature_importances_"):
            return

        feature_names = preprocessor.get_feature_names_out()
        importances = final_model.feature_importances_

        out = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        out.to_csv(output_path, index=False)
    except Exception as e:
        print("Feature importance failed:", e)


def main():
    df = prepare_training_data()

    risk_results, risk_model = train_model(df, "access_risk", "Access Risk Prediction")
    support_results, support_model = train_model(df, "recommended_support", "Support Recommendation")

    results = pd.concat([risk_results, support_results], ignore_index=True)
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    joblib.dump(risk_model, MODEL_DIR / "risk_model.pkl")
    joblib.dump(support_model, MODEL_DIR / "support_model.pkl")

    save_feature_importance(risk_model, RESULTS_DIR / "risk_feature_importance.csv")
    save_feature_importance(support_model, RESULTS_DIR / "support_feature_importance.csv")

    print("\nSaved models and results successfully.")


if __name__ == "__main__":
    main()