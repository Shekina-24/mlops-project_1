"""Streamlit demo app for the Bias in Bios profession classifier.

Deployed to a Hugging Face Space. Loads the trained pipeline and the
precomputed fairness report from the Hugging Face Hub model repo (never
recomputes fairness metrics at request time).
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO = "S12-24/bias-in-bios-profession-classifier"
DATASET_URL = "https://huggingface.co/datasets/LabHC/bias_in_bios"
PAPER_URL = "https://arxiv.org/abs/1901.09451"

st.set_page_config(page_title="Bias in Bios — Profession Classifier", page_icon="⚖️", layout="wide")

DISCLAIMER = (
    "**Research / educational demo.** This app studies gender bias in an automated résumé/CV "
    "profession classifier trained on the *Bias in Bios* dataset. It is **not** a hiring tool and "
    "must **not** be used to screen real candidates or make employment decisions."
)


@st.cache_resource(show_spinner="Loading model from the Hugging Face Hub...")
def load_pipeline():
    import skops.io as sio

    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.skops")
    untrusted = sio.get_untrusted_types(file=model_path)
    return sio.load(model_path, trusted=untrusted)


@st.cache_data(show_spinner=False)
def load_json_artifact(filename: str) -> dict:
    path = hf_hub_download(repo_id=MODEL_REPO, filename=filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_profession_names() -> list[str]:
    label_names = load_json_artifact("label_names.json")
    return label_names["profession"]


def predict_top_k(pipeline, text: str, profession_names: list[str], k: int = 5) -> pd.DataFrame:
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    top_idx = proba.argsort()[::-1][:k]
    rows = [
        {"profession": profession_names[classes[i]], "probability": float(proba[i])}
        for i in top_idx
    ]
    return pd.DataFrame(rows)


def grouped_gender_bar(categories, male_values, female_values, title, y_title):
    fig = go.Figure()
    fig.add_bar(name="Male", x=categories, y=male_values, marker_color="#4C78A8")
    fig.add_bar(name="Female", x=categories, y=female_values, marker_color="#F58518")
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        barmode="group",
        legend_title_text="Gender",
        margin=dict(t=50, b=40),
    )
    return fig


st.title("⚖️ Bias in Bios — Profession Classifier & Fairness Audit")
st.warning(DISCLAIMER)

tab_predict, tab_bias, tab_about = st.tabs(["🔍 Prediction", "📊 Bias Dashboard", "ℹ️ About / Methodology"])

# --------------------------------------------------------------------------------------
# Tab 1: Prediction
# --------------------------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict a profession from a biography")
    st.caption(
        "Paste an English-language biography or CV summary (similar in style to the LinkedIn-style "
        "bios in the training data). The model predicts one of 28 professions."
    )

    default_example = (
        "She completed her residency in internal medicine and now practices as a hospitalist, "
        "focusing on complex diagnostic cases and patient safety initiatives."
    )
    text_input = st.text_area("Biography text", value=default_example, height=160)

    if st.button("Predict profession", type="primary"):
        if not text_input.strip():
            st.error("Please enter some text first.")
        else:
            with st.spinner("Loading model and predicting..."):
                pipeline = load_pipeline()
                profession_names = get_profession_names()
                results = predict_top_k(pipeline, text_input, profession_names, k=5)

            st.success(f"Predicted profession: **{results.iloc[0]['profession']}**")

            fig = go.Figure(
                go.Bar(
                    x=results["probability"][::-1],
                    y=results["profession"][::-1],
                    orientation="h",
                    marker_color="#4C78A8",
                    text=[f"{p:.1%}" for p in results["probability"][::-1]],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Top-5 predicted professions",
                xaxis_title="Probability",
                xaxis_tickformat=".0%",
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------
# Tab 2: Bias Dashboard
# --------------------------------------------------------------------------------------
with tab_bias:
    st.subheader("Gender fairness audit (precomputed on the held-out test set)")
    st.caption(
        "All numbers below come from `fairness_report.json`, computed once during `src/evaluate.py` "
        "with [fairlearn](https://fairlearn.org/), and are not recalculated on the fly."
    )

    try:
        report = load_json_artifact("fairness_report.json")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load the fairness report from the Hub: {e}")
        st.stop()

    overall = report["overall"]
    by_gender = report["by_gender"]
    gap_summary = report["tpr_gap_summary"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall accuracy", f"{overall['accuracy']:.1%}")
    col2.metric("Overall F1 macro", f"{overall['f1_macro']:.3f}")
    col3.metric("Mean |TPR gap|", f"{gap_summary['mean_abs_tpr_gap']:.3f}")
    col4.metric(
        "Largest TPR gap",
        f"{gap_summary['max_tpr_gap']:.3f}",
        help=f"Profession: {gap_summary['max_tpr_gap_profession']}",
    )

    st.markdown("#### Accuracy and recall by gender (overall)")
    genders = ["male", "female"]
    acc_vals = [by_gender[g]["accuracy"] for g in genders]
    recall_vals = [by_gender[g]["recall_macro"] for g in genders]
    fig_overall = go.Figure()
    fig_overall.add_bar(name="Accuracy", x=["Male", "Female"], y=acc_vals, marker_color="#4C78A8")
    fig_overall.add_bar(name="Recall (macro)", x=["Male", "Female"], y=recall_vals, marker_color="#F58518")
    fig_overall.update_layout(barmode="group", yaxis_tickformat=".0%", margin=dict(t=20, b=40))
    st.plotly_chart(fig_overall, use_container_width=True)

    st.markdown("#### True Positive Rate (recall) gap by profession")
    st.caption(
        "For each profession, the True Positive Rate is the share of people truly in that profession "
        "who were correctly predicted. A large gap between male and female TPR means the model is "
        "much better at recognizing one gender in that profession than the other."
    )

    tpr_df = pd.DataFrame(report["tpr_by_profession_gender"]).dropna(subset=["male_tpr", "female_tpr"])
    tpr_df = tpr_df.sort_values("tpr_gap", ascending=False)

    highlight_only = st.checkbox(
        "Show only the most gender-imbalanced professions in the dataset", value=True
    )
    plot_df = tpr_df[tpr_df["is_highlighted_imbalanced"]] if highlight_only else tpr_df.head(15)

    fig_tpr = grouped_gender_bar(
        plot_df["profession"],
        plot_df["male_tpr"],
        plot_df["female_tpr"],
        title="TPR (recall) by profession and gender",
        y_title="True Positive Rate",
    )
    fig_tpr.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_tpr, use_container_width=True)

    with st.expander("Full per-profession TPR table"):
        st.dataframe(
            tpr_df[["profession", "male_tpr", "female_tpr", "tpr_gap"]].reset_index(drop=True),
            use_container_width=True,
        )

    st.markdown("#### Gender imbalance in the training data")
    st.caption(
        "Professions where the male/female split in the dataset itself is most skewed — these are "
        "the cases most likely to produce a biased model."
    )
    dist_df = pd.DataFrame(report["gender_distribution_by_profession"]).sort_values(
        "imbalance", ascending=False
    ).head(10)
    fig_dist = grouped_gender_bar(
        dist_df["profession"], dist_df["male"], dist_df["female"], title="Sample count by gender", y_title="Count"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# --------------------------------------------------------------------------------------
# Tab 3: About / Methodology
# --------------------------------------------------------------------------------------
with tab_about:
    st.warning(DISCLAIMER)

    st.markdown(
        f"""
### The problem

Companies increasingly use automated tools to screen résumés and CVs, sometimes predicting
attributes like a candidate's profession or seniority directly from free text. Research —
notably [*Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes
Setting*]({PAPER_URL}) (De-Arteaga et al., FAccT 2019) — has shown that such models reproduce
gender stereotypes from their training data: they are, for example, less accurate at predicting
"surgeon" for a biography written by a woman, or "nurse" for one written by a man.

This project trains a real occupation classifier on the same data and **audits it explicitly for
gender bias**, rather than treating fairness as an afterthought.

### Dataset

[LabHC/bias_in_bios]({DATASET_URL}) — ~396k English-language biographies, each labeled with one of
28 professions and a binary gender attribute. Released under the MIT license. This app's model was
trained on a **stratified sub-sample** (by profession and gender) of about 40,000 rows, to keep
training fast on a CPU-only machine.

### Methodology

**Classifier**: TF-IDF vectorizer + Logistic Regression (scikit-learn), with hyperparameters chosen
via a small randomized search. Trained and evaluated with the code in this project's
[GitHub repository](https://github.com/Shekina-24/mlops-project_1).

**Fairness audit**: computed with [fairlearn](https://fairlearn.org/), using `gender` as the
sensitive feature. The headline metric is the **True Positive Rate (TPR) gap** per profession — the
difference in recall between men and women for a given true profession — which is precisely the
metric used in the original Bias in Bios paper. All fairness numbers shown in the Bias Dashboard tab
are precomputed once (`src/evaluate.py`) and shipped as a JSON artifact alongside the model; the app
never recomputes them from raw predictions.

### Limitations & ethical considerations

- **Not for production hiring use.** This is a research/education project, full stop.
- **Only binary gender** is available in this dataset, which erases non-binary identities and is
  itself a simplification the original study inherited.
- **Sub-sampling** trades off some statistical precision (especially for rare, small professions)
  for fast CPU training — the fairness gaps shown should be read as informative signals, not
  precise population estimates.
- **Possible indirect gender leakage**: while the dataset's biographies are scrubbed of names at the
  start of the text in most cases, first names, pronouns, and other cues can appear later in some
  bios, meaning the model may partly learn from gendered language rather than profession-relevant
  content alone. See the project README for a fuller discussion.
- Predicting someone's profession from free text is itself a use case that can enable discriminatory
  screening; publishing this app is meant to make that risk visible and measurable, not to endorse it.
        """
    )
