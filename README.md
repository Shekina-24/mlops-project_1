# Bias in Bios — Profession Classifier & Gender Fairness Audit

A CV/résumé profession classifier that works well on average — and an explicit, precomputed audit of exactly where and how it discriminates by gender.

## The problem

Companies increasingly use automated tools to screen résumés and CVs, sometimes predicting attributes like a candidate's profession or seniority directly from free text. Research — notably [*Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting*](https://arxiv.org/abs/1901.09451) (De-Arteaga et al., FAccT 2019) — has shown that such models reproduce gender stereotypes present in their training data: they are, for example, less accurate at predicting "surgeon" for a biography written by a woman, or "nurse" for one written by a man.

This project does two things, not one:

1. **Trains a real classifier** that predicts a person's profession from the free text of their biography.
2. **Audits that classifier for gender bias** and makes the audit visible and understandable in the final application — this is the core of the project, not an afterthought.

The message: *here is a CV-sorting model that performs reasonably well overall — and here, precisely, is where and how it discriminates by gender, and how we measure it.*

## Dataset

[LabHC/bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios) — MIT licensed, ~396,189 rows across `train` (257,478) / `dev` (39,642) / `test` (99,069). Each row has:

- `hard_text`: a free-text biography (English)
- `profession`: one of 28 profession labels (int 0–27; not exposed as an HF `ClassLabel`, so this project hardcodes the name mapping from the [dataset card](https://huggingface.co/datasets/LabHC/bias_in_bios), see `src/get-data.py`)
- `gender`: binary sensitive attribute (0 = male, 1 = female)

To keep training fast on a CPU-only machine, the processed dataset used here is a **stratified sub-sample** (by `profession` **and** `gender` jointly, so the per-profession gender balance is preserved) of about 40,000 rows total, proportionally split across the original train/dev/test splits. This is controlled entirely by `configs/config.yaml` (`sampling.total_samples`, `sampling.stratify_by`), not hardcoded.

**Known limitation on name/gender leakage**: the original Bias in Bios study scrubs explicit first names from the start of most biographies, and this project does not re-clean the text further. Gendered pronouns, titles ("Mr.", "Ms.", "Dr."), and occasional names later in the text can still leak gender information into the model. This is a known, documented limitation — see [Limitations](#limitations--ethical-considerations) below.

## Methodology

### ML pipeline

- **Vectorization**: `TfidfVectorizer` (scikit-learn), unigrams + bigrams, `max_features=30000`.
- **Model**: multinomial `LogisticRegression`, compared against `LinearSVC` (wrapped in `CalibratedClassifierCV` for probability outputs) via a small `RandomizedSearchCV` over `clf__C` (kept intentionally cheap — a handful of candidates, 3-fold CV — so the whole pipeline trains in a few minutes on CPU).
- Everything is config-driven (`configs/config.yaml`): model type, hyperparameter grid, TF-IDF params, sub-sampling ratio, and split ratios are parameters, not hardcoded.

### Fairness audit methodology

Computed once, after training, with [fairlearn](https://fairlearn.org/) (`src/evaluate.py`), using `gender` as the sensitive feature on the held-out **test** split:

- Overall accuracy / F1 macro.
- Accuracy and macro recall **by gender**, via a `fairlearn.metrics.MetricFrame`.
- **True Positive Rate (TPR) gap per profession**: for each profession, the recall (TPR) is computed separately for male and female subjects (using `MetricFrame` with `control_features=true_profession`), and the absolute difference is the headline fairness metric — this is exactly the metric used in the original Bias in Bios paper.
- The professions with the **largest male/female imbalance in the training data itself** are automatically identified (sorted by `|pct_female − 0.5|`) and highlighted, since these are the cases most likely to produce a biased model.

The resulting `fairness_report.json` (and `metrics.json`) are **precomputed artifacts**, pushed to the Hugging Face model repo alongside the model. The Streamlit app reads and displays them directly — it never recomputes fairness metrics on the fly.

## Results

Trained on 25,997 rows, evaluated on the held-out `test` split (9,999 rows). Best model: Logistic Regression, `C=10.0`.

### Overall performance

| Split | Accuracy | F1 macro |
|---|---|---|
| Dev (4,002 rows) | 0.789 | 0.675 |
| **Test (9,999 rows)** | **0.784** | **0.682** |

### Fairness — by gender (test split)

| Gender | n | Accuracy | Recall (macro) |
|---|---|---|---|
| Male | 5,390 | 0.781 | 0.584 |
| Female | 4,609 | 0.787 | 0.595 |

Overall accuracy is close between genders — which is exactly why a per-profession breakdown matters: the two errors mostly cancel out in the aggregate, hiding large opposite-direction gaps underneath.

### Fairness — TPR gap by profession (test split)

Mean absolute TPR gap across all 28 professions: **0.130**. Largest gap: **0.406** (profession: "model").

Selected professions with a strong, paper-consistent bias pattern:

| Profession | Male TPR | Female TPR | Gap | Direction |
|---|---|---|---|---|
| surgeon | 0.634 | 0.353 | 0.281 | under-recognizes female surgeons |
| dietitian | 0.429 | 0.753 | 0.324 | under-recognizes male dietitians |
| nurse | 0.682 | 0.814 | 0.132 | under-recognizes male nurses |
| software_engineer | 0.544 | 0.393 | 0.151 | under-recognizes female software engineers |
| dj | 0.625 | 0.400 | 0.225 | under-recognizes female DJs |
| model | 0.364 | 0.769 | 0.406 | under-recognizes male models (largest gap overall) |

This reproduces the classic Bias in Bios pattern: for stereotypically male-coded professions (surgeon, software engineer), the model is noticeably worse at recognizing *female* members of that profession, and vice versa for stereotypically female-coded ones (nurse, dietitian).

Full per-profession breakdown: `data/processed/fairness_report.json` after running `make all`, or in the [model repo](https://huggingface.co/S12-24/bias-in-bios-profession-classifier) on the Hub, or interactively in the app's Bias Dashboard tab.

## Links

- **Model on the Hugging Face Hub**: [S12-24/bias-in-bios-profession-classifier](https://huggingface.co/S12-24/bias-in-bios-profession-classifier) — pipeline (skops), model card, `fairness_report.json`, `metrics.json`.
- **Live demo**: deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) from this repo's `app/` folder — see [Deploying the app](#deploying-the-app) below for the one-click setup (link added here once deployed).

## Reproducing locally

Requires Python 3.10+ and a HF account (with a write-access token) if you want to push the trained model.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

huggingface-cli login            # only needed to push to the HF Hub

make all                         # data -> preprocess -> train -> evaluate -> test
```

Individual steps (see `Makefile`):

```bash
make data        # download LabHC/bias_in_bios -> data/raw/
make preprocess  # clean + stratified sub-sample -> data/processed/
make train       # train + tune + push to HF Hub (add --no-push to skip the push)
make evaluate    # test-set evaluation + fairness audit -> data/processed/, pushed to the Hub
make test        # pytest tests/
make app         # run the Streamlit app locally
```

All paths, split ratios, sampling size, model type/hyperparameters, and HF repo IDs are set in `configs/config.yaml`.

## Deploying the app

The `app/` folder is self-contained (its own `app.py` + `requirements.txt`) and loads the model + fairness report straight from the Hugging Face Hub, so it has no dependency on `src/` at runtime.

**Streamlit Community Cloud (used for this project — free)**:
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. "New app" → repository `Shekina-24/mlops-project_1`, branch `main`, main file path `app/app.py`.
3. Deploy. Streamlit Cloud auto-redeploys on every push to `main`.

**Hugging Face Spaces (alternative)**: Streamlit Spaces require a Space with compute (Gradio/Docker/Streamlit SDK), which needs a paid HF PRO plan as of this writing — free HF accounts are limited to Static Spaces. If you have a PRO plan, create a Space with SDK `streamlit`, and push the contents of `app/` to it (the app is already structured for this — `app/README.md` includes the Space metadata header).

## Limitations & ethical considerations

- **This is a pedagogical / research project.** The model, the app, and every artifact in this repo are intended to study and demonstrate algorithmic gender bias — **not** to be used for real hiring, résumé screening, or any employment decision.
- **Binary gender only.** The dataset encodes gender as binary (male/female), which erases non-binary identities. This is a limitation inherited from the original Bias in Bios study, not something this project can fix downstream.
- **Sub-sampling trade-off.** Training on ~40k rows (vs. the full ~396k) keeps CPU training fast but reduces statistical precision, especially for small professions (e.g. `personal_trainer`, `dj`, `paralegal` have well under 100 test examples) — read the fairness gaps as informative signals, not precise population estimates.
- **Possible indirect gender leakage.** The dataset scrubs explicit first names from the start of most biographies, but pronouns, honorifics, and occasional names elsewhere in the text can still leak gender to the model. This means the model may partly be learning "gendered language patterns" rather than purely profession-relevant content — which is itself part of what the fairness audit is trying to surface, but worth being explicit about.
- **Aggregate accuracy hides bias.** As shown above, overall accuracy is nearly identical between genders (0.781 vs 0.787) while individual professions show TPR gaps up to 0.41 — a reminder that a single global fairness number is not sufficient to characterize a model's bias.

## License

MIT — see [LICENSE](LICENSE). The underlying dataset ([LabHC/bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios)) is also MIT licensed.
