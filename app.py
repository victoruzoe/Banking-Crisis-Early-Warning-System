from __future__ import annotations

import pandas as pd
import streamlit as st

from src.modeling import (
    RAW_FEATURE_COLUMNS,
    build_model_features,
    fit_deployment_model,
    load_raw_data,
    prepare_early_warning_frame,
)

st.set_page_config(
    page_title="Banking Crisis Early Warning",
    page_icon="🏦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_source_data() -> pd.DataFrame:
    return load_raw_data()


@st.cache_resource(show_spinner="Preparing the historical early-warning model...")
def get_model_bundle():
    raw = get_source_data()
    frame = prepare_early_warning_frame(raw)
    model, metadata = fit_deployment_model(frame)
    return model, metadata, frame


def yes_no_input(label: str, default: str = "No") -> int:
    value = st.sidebar.selectbox(
        label, ["No", "Yes"], index=0 if default == "No" else 1
    )
    return 1 if value == "Yes" else 0


st.title("🏦 Banking Crisis Early Warning and Macro-Financial Risk Monitoring")
st.write(
    "This portfolio application uses historical African macro-financial data to demonstrate "
    "a one-year-ahead banking-crisis early-warning workflow. Enter a country-year scenario "
    "to obtain a historical-pattern warning score."
)

st.warning(
    "This is an educational portfolio demonstration based on historical data. It is not a "
    "current country forecast, stress test, investment recommendation, regulatory assessment, "
    "or policy decision tool."
)

try:
    raw_data = get_source_data()
    model, metadata, early_warning_data = get_model_bundle()
except Exception as exc:
    st.error("The source dataset or model workflow could not be prepared.")
    st.code(str(exc))
    st.stop()

countries = sorted(raw_data["country"].dropna().astype(str).unique())
clean_numeric = raw_data.replace([float("inf"), float("-inf")], pd.NA)

st.sidebar.header("Current-Year Macro-Financial Scenario")
country = st.sidebar.selectbox("Country", countries)

country_rows = clean_numeric[clean_numeric["country"].astype(str) == country]
if country_rows.empty:
    country_rows = clean_numeric


def numeric_default(column: str, fallback: float = 0.0) -> float:
    series = pd.to_numeric(country_rows[column], errors="coerce").dropna()
    return float(series.median()) if not series.empty else float(fallback)


exch_usd = st.sidebar.number_input(
    "Exchange rate against USD",
    min_value=0.0,
    value=max(0.0, numeric_default("exch_usd", 1.0)),
    step=0.1,
    format="%.4f",
)
inflation_annual_cpi = st.sidebar.number_input(
    "Annual CPI inflation rate",
    value=numeric_default("inflation_annual_cpi", 5.0),
    step=1.0,
    format="%.3f",
)
gdp_weighted_default = st.sidebar.number_input(
    "Debt in default as a share of GDP",
    min_value=0.0,
    value=max(0.0, numeric_default("gdp_weighted_default", 0.0)),
    step=0.01,
    format="%.4f",
)

systemic_crisis = yes_no_input("Systemic crisis recorded this year?")
domestic_debt_in_default = yes_no_input("Domestic debt default recorded this year?")
sovereign_external_debt_default = yes_no_input("External sovereign debt default recorded this year?")
independence = yes_no_input("Country independence indicator", default="Yes")
currency_crises = yes_no_input("Currency crisis recorded this year?")
inflation_crises = yes_no_input("Inflation crisis recorded this year?")

scenario = {
    "country": country,
    "systemic_crisis": systemic_crisis,
    "exch_usd": exch_usd,
    "domestic_debt_in_default": domestic_debt_in_default,
    "sovereign_external_debt_default": sovereign_external_debt_default,
    "gdp_weighted_default": gdp_weighted_default,
    "inflation_annual_cpi": inflation_annual_cpi,
    "independence": independence,
    "currency_crises": currency_crises,
    "inflation_crises": inflation_crises,
}

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Scenario Summary")
    display_rows = {
        "Country": country,
        "Exchange rate against USD": exch_usd,
        "Annual CPI inflation": inflation_annual_cpi,
        "Debt in default / GDP": gdp_weighted_default,
        "Systemic crisis": "Yes" if systemic_crisis else "No",
        "Domestic debt default": "Yes" if domestic_debt_in_default else "No",
        "External sovereign debt default": "Yes" if sovereign_external_debt_default else "No",
        "Independence indicator": "Yes" if independence else "No",
        "Currency crisis": "Yes" if currency_crises else "No",
        "Inflation crisis": "Yes" if inflation_crises else "No",
    }
    st.dataframe(
        pd.DataFrame(list(display_rows.items()), columns=["Indicator", "Value"]),
        hide_index=True,
        use_container_width=True,
    )

with right:
    st.subheader("Historical Model Information")
    st.metric("Selected model", metadata["selected_model"])
    st.metric("Historical labelled observations", f"{metadata['n_observations']:,}")
    st.metric("One-year-ahead crisis target rate", f"{metadata['target_rate']:.1%}")

if st.button("Generate Early-Warning Score", type="primary", use_container_width=True):
    raw_input = pd.DataFrame([scenario], columns=RAW_FEATURE_COLUMNS)
    model_input = build_model_features(raw_input)
    crisis_score = float(model.predict_proba(model_input)[0, 1])
    predicted_class = int(model.predict(model_input)[0])

    st.subheader("Early-Warning Output")
    col1, col2 = st.columns(2)
    col1.metric("Model-estimated crisis score", f"{crisis_score:.1%}")
    col2.metric(
        "Threshold classification",
        "Higher warning signal" if predicted_class == 1 else "Lower warning signal",
    )

    st.progress(min(max(crisis_score, 0.0), 1.0))
    st.caption(
        "The score is produced by a historical classification model and should not be treated "
        "as a calibrated real-world probability of a future banking crisis."
    )

    if predicted_class == 1:
        st.info(
            "Under the model's default threshold, this scenario resembles historical patterns "
            "associated with a higher one-year-ahead warning signal. It warrants analytical "
            "review, not an automatic policy conclusion."
        )
    else:
        st.success(
            "Under the model's default threshold, this scenario produces a lower historical "
            "warning signal. This does not establish that a banking crisis cannot occur."
        )

with st.expander("Historical backtest and limitations"):
    metrics = metadata["holdout_metrics"]
    st.write(
        f"The model-selection workflow uses expanding-year validation on the earlier historical "
        f"period and reserves target years from {metadata['holdout_cutoff_year']} onward for a "
        "later-period holdout before the deployment specification is refitted on all labelled data."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Holdout ROC-AUC", f"{metrics['roc_auc']:.3f}")
    m2.metric("Holdout PR-AUC", f"{metrics['pr_auc']:.3f}")
    m3.metric("Holdout recall", f"{metrics['recall']:.3f}")
    m4.metric("Holdout F1", f"{metrics['f1']:.3f}")
    st.write(
        "Important limitations include the small historical sample, rare crisis events, strong "
        "cross-country heterogeneity, structural economic change across more than a century, and "
        "the absence of many variables used in modern macroprudential surveillance. The target "
        "is next-year crisis status rather than first-onset only, so some signal may reflect "
        "persistence across multi-year crisis episodes."
    )
