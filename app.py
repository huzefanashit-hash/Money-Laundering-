import streamlit as st
import pandas as pd
import numpy as np

def calculate_laundering_probability(df):
    weights = {
        'route_distance': 0.25,
        'pricing_anomaly': 0.25,
        'tax_haven': 0.15,
        'doc_discrepancy_yes': 0.15,
        'doc_discrepancy_no': 0.05,
        'company_age': 0.15,
        'owner_verification_no': 0.20
    }
    tax_havens = ['Switzerland', 'Mauritius', 'Cayman Islands']
    pricing_anomaly_threshold = 0.50

    df['ml_probability'] = 0.0
    route_ratio = df['actual_distance'] / df['shortest_distance'].replace(0, np.nan)
    df['ml_probability'] += np.clip((route_ratio - 1) * 0.1, 0, weights['route_distance'])

    if df['market_price'].iloc[0] > 0:
        price_difference = np.abs(df['unit_price'] - df['market_price']) / df['market_price'].replace(0, np.nan)
        df.loc[price_difference > pricing_anomaly_threshold, 'ml_probability'] += weights['pricing_anomaly']

    df.loc[df['origin_country'].isin(tax_havens), 'ml_probability'] += weights['tax_haven']
    df.loc[df['document_discrepancy'] == True, 'ml_probability'] += weights['doc_discrepancy_yes']
    df.loc[df['document_discrepancy'] == False, 'ml_probability'] += weights['doc_discrepancy_no']

    if 'company_age' in df.columns:
        company_age = df['company_age'].iloc[0]
        if company_age < 2:
            df.loc[:, 'ml_probability'] += weights['company_age']
        elif 2 <= company_age < 5:
            df.loc[:, 'ml_probability'] += weights['company_age'] * 0.5

    if 'owner_verification' in df.columns:
        if str(df['owner_verification'].iloc[0]).lower() == "no":
            df.loc[:, 'ml_probability'] += weights['owner_verification_no']

    df['ml_probability'] = df['ml_probability'].clip(0, 1)
    return df


st.title("💸 Money Laundering Probability Checker")

st.write("Enter transaction details below:")

actual_dist = st.number_input("Actual Transaction Distance", min_value=0.0)
shortest_dist = st.number_input("Shortest Possible Distance", min_value=0.0)
unit_price = st.number_input("Unit Price", min_value=0.0)
market_price = st.number_input("Market Price", min_value=0.0)
origin_country = st.text_input("Origin Country (e.g., Panama, UK, China)")
doc_discrepancy = st.radio("Document Discrepancy?", ["Yes", "No"])
company_age = st.number_input("Company Age (years)", min_value=0.0)
owner_verification = st.radio("Owner Verification?", ["Yes", "No"])

if st.button("Analyze Transaction"):
    new_df = pd.DataFrame({
        "transaction_id": ["NEW_TXN"],
        "actual_distance": [actual_dist],
        "shortest_distance": [shortest_dist],
        "unit_price": [unit_price],
        "market_price": [market_price],
        "origin_country": [origin_country],
        "document_discrepancy": [doc_discrepancy == "Yes"],
        "company_age": [company_age],
        "owner_verification": [owner_verification],
    })

    result_df = calculate_laundering_probability(new_df)
    probability = result_df.loc[0, "ml_probability"]

    st.subheader("📊 Analysis Result")
    st.write(f"**Money Laundering Probability:** {probability:.2f} ({probability:.0%})")

    if probability >= 0.5:
        st.error("⚠️ HIGH RISK Transaction")
    elif probability >= 0.25:
        st.warning("⚠️ MODERATE RISK Transaction - Needs Review")
    else:
        st.success("✅ LOW RISK Transaction")
