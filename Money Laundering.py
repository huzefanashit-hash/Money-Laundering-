import pandas as pd
import numpy as np

def calculate_laundering_probability(df):
    """
    Calculates the probability of money laundering for each transaction in a DataFrame
    based on a set of predefined patterns and weights.
    """

    # --- 1. Define Risk Factors and Weights ---
    weights = {
        'route_distance': 0.25,
        'pricing_anomaly': 0.25,
        'tax_haven': 0.15,
        'doc_discrepancy_yes': 0.15,
        'doc_discrepancy_no': 0.05,
        'company_age': 0.15,
        'owner_verification_no': 0.20
    }

    # --- 2. Define Thresholds and Lists ---
    tax_havens = ['Switzerland', 'Mauritius', 'Cayman Islands']
    pricing_anomaly_threshold = 0.50  # 50% deviation

    # --- 3. Initialize ---
    df['ml_probability'] = 0.0

    # --- 4. Apply Rules ---
    # Route distance risk (scaled)
    route_ratio = df['actual_distance'] / df['shortest_distance'].replace(0, np.nan)
    df['ml_probability'] += np.clip((route_ratio - 1) * 0.1, 0, weights['route_distance'])

    # Pricing anomaly
    if df['market_price'].iloc[0] > 0:
        price_difference = np.abs(df['unit_price'] - df['market_price']) / df['market_price'].replace(0, np.nan)
        df.loc[price_difference > pricing_anomaly_threshold, 'ml_probability'] += weights['pricing_anomaly']

    # Tax haven check
    df.loc[df['origin_country'].isin(tax_havens), 'ml_probability'] += weights['tax_haven']

    # Document discrepancy risk
    df.loc[df['document_discrepancy'] == True, 'ml_probability'] += weights['doc_discrepancy_yes']
    df.loc[df['document_discrepancy'] == False, 'ml_probability'] += weights['doc_discrepancy_no']

    # Company age → younger = more risk, older = less
    if 'company_age' in df.columns:
        company_age = df['company_age'].iloc[0]
        if company_age < 2:
            df.loc[:, 'ml_probability'] += weights['company_age']
        elif 2 <= company_age < 5:
            df.loc[:, 'ml_probability'] += weights['company_age'] * 0.5
        else:
            df.loc[:, 'ml_probability'] += 0.0  # safe if >5 years

    # Owner verification risk
    if 'owner_verification' in df.columns:
        if str(df['owner_verification'].iloc[0]).lower() == "no":
            df.loc[:, 'ml_probability'] += weights['owner_verification_no']

    # Cap probability at 1
    df['ml_probability'] = df['ml_probability'].clip(0, 1)

    return df


def analyze_new_transaction():
    """
    Interactive prompt to analyze one transaction.
    """
    print("\n--- Analyze a New Transaction ---")
    try:
        # --- Collect User Input ---
        actual_dist = float(input("Enter Actual Transaction Distance (e.g., 8000): "))
        shortest_dist = float(input("Enter Shortest Possible Transaction Distance (e.g., 2000): "))

        # Export to India condition
        exporting_india = input("Is this transaction done through exporting to India? (yes/no): ").lower()
        product_sold = None
        unit_price = 0.0
        market_price = 0.0

        if exporting_india in ['yes', 'y']:
            product_sold = input("What product is being sold or exporting to India?: ")
            unit_price = float(input("Enter Unit Price (e.g., 1500): "))
            market_price = float(input("Enter Market Price (e.g., 950): "))

        origin_country = input("Enter Origin Country (e.g., Panama, UK, China): ")
        doc_discrepancy_input = input("Are there document discrepancies? (yes/no): ").lower()
        doc_discrepancy = doc_discrepancy_input in ['yes', 'y']

        shell_fdi = input("Is this transferring to a shell company in the name of FDI? (yes/no): ").lower()
        company_age = None
        owner_verification = None
        if shell_fdi in ['yes', 'y']:
            owner_verification = input("Enter Owner Verification details (type 'no' if not verified): ")
            company_age = float(input("Enter the age of the company in years (e.g., 1.5): "))

        # --- Build Data ---
        new_data = {
            'transaction_id': ['NEW_TXN'],
            'actual_distance': [actual_dist],
            'shortest_distance': [shortest_dist],
            'unit_price': [unit_price],
            'market_price': [market_price],
            'origin_country': [origin_country],
            'document_discrepancy': [doc_discrepancy],
        }

        if company_age is not None:
            new_data['company_age'] = [company_age]
        if owner_verification is not None:
            new_data['owner_verification'] = [owner_verification]

        new_df = pd.DataFrame(new_data)

        # --- Analyze ---
        result_df = calculate_laundering_probability(new_df)
        probability = result_df.loc[0, 'ml_probability']

        # --- Display ---
        print("\n--- Analysis Result ---")
        print(f"The calculated Money Laundering Probability is: {probability:.2f} ({probability:.0%})")
        if probability >= 0.5:
            print("⚠️ This transaction is considered HIGH RISK.")
        elif probability >= 0.25:
            print("⚠️ This transaction requires further review (MODERATE RISK).")
        else:
            print("✅ This transaction is considered LOW RISK.")
        print("-----------------------")

    except ValueError:
        print("\n[Error] Invalid input. Please enter valid numbers for distance, price, and age fields.")
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")


# --- Main Loop ---
if __name__ == "__main__":
    while True:
        analyze_new_transaction()
        another = input("\nDo you want to analyze another transaction? (yes/no): ").lower()
        if another not in ['yes', 'y']:
            print("Exiting program.")
            break
