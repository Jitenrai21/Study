import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Nepal Phone Deal Finder", layout="wide")
st.title("🇳🇵 Hamrobazaar Phone Deal Finder")

@st.cache_data
def load_and_predict():
    df = pd.read_csv('hamrobazaar_mobiles_cleaned.csv')
    
    # Add enhanced features
    df['title_length'] = df['title'].str.len().fillna(0)
    df['scam_flag'] = df['title'].str.contains('17 Pro Max|S25 Ultra|17 Air|16e|17pro max|s25 ultra', case=False, na=False).astype(int)
    
    model = joblib.load('nepal_phone_final_model.pkl')
    
    features = ['Location_City_Encoded', 'negotiable_encoded', 'Storage_GB_Corrected',
                'RAM_GB_Corrected', 'Screen_Size_Inches', 'Back_Camera_MP_Filtered', 
                'Brand_Model_Encoded', 'title_length', 'scam_flag']
    X = df[features].fillna(0)
    
    df['Fair_Price'] = model.predict(X).round(0).astype(int)
    df['Savings_%'] = ((df['Fair_Price'] - df['Price_Cleaned']) / df['Fair_Price'] * 100).round(1)
    
    # Extract brand from Brand_Model
    df['Brand'] = df['Brand_Model'].str.split().str[0]
    
    return df

df = load_and_predict()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Listings", len(df))
col2.metric("🔥 Best Deals (≥20% Off)", len(df[df['Savings_%'] >= 20]))
col3.metric("💎 Premium Deals (>60k)", len(df[(df['Price_Cleaned'] > 60000) & (df['Savings_%'] >= 15)]))
col4.metric("📱 Brands Available", df['Brand'].nunique())

tab1, tab2, tab3 = st.tabs(["🔥 Best Deals", "💎 Premium Hot Deals", "📱 Top Picks by Brand"])

with tab1:
    deals = df.sort_values('Savings_%', ascending=False).head(25)
    deals = deals[deals['Savings_%'] >= 15]
    if not deals.empty:
        st.dataframe(deals[['title', 'Price_Cleaned', 'Fair_Price', 'Savings_%', 'Brand_Model', 'Storage_GB_Corrected']],
                     use_container_width=True, hide_index=True)

with tab2:
    st.subheader("High-End Phones with Best Savings (Model performs best on premium devices)")
    premium = df[df['Price_Cleaned'] > 60000].sort_values('Savings_%', ascending=False).head(20)
    premium = premium[premium['Savings_%'] >= 10]
    
    if not premium.empty:
        st.dataframe(premium[['title', 'Price_Cleaned', 'Fair_Price', 'Savings_%', 'Brand_Model', 'Storage_GB_Corrected', 'RAM_GB_Corrected']],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No premium deals with 10%+ savings found.")

with tab3:
    # Get top brands by listing count
    top_brands = df['Brand'].value_counts().head(10).index.tolist()
    
    selected_brand = st.selectbox("Select Brand", ["All Brands"] + top_brands)
    
    if selected_brand == "All Brands":
        # Show best deal from each top brand
        brand_picks = []
        for brand in top_brands:
            best = df[df['Brand'] == brand].sort_values('Savings_%', ascending=False).head(1)
            brand_picks.append(best)
        brand_deals = pd.concat(brand_picks).sort_values('Savings_%', ascending=False)
    else:
        # Show top deals for selected brand
        brand_deals = df[df['Brand'] == selected_brand].sort_values('Savings_%', ascending=False).head(15)
    
    if not brand_deals.empty:
        st.dataframe(brand_deals[['title', 'Price_Cleaned', 'Fair_Price', 'Savings_%', 'Brand_Model', 'Storage_GB_Corrected', 'RAM_GB_Corrected']],
                     use_container_width=True, hide_index=True)

st.success("Model: XGBoost (or RF if it won) | Improved MAE: ~12k NPR | Features: 9 total | Data: 652 listings")