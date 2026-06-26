# Nepal Used Car & Mobile Market Analysis

## Overview
This project explores and analyzes the Nepalese used car and mobile phone markets using real-world data scraped from Hamrobazaar. It includes comprehensive data cleaning, exploratory data analysis (EDA), and a Streamlit app for phone deal discovery.

---

## 1. Used Car Market EDA
- **Dataset**: 847 cleaned car listings (1975–2023)
- **Key Insights**:
  - Engine size is the strongest price driver (stronger than mileage)
  - Mileage penalty is moderate; buyers accept higher odometer if price reflects it
  - Toyota commands the highest brand premium
  - Hybrids/EVs command huge premiums despite low volume
  - Diesel vehicles hold value well
  - Sweet spot: 10–18L petrol cars under 100k km (average ~12 Lakh)
- **Analysis**: See `CarDataset-EDA/` notebooks and summary docs for detailed findings.

---

## 2. Mobile Phone Data Scraping & Cleaning
- **Source**: Hamrobazaar mobile listings
- **Scripts/Notebooks**: See `ScrapedDataset-EDA/hamrobazaar_scraper.py` and `DataCleaning.ipynb`
- **Output**: Cleaned dataset (`hamrobazaar_mobiles_cleaned.csv`) with enhanced features for modeling

---

## 3. Streamlit App: Nepal Phone Deal Finder
- **App**: `ScrapedDataset-EDA/app.py`
- **Features**:
  - Loads cleaned mobile data and predicts fair prices using a trained ML model
  - Flags potential scam listings
  - Highlights best deals, premium deals, and top picks by brand
  - Interactive dashboard for users to explore deals

---

## Project Structure
See `PROJECT_STRUCTURE.md` for a full directory outline.

---

## Requirements
- Python 3.8+
- See `ScrapedDataset-EDA/requirements_scraper.txt` for dependencies

---

## Quick Start
1. Install requirements: `pip install -r ScrapedDataset-EDA/requirements_scraper.txt`
2. Run the Streamlit app: `streamlit run ScrapedDataset-EDA/app.py`
