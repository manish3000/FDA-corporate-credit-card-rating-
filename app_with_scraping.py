import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
import joblib
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SEC Credit Rating Predictor",
    page_icon="📊",
    layout="wide"
)


st.title("🏦 SEC Credit Rating Predictor")
st.markdown("Analyze company SEC filings and predict credit ratings using AI models")

# Initialize session state
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ticker input
    ticker = st.text_input("Company Ticker", "AAPL").upper()
    
    # Filing type
    filing_type = st.selectbox(
        "Filing Type",
        ["10-K", "10-Q"],
        help="10-K: Annual report, 10-Q: Quarterly report"
    )
    
    # Years to analyze
    years = st.slider("Years to analyze", 1, 5, 3)
    
    # Model selection
    model_type = st.radio(
        "Model Type",
        ["Stacking Ensemble (Recommended)", "Random Forest", "XGBoost"]
    )
    
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        fetch_data = st.button("📥 Fetch SEC Data", type="primary", use_container_width=True)
    with col2:
        make_prediction = st.button("🔮 Predict Ratings", use_container_width=True)

# Load models
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        # Load binary classification model
        binary_model = joblib.load('best_stacking_ensemble_without_ticker.pkl')
        # Load multiclass model
        multiclass_model = joblib.load('best_multiclass_rf_without_ticker.pkl')
        return binary_model, multiclass_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Function to get CIK from ticker
def get_cik_from_ticker(ticker):
    """Convert ticker to CIK number for SEC API"""
    
    cik_mapping = {
        "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044",
        "AMZN": "0001018724", "META": "0001326801", "TSLA": "0001318605",
        "JPM": "0000019617", "WMT": "0000104169", "XOM": "0000034088",
        "JNJ": "0000200406", "V": "0001403161", "PG": "0000080424",
        "NVDA": "0001045810", "MA": "0001141391", "HD": "0000354950",
        "BAC": "0000070858", "DIS": "0001001039", "NFLX": "0001065280",
        "CSCO": "0000858877", "INTC": "0000050863", "IBM": "0000051143",
        "GS": "0000886982", "KO": "0000021344", "PEP": "0000077476",
        "MRK": "0000310158", "CVX": "0000093410", "CMCSA": "0001166691"
    }
    
    if ticker in cik_mapping:
        return cik_mapping[ticker]
    
    try:
        response = requests.get("https://www.sec.gov/files/company_tickers.json",
                               headers={'User-Agent': 'Mozilla/5.0'})
        data = response.json()
        for company in data.values():
            if company['ticker'] == ticker:
                return str(company['cik_str']).zfill(10)
    except:
        pass
    return None

# Function to fetch SEC filings
def fetch_sec_filings(cik, filing_type, years):
    """Fetch SEC filings using web scraping/API"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔍 Finding company information...")
        progress_bar.progress(10)
        
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = {
            'User-Agent': 'Your Company Name your-email@domain.com',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        }
        
        response = requests.get(submissions_url, headers=headers)
        submissions = response.json()
        
        progress_bar.progress(30)
        status_text.text("📋 Analyzing filings...")
        
        filings = submissions.get('filings', {}).get('recent', {})
        accession_numbers = filings.get('accessionNumber', [])
        filing_dates = filings.get('filingDate', [])
        forms = filings.get('form', [])
        primary_documents = filings.get('primaryDocument', [])
        
        target_filings = []
        current_year = datetime.now().year
        
        for i in range(len(forms)):
            if forms[i] == filing_type:
                filing_date = datetime.strptime(filing_dates[i], '%Y-%m-%d')
                if filing_date.year >= current_year - years:
                    target_filings.append({
                        'accession_number': accession_numbers[i],
                        'filing_date': filing_dates[i],
                        'form': forms[i],
                        'primary_doc': primary_documents[i]
                    })
        
        if not target_filings:
            st.warning(f"No {filing_type} filings found for the last {years} years")
            return None
        
        status_text.text(f"📄 Found {len(target_filings)} {filing_type} filings")
        progress_bar.progress(50)
        
        all_financial_data = []
        
        for idx, filing in enumerate(target_filings[:5]):
            status_text.text(f"📊 Processing filing {idx + 1}/{len(target_filings[:5])}...")
            
            accession = filing['accession_number'].replace('-', '')
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filing['primary_doc']}"
            filing_response = requests.get(filing_url, headers=headers)
            
            if filing_response.status_code == 200:
                soup = BeautifulSoup(filing_response.content, 'html.parser')
                financial_data = extract_financial_data_from_filing(soup, filing['filing_date'])
                
                if financial_data:
                    all_financial_data.append(financial_data)
            
            progress_bar.progress(50 + (idx + 1) * 10)
            time.sleep(0.5)  # Be respectful to SEC servers
        
        progress_bar.progress(90)
        status_text.text("📈 Calculating financial ratios...")
        
        if all_financial_data:
            df = pd.DataFrame(all_financial_data)
            df['date'] = pd.to_datetime(df['filing_date'])
            df = df.sort_values('date')
            
            # Calculate additional financial ratios
            df = calculate_financial_ratios(df)
            
            progress_bar.progress(100)
            status_text.text("✅ Data fetch complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            return df
        else:
            st.error("Could not extract financial data from filings")
            return None
            
    except Exception as e:
        st.error(f"Error fetching SEC data: {str(e)}")
        status_text.empty()
        progress_bar.empty()
        return None

def extract_financial_data_from_filing(soup, filing_date):
    """Extract financial data from SEC filing HTML"""
    
    # This is a simplified example. In production, you would:
    # 1. Parse XBRL data if available
    # 2. Look for specific financial statement tables
    # 3. Use regex patterns to find financial data
    
    financial_data = {
        'filing_date': filing_date,
        'currentRatio': np.random.uniform(1.0, 3.0),  # Simulated
        'quickRatio': np.random.uniform(0.8, 2.5),
        'cashRatio': np.random.uniform(0.2, 1.0),
        'daysOfSalesOutstanding': np.random.uniform(30, 60),
        'netProfitMargin': np.random.uniform(0.03, 0.25),
        'pretaxProfitMargin': np.random.uniform(0.05, 0.30),
        'grossProfitMargin': np.random.uniform(0.20, 0.50),
        'operatingProfitMargin': np.random.uniform(0.10, 0.35),
        'returnOnAssets': np.random.uniform(0.02, 0.15),
        'returnOnCapitalEmployed': np.random.uniform(0.04, 0.20),
        'returnOnEquity': np.random.uniform(0.05, 0.25),
        'assetTurnover': np.random.uniform(0.5, 1.5),
        'fixedAssetTurnover': np.random.uniform(0.7, 2.0),
        'debtEquityRatio': np.random.uniform(0.2, 1.5),
        'debtRatio': np.random.uniform(0.15, 0.65),
        'effectiveTaxRate': np.random.uniform(0.15, 0.35),
        'freeCashFlowOperatingCashFlowRatio': np.random.uniform(0.5, 0.95),
        'freeCashFlowPerShare': np.random.uniform(1.0, 8.0),
        'cashPerShare': np.random.uniform(0.5, 5.0),
        'companyEquityMultiplier': np.random.uniform(1.2, 4.0),
        'ebitPerRevenue': np.random.uniform(0.08, 0.30),
        'enterpriseValueMultiple': np.random.uniform(6.0, 25.0),
        'operatingCashFlowPerShare': np.random.uniform(2.0, 10.0),
        'operatingCashFlowSalesRatio': np.random.uniform(0.10, 0.40),
        'payablesTurnover': np.random.uniform(4.0, 12.0),
        'Sector': 'Technology'
    }
    
    # Try to extract real data from tables
    try:
        # Look for balance sheet tables
        tables = soup.find_all('table')
        
        for table in tables:
            text = table.get_text().lower()
            
            # Look for current assets
            if 'current assets' in text or 'total current assets' in text:
                # Parse table rows to get values
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        cell_text = cells[0].get_text().lower().strip()
                        if 'current assets' in cell_text:
                            try:
                                value = float(cells[1].get_text().replace(',', '').replace('$', '').strip())
                                # Use this value in calculations
                                pass
                            except:
                                pass
    except:
        pass  # If extraction fails, use simulated data
    
    return financial_data

def calculate_financial_ratios(df):
    """Calculate additional financial ratios"""
    
    # Calculate derived ratios
    if 'debtEquityRatio' in df.columns:
        df['financialLeverage'] = 1 + df['debtEquityRatio']
    
    if 'returnOnAssets' in df.columns and 'debtEquityRatio' in df.columns:
        df['returnOnEquity_derived'] = df['returnOnAssets'] * (1 + df['debtEquityRatio'])
    
    # Calculate trends
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['year', 'quarter', 'Sector_Encoded']:
            df[f'{col}_trend'] = df[col].pct_change().fillna(0)
    
    return df

def preprocess_for_prediction(df, models):
    """Preprocess data for model prediction"""
    
    # Get model components
    binary_package, multiclass_package = models
    
    # Prepare binary features
    binary_features = binary_package['features']
    binary_scaler = binary_package['scaler']
    
    # Prepare multiclass features
    multiclass_features = multiclass_package['feature_names']
    multiclass_scaler = multiclass_package['scaler']
    sector_encoder = multiclass_package['sector_encoder']
    
    # Process DataFrame
    processed_df = df.copy()
    
    # Encode sector
    try:
        processed_df['Sector_Encoded'] = sector_encoder.transform([processed_df['Sector'].iloc[0]] * len(processed_df))
    except:
        processed_df['Sector_Encoded'] = 0
    
    # Add date features
    if 'date' in processed_df.columns:
        processed_df['Year'] = processed_df['date'].dt.year
        processed_df['Month'] = processed_df['date'].dt.month
        processed_df['Quarter'] = processed_df['date'].dt.quarter
    
    # Ensure all required features exist
    for feature in binary_features + multiclass_features:
        if feature not in processed_df.columns:
            processed_df[feature] = 0
    
    # Scale features
    X_binary = processed_df[binary_features]
    X_multiclass = processed_df[multiclass_features]
    
    X_binary_scaled = binary_scaler.transform(X_binary)
    X_multiclass_scaled = multiclass_scaler.transform(X_multiclass)
    
    return {
        'X_binary': X_binary_scaled,
        'X_multiclass': X_multiclass_scaled,
        'processed_df': processed_df
    }

def make_predictions(preprocessed_data, models):
    """Make predictions using loaded models"""
    
    binary_package, multiclass_package = models
    
    # Get models
    binary_model = binary_package['model']
    multiclass_model = multiclass_package['model']
    
    # Make predictions
    binary_predictions = binary_model.predict(preprocessed_data['X_binary'])
    binary_probabilities = binary_model.predict_proba(preprocessed_data['X_binary'])
    
    multiclass_predictions = multiclass_model.predict(preprocessed_data['X_multiclass'])
    multiclass_probabilities = multiclass_model.predict_proba(preprocessed_data['X_multiclass'])
    
    # Create results DataFrame
    results = []
    dates = preprocessed_data['processed_df']['date'].tolist()
    
    for i in range(len(binary_predictions)):
        result = {
            'date': dates[i] if i < len(dates) else f"Period {i+1}",
            'binary_rating': 'Investment Grade' if binary_predictions[i] == 1 else 'Below Investment Grade',
            'binary_confidence': binary_probabilities[i][1] if binary_predictions[i] == 1 else binary_probabilities[i][0],
            'multiclass_rating': map_multiclass_rating(multiclass_predictions[i]),
            'multiclass_confidence': np.max(multiclass_probabilities[i]),
            'detailed_probabilities': multiclass_probabilities[i]
        }
        results.append(result)
    
    return pd.DataFrame(results)

def map_multiclass_rating(prediction):
    """Map numeric prediction to rating category"""
    rating_map = {
        0: "CCC-",
        1: "B",
        2: "BB",
        3: "BBB",
        4: "A",
        5: "AA+"
    }
    return rating_map.get(prediction, f"Class {prediction}")

# Main app logic
def main():
    # Load models
    binary_package, multiclass_package = load_models()
    
    if binary_package is None or multiclass_package is None:
        st.error("Models not loaded. Please ensure model files exist.")
        return
    
    models = (binary_package, multiclass_package)
    
    # Fetch data button
    if fetch_data and ticker:
        with st.spinner(f"Fetching SEC data for {ticker}..."):
            # Get CIK
            cik = get_cik_from_ticker(ticker)
            
            if cik is None:
                st.error(f"Could not find CIK for ticker {ticker}")
                return
            
            # Fetch filings
            financial_data = fetch_sec_filings(cik, filing_type, years)
            
            if financial_data is not None:
                st.session_state.financial_data = financial_data
                st.success(f"Successfully fetched {len(financial_data)} periods of data")
                
                # Display data
                with st.expander("📊 View Financial Data"):
                    st.dataframe(financial_data.drop(columns=['Sector'] if 'Sector' in financial_data.columns else []))
    
    # Make predictions button
    if make_prediction and st.session_state.financial_data is not None:
        with st.spinner("Making predictions..."):
            # Preprocess data
            preprocessed = preprocess_for_prediction(st.session_state.financial_data, models)
            
            # Make predictions
            predictions_df = make_predictions(preprocessed, models)
            st.session_state.predictions = predictions_df
            
            # Display predictions
            st.subheader("🎯 Prediction Results")
            
            # Latest prediction
            latest = predictions_df.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Binary Rating",
                    latest['binary_rating'],
                    f"{latest['binary_confidence']:.1%} confidence"
                )
            
            with col2:
                st.metric(
                    "Multiclass Rating",
                    latest['multiclass_rating'],
                    f"{latest['multiclass_confidence']:.1%} confidence"
                )
            
            with col3:
                # Risk assessment
                if latest['binary_rating'] == 'Investment Grade':
                    st.success("✅ Low Risk")
                else:
                    st.error("⚠️ High Risk")
            
            # Detailed predictions table
            st.dataframe(predictions_df)
            
            # Visualizations
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Binary confidence over time
                fig1 = px.line(
                    predictions_df,
                    x='date',
                    y='binary_confidence',
                    title='Investment Grade Confidence Over Time',
                    markers=True
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Multiclass confidence distribution
                fig2 = px.box(
                    predictions_df,
                    y='multiclass_confidence',
                    title='Rating Confidence Distribution'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Financial metrics vs predictions
            if st.session_state.financial_data is not None:
                st.subheader("📊 Financial Metrics Analysis")
                
                # Combine financial data with predictions
                analysis_df = st.session_state.financial_data.copy()
                analysis_df['binary_rating'] = predictions_df['binary_rating'].values
                analysis_df['multiclass_rating'] = predictions_df['multiclass_rating'].values
                
                # Select key metrics to display
                key_metrics = ['currentRatio', 'debtEquityRatio', 'returnOnEquity', 'netProfitMargin']
                
                for metric in key_metrics:
                    if metric in analysis_df.columns:
                        fig = px.line(
                            analysis_df,
                            x='date',
                            y=metric,
                            color='binary_rating',
                            title=f'{metric} by Rating Category',
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export predictions
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions (CSV)",
                    data=csv,
                    file_name=f"{ticker}_predictions.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export financial data
                if st.session_state.financial_data is not None:
                    fin_csv = st.session_state.financial_data.to_csv(index=False)
                    st.download_button(
                        label="Download Financial Data (CSV)",
                        data=fin_csv,
                        file_name=f"{ticker}_financial_data.csv",
                        mime="text/csv"
                    )
    
    # Display instructions if no data
    if st.session_state.financial_data is None:
        st.info("""
        ### 📋 How to Use This App
        
        1. **Enter a company ticker** in the sidebar (e.g., AAPL, MSFT, GOOGL)
        2. **Select filing type** (10-K for annual, 10-Q for quarterly)
        3. **Choose years to analyze** (1-5 years)
        4. **Click "Fetch SEC Data"** to download financial data
        5. **Click "Predict Ratings"** to generate credit rating predictions
        
        ### 🔍 Supported Companies
        
        This app supports major US publicly traded companies including:
        - **Technology**: AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA
        - **Finance**: JPM, BAC, GS, V, MA
        - **Consumer**: WMT, KO, PEP, PG, DIS
        - **Energy**: XOM, CVX
        - **Healthcare**: JNJ, MRK
        
        ### ⚠️ Important Notes
        
        - Data is fetched directly from SEC EDGAR database
        - Processing may take 30-60 seconds
        - Financial ratio extraction is simulated for demonstration
        - In production, you would implement full XBRL parsing
        
        ### 🎯 Model Information
        
        **Binary Classification Model:**
        - Type: Stacking Ensemble
        - Accuracy: 83.25%
        - Predicts: Investment Grade vs Below Investment Grade
        
        **Multiclass Classification Model:**
        - Type: Random Forest
        - Accuracy: 56.16%
        - Predicts: AA+, A, BBB, BB, B, or CCC-
        """)

# Run the app
if __name__ == "__main__":
    main()