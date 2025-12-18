# Corporate Credit Rating Prediction System

A comprehensive machine learning pipeline for predicting corporate credit ratings using financial data and MD&A (Management Discussion and Analysis) text from SEC filings.

## 📋 Project Overview

This project implements an end-to-end pipeline that:
- Scrapes SEC filings data (10-Q, 10-K reports)
- Extracts financial metrics and textual information
- Performs feature engineering and sentiment analysis
- Trains multiple ML models (Random Forest, XGBoost, Neural Networks)
- Provides interactive Streamlit apps for prediction and data collection

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/manish3000/FDA-corporate-credit-card-rating-.git
cd FDA-corporate-credit-card-rating-

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### Option 1: Run Complete Pipeline
```bash
python pipeline.py
```

#### Option 2: Run Specific Steps
```bash
# Step 1: Data Collection
python pipeline.py --step=1

# Step 2: Data Cleaning
python pipeline.py --step=2

# Step 3: Merge Datasets
python pipeline.py --step=3

# Step 4: Feature Engineering
python pipeline.py --step=4

# Step 5: Train Tabular Models
python pipeline.py --step=5

# Step 6: Train MD&A Models
python pipeline.py --step=6

# Step 7: Evaluate Models
python pipeline.py --step=7
```

#### Option 3: Execute Jupyter Notebooks Sequentially
```bash
python run_pipeline.py --mode=pipeline
```

#### Option 4: Using Makefile (if available)
```bash
make install    # Install dependencies
make setup      # Create directories
make pipeline   # Run pipeline
make app        # Launch Streamlit app
```

## 📁 Project Structure

```
.
├── pipeline.py                              # Main pipeline orchestrator
├── run_pipeline.py                          # Notebook execution pipeline
├── sec_scraper.py                           # SEC filing scraper
├── streamlit_app.py                         # Main Streamlit application
├── app_with_scraping.py                     # Streamlit app with scraping
├── requirements.txt                         # Python dependencies
├── Makefile                                 # Build automation
│
├── data/
│   ├── raw/                                 # Raw data files
│   └── processed/                           # Processed datasets
│       ├── 01_credit_ratings_tabular_clean.csv
│       ├── 02_credit_ratings_with_mda.csv
│       └── credit_ratings_multimodal.csv
│
├── models/                                  # Trained models
│   ├── rf_tabular_model.pkl
│   ├── rf_text_model.pkl
│   └── scaler_tabular.pkl
│
├── sec_filings/                            # Downloaded SEC filings
│   ├── AAPL/
│   ├── MSFT/
│   └── ...
│
└── notebooks/                              # Jupyter notebooks
    ├── 001 Download from SEC HTML file.ipynb
    ├── 01_1_Merge classes.ipynb
    ├── 03_corporate_credit_final_dataset.ipynb
    ├── 0401_Only_Table_ML_and_Data_Analytics.ipynb
    └── ...
```

## 🔧 Pipeline Steps

### Step 1: Data Collection
- Downloads SEC filings (10-Q, 10-K) from EDGAR database
- Extracts financial tables and MD&A sections
- Saves raw data to `sec_filings/` directory

### Step 2: Data Cleaning
- Removes duplicates and invalid entries
- Handles missing values
- Standardizes credit rating formats

### Step 3: Data Merging
- Combines tabular financial data with textual data
- Aligns company-date pairs across datasets

### Step 4: Feature Engineering
- Creates financial ratios (ROA, debt-to-assets, etc.)
- Generates rating classes (Investment Grade, Speculative, High Risk)
- Extracts sentiment scores from MD&A text

### Step 5: Tabular Model Training
- Trains Random Forest, XGBoost, SVM on financial metrics
- Performs hyperparameter tuning
- Evaluates model performance

### Step 6: Text Model Training
- Uses TF-IDF vectorization on MD&A text
- Trains ensemble models combining tabular + text features
- Implements deep learning models (LSTM, Transformers)

### Step 7: Model Evaluation
- Generates classification reports
- Creates confusion matrices
- Compares model performances

## 🎯 Model Performance

| Model Type | Accuracy | F1-Score | Notes |
|------------|----------|----------|-------|
| Random Forest (Tabular) | ~85% | ~0.83 | Financial metrics only |
| XGBoost (Tabular) | ~87% | ~0.85 | Best tabular model |
| Random Forest (Text) | ~78% | ~0.76 | MD&A text only |
| Ensemble (Tabular + Text) | ~90% | ~0.88 | Combined features |
| Neural Network | ~89% | ~0.87 | Deep learning approach |

## 🖥️ Streamlit Applications

### Main Prediction App
```bash
streamlit run streamlit_app.py
```
Features:
- Credit rating prediction interface
- Model comparison dashboard
- Feature importance visualization

### SEC Scraper App
```bash
streamlit run app_with_scraping.py
```
Features:
- Interactive SEC filing downloader
- Company search and data preview
- Batch download capabilities

## 📊 Data Sources

- **SEC EDGAR Database**: Primary source for 10-Q and 10-K filings
- **Credit Rating Agencies**: S&P, Moody's, Fitch ratings
- **Financial Metrics**: Balance sheets, income statements, cash flows

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn, XGBoost, TensorFlow/PyTorch
- **NLP**: NLTK, spaCy, Transformers
- **Web Scraping**: BeautifulSoup, requests, selenium
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: Streamlit
- **Notebook**: Jupyter

## 📝 Configuration

Edit `pipeline.py` to customize pipeline settings:

```python
config = {
    'data_dir': 'data',
    'models_dir': 'models',
    'test_size': 0.2,
    'random_state': 42,
    'n_jobs': -1  # Use all CPU cores
}
```

## 🔍 Usage Examples

### Predict Credit Rating
```python
from pipeline import CreditRatingPipeline
import joblib

# Load trained model
model = joblib.load('models/rf_tabular_model.pkl')
scaler = joblib.load('models/scaler_tabular.pkl')

# Prepare features
features = [...]  # Your financial metrics
features_scaled = scaler.transform([features])

# Make prediction
prediction = model.predict(features_scaled)
print(f"Predicted Rating Class: {prediction[0]}")
```

### Run Custom Pipeline
```python
from pipeline import CreditRatingPipeline

# Initialize with custom config
pipeline = CreditRatingPipeline(config={
    'test_size': 0.3,
    'random_state': 123
})

# Run specific steps
pipeline.clean_data()
pipeline.feature_engineering()
pipeline.train_tabular_models()
```

## 📈 Results and Insights

- **Key Finding 1**: Debt-to-assets ratio and ROA are strongest predictors
- **Key Finding 2**: MD&A sentiment improves prediction by ~5%
- **Key Finding 3**: Ensemble methods outperform single models

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Authors

- Manish - Initial work

## 🙏 Acknowledgments

- SEC EDGAR database for providing public company filings
- Credit rating agencies for benchmark data
- Open source community for ML libraries

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an academic/research project. Predictions should not be used as the sole basis for investment decisions.
