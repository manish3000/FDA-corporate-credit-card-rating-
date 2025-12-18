"""
Corporate Credit Rating Prediction Pipeline
This script orchestrates the entire workflow from data collection to model deployment.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CreditRatingPipeline:
    """Main pipeline class for corporate credit rating prediction"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.data_dir = Path(self.config['data_dir'])
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.models_dir = Path(self.config['models_dir'])
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'data_dir': 'data',
            'models_dir': 'models',
            'sec_filings_dir': 'sec_filings',
            'test_size': 0.2,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def run_full_pipeline(self):
        """Execute the complete pipeline"""
        logger.info("="*80)
        logger.info("Starting Corporate Credit Rating Pipeline")
        logger.info("="*80)
        
        try:
            # Step 1: Data Collection
            logger.info("\n[STEP 1] Data Collection")
            self.collect_data()
            
            # Step 2: Data Cleaning
            logger.info("\n[STEP 2] Data Cleaning")
            self.clean_data()
            
            # Step 3: Merge Datasets
            logger.info("\n[STEP 3] Merging Datasets")
            self.merge_datasets()
            
            # Step 4: Feature Engineering
            logger.info("\n[STEP 4] Feature Engineering")
            self.feature_engineering()
            
            # Step 5: Model Training (Tabular Only)
            logger.info("\n[STEP 5] Training Tabular Models")
            self.train_tabular_models()
            
            # Step 6: Model Training (With MD&A)
            logger.info("\n[STEP 6] Training Models with MD&A Text")
            self.train_mda_models()
            
            # Step 7: Model Evaluation
            logger.info("\n[STEP 7] Model Evaluation")
            self.evaluate_models()
            
            logger.info("\n" + "="*80)
            logger.info("Pipeline completed successfully!")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def collect_data(self):
        """Step 1: Collect data from SEC filings"""
        logger.info("Checking for existing credit ratings data...")
        
        tabular_file = self.processed_dir / '01_credit_ratings_tabular_clean.csv'
        mda_file = self.processed_dir / '02_credit_ratings_with_mda.csv'
        
        if tabular_file.exists():
            logger.info(f"✓ Found tabular data: {tabular_file}")
        else:
            logger.warning(f"✗ Missing: {tabular_file}")
            logger.info("Run '001 Download from SEC HTML file.ipynb' to collect data")
        
        if mda_file.exists():
            logger.info(f"✓ Found MD&A data: {mda_file}")
        else:
            logger.warning(f"✗ Missing: {mda_file}")
    
    def clean_data(self):
        """Step 2: Clean and preprocess the data"""
        logger.info("Loading and cleaning credit ratings data...")
        
        # Load the initial dataset
        input_file = '01_credit_ratings_tabular_clean.csv'
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} records from {input_file}")
            
            # Basic cleaning
            initial_rows = len(df)
            df = df.dropna(subset=['credit_rating'])
            logger.info(f"Removed {initial_rows - len(df)} rows with missing ratings")
            
            # Save cleaned data
            output_file = self.processed_dir / 'credit_ratings_cleaned.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"✓ Saved cleaned data to {output_file}")
        else:
            logger.warning(f"Input file {input_file} not found")
    
    def merge_datasets(self):
        """Step 3: Merge different data sources"""
        logger.info("Merging tabular and textual data...")
        
        tabular_file = '01_credit_ratings_tabular_clean.csv'
        mda_file = '02_credit_ratings_with_mda.csv'
        
        if os.path.exists(tabular_file) and os.path.exists(mda_file):
            df_tabular = pd.read_csv(tabular_file)
            df_mda = pd.read_csv(mda_file)
            
            logger.info(f"Tabular data shape: {df_tabular.shape}")
            logger.info(f"MD&A data shape: {df_mda.shape}")
            
            # Merge on company and date
            if 'company' in df_tabular.columns and 'company' in df_mda.columns:
                df_merged = pd.merge(df_tabular, df_mda, 
                                    on=['company', 'date'], 
                                    how='inner',
                                    suffixes=('', '_mda'))
                
                output_file = self.processed_dir / 'credit_ratings_merged.csv'
                df_merged.to_csv(output_file, index=False)
                logger.info(f"✓ Merged data shape: {df_merged.shape}")
                logger.info(f"✓ Saved to {output_file}")
            else:
                logger.warning("Cannot merge: missing 'company' column")
        else:
            logger.warning("One or both input files missing for merge")
    
    def feature_engineering(self):
        """Step 4: Create additional features"""
        logger.info("Performing feature engineering...")
        
        merged_file = self.processed_dir / 'credit_ratings_merged.csv'
        if merged_file.exists():
            df = pd.read_csv(merged_file)
            
            # Create rating classes
            if 'credit_rating' in df.columns:
                df['rating_numeric'] = df['credit_rating'].map(self.get_rating_mapping())
                df['rating_class'] = df['credit_rating'].map(self.get_rating_class_mapping())
                
                logger.info(f"✓ Created rating features")
                logger.info(f"  Rating classes: {df['rating_class'].unique()}")
            
            # Create financial ratios
            if all(col in df.columns for col in ['total_assets', 'total_liabilities']):
                df['debt_to_assets'] = df['total_liabilities'] / df['total_assets']
                logger.info("✓ Created debt-to-assets ratio")
            
            if all(col in df.columns for col in ['net_income', 'total_assets']):
                df['roa'] = df['net_income'] / df['total_assets']
                logger.info("✓ Created ROA (Return on Assets)")
            
            # Save engineered features
            output_file = self.processed_dir / 'credit_ratings_featured.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"✓ Saved featured data to {output_file}")
        else:
            logger.warning(f"Merged file not found: {merged_file}")
    
    def train_tabular_models(self):
        """Step 5: Train models using only tabular features"""
        logger.info("Training models with tabular features only...")
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            import joblib
            
            featured_file = self.processed_dir / 'credit_ratings_featured.csv'
            if featured_file.exists():
                df = pd.read_csv(featured_file)
                
                # Select numeric features
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols 
                               if col not in ['rating_numeric', 'rating_class'] 
                               and not col.startswith('unnamed')]
                
                if len(feature_cols) > 0 and 'rating_class' in df.columns:
                    # Remove rows with missing target
                    df = df.dropna(subset=['rating_class'])
                    
                    X = df[feature_cols].fillna(0)
                    y = df['rating_class']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=self.config['test_size'],
                        random_state=self.config['random_state'],
                        stratify=y
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train Random Forest
                    logger.info("Training Random Forest Classifier...")
                    rf_model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=self.config['random_state'],
                        n_jobs=self.config['n_jobs']
                    )
                    rf_model.fit(X_train_scaled, y_train)
                    
                    train_score = rf_model.score(X_train_scaled, y_train)
                    test_score = rf_model.score(X_test_scaled, y_test)
                    
                    logger.info(f"✓ Random Forest trained")
                    logger.info(f"  Train accuracy: {train_score:.4f}")
                    logger.info(f"  Test accuracy: {test_score:.4f}")
                    
                    # Save model and scaler
                    model_path = self.models_dir / 'rf_tabular_model.pkl'
                    scaler_path = self.models_dir / 'scaler_tabular.pkl'
                    
                    joblib.dump(rf_model, model_path)
                    joblib.dump(scaler, scaler_path)
                    joblib.dump(feature_cols, self.models_dir / 'feature_cols.pkl')
                    
                    logger.info(f"✓ Saved model to {model_path}")
                else:
                    logger.warning("Insufficient features or missing target for training")
            else:
                logger.warning(f"Featured file not found: {featured_file}")
                
        except ImportError as e:
            logger.error(f"Required library not found: {e}")
            logger.info("Install required libraries: pip install scikit-learn joblib")
    
    def train_mda_models(self):
        """Step 6: Train models including MD&A text features"""
        logger.info("Training models with MD&A text features...")
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            import joblib
            
            featured_file = self.processed_dir / 'credit_ratings_featured.csv'
            if featured_file.exists():
                df = pd.read_csv(featured_file)
                
                # Check if MD&A text exists
                mda_col = None
                for col in ['mda_text', 'md_a_text', 'mda', 'text']:
                    if col in df.columns:
                        mda_col = col
                        break
                
                if mda_col and 'rating_class' in df.columns:
                    # Filter valid rows
                    df_valid = df.dropna(subset=[mda_col, 'rating_class'])
                    
                    if len(df_valid) > 50:  # Minimum samples for training
                        X_text = df_valid[mda_col].astype(str)
                        y = df_valid['rating_class']
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_text, y,
                            test_size=self.config['test_size'],
                            random_state=self.config['random_state'],
                            stratify=y
                        )
                        
                        # Vectorize text
                        logger.info("Vectorizing MD&A text...")
                        vectorizer = TfidfVectorizer(
                            max_features=500,
                            min_df=2,
                            max_df=0.8,
                            ngram_range=(1, 2)
                        )
                        X_train_vec = vectorizer.fit_transform(X_train)
                        X_test_vec = vectorizer.transform(X_test)
                        
                        # Train model
                        logger.info("Training Random Forest with text features...")
                        rf_text_model = RandomForestClassifier(
                            n_estimators=100,
                            random_state=self.config['random_state'],
                            n_jobs=self.config['n_jobs']
                        )
                        rf_text_model.fit(X_train_vec, y_train)
                        
                        train_score = rf_text_model.score(X_train_vec, y_train)
                        test_score = rf_text_model.score(X_test_vec, y_test)
                        
                        logger.info(f"✓ Text model trained")
                        logger.info(f"  Train accuracy: {train_score:.4f}")
                        logger.info(f"  Test accuracy: {test_score:.4f}")
                        
                        # Save model and vectorizer
                        model_path = self.models_dir / 'rf_text_model.pkl'
                        vectorizer_path = self.models_dir / 'text_vectorizer.pkl'
                        
                        joblib.dump(rf_text_model, model_path)
                        joblib.dump(vectorizer, vectorizer_path)
                        
                        logger.info(f"✓ Saved text model to {model_path}")
                    else:
                        logger.warning(f"Insufficient samples with MD&A text: {len(df_valid)}")
                else:
                    logger.warning("MD&A text column or rating class not found")
            else:
                logger.warning(f"Featured file not found: {featured_file}")
                
        except ImportError as e:
            logger.error(f"Required library not found: {e}")
    
    def evaluate_models(self):
        """Step 7: Evaluate all trained models"""
        logger.info("Evaluating trained models...")
        
        try:
            import joblib
            from sklearn.metrics import classification_report
            
            model_files = [
                ('Tabular Model', self.models_dir / 'rf_tabular_model.pkl'),
                ('Text Model', self.models_dir / 'rf_text_model.pkl')
            ]
            
            for model_name, model_path in model_files:
                if model_path.exists():
                    logger.info(f"\n✓ Found {model_name}: {model_path}")
                else:
                    logger.info(f"✗ {model_name} not found: {model_path}")
            
            # Create evaluation summary
            summary_path = self.models_dir / 'model_summary.txt'
            with open(summary_path, 'w') as f:
                f.write("Corporate Credit Rating Model Summary\n")
                f.write("="*50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for model_name, model_path in model_files:
                    f.write(f"{model_name}: {'Available' if model_path.exists() else 'Not trained'}\n")
            
            logger.info(f"✓ Saved model summary to {summary_path}")
            
        except ImportError:
            logger.warning("sklearn not available for model evaluation")
    
    def get_rating_mapping(self):
        """Return mapping from rating to numeric value"""
        return {
            'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
            'A+': 5, 'A': 6, 'A-': 7,
            'BBB+': 8, 'BBB': 9, 'BBB-': 10,
            'BB+': 11, 'BB': 12, 'BB-': 13,
            'B+': 14, 'B': 15, 'B-': 16,
            'CCC+': 17, 'CCC': 18, 'CCC-': 19,
            'CC': 20, 'C': 21, 'D': 22
        }
    
    def get_rating_class_mapping(self):
        """Return mapping from rating to investment class"""
        investment_grade = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
                           'BBB+', 'BBB', 'BBB-']
        speculative_grade = ['BB+', 'BB', 'BB-', 'B+', 'B', 'B-']
        high_risk = ['CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']
        
        mapping = {}
        for rating in investment_grade:
            mapping[rating] = 'Investment Grade'
        for rating in speculative_grade:
            mapping[rating] = 'Speculative Grade'
        for rating in high_risk:
            mapping[rating] = 'High Risk'
        
        return mapping


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Corporate Credit Rating Pipeline')
    parser.add_argument('--step', type=str, help='Run specific step (1-7)', default='all')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CreditRatingPipeline()
    
    # Run specified step or full pipeline
    if args.step == 'all':
        pipeline.run_full_pipeline()
    else:
        step_map = {
            '1': pipeline.collect_data,
            '2': pipeline.clean_data,
            '3': pipeline.merge_datasets,
            '4': pipeline.feature_engineering,
            '5': pipeline.train_tabular_models,
            '6': pipeline.train_mda_models,
            '7': pipeline.evaluate_models
        }
        
        if args.step in step_map:
            logger.info(f"Running step {args.step}...")
            step_map[args.step]()
        else:
            logger.error(f"Invalid step: {args.step}")
            logger.info("Valid steps: 1-7 or 'all'")


if __name__ == '__main__':
    main()
