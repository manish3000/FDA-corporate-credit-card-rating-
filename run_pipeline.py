"""
Simplified pipeline runner with Jupyter notebook execution
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NotebookPipeline:
    """Pipeline that executes Jupyter notebooks in sequence"""
    
    def __init__(self):
        self.notebooks = [
            {
                'name': 'Data Collection',
                'file': '001 Download from SEC HTML file.ipynb',
                'description': 'Download and parse SEC filings'
            },
            {
                'name': 'Merge Classes',
                'file': '01_1_Merge classes.ipynb',
                'description': 'Merge credit rating classes'
            },
            {
                'name': 'Corporate Credit Ratio',
                'file': '01_2 coprate credit ration.ipynb',
                'description': 'Calculate financial ratios'
            },
            {
                'name': 'Final Dataset Creation',
                'file': '03_corporate_credit_final_dataset.ipynb',
                'description': 'Create final dataset'
            },
            {
                'name': 'Tabular ML Analysis',
                'file': '0401_Only_Table_ML_and_Data_Analytics.ipynb',
                'description': 'Train models on tabular data'
            },
            {
                'name': 'Tabular + Sentiment',
                'file': '0402_Only_Table_ML_and_Data_Analytics_with_Tabular_sentiment_and_risk_scores.ipynb',
                'description': 'Add sentiment and risk scores'
            },
            {
                'name': 'Tabular + MD&A',
                'file': '0403_Only_Table_ML_and_Data_Analytics_with_Tabular_MD&A.ipynb',
                'description': 'Include MD&A text features'
            },
            {
                'name': 'Neural Network Models',
                'file': '0403_with_NN_ML_and_Data_Analytics_with_Tabular_MD&A_ipynb.ipynb',
                'description': 'Train neural network models'
            }
        ]
    
    def check_dependencies(self):
        """Check if required tools are available"""
        logger.info("Checking dependencies...")
        
        # Check for jupyter
        try:
            result = subprocess.run(['jupyter', '--version'], 
                                   capture_output=True, text=True)
            logger.info(f"✓ Jupyter found: {result.stdout.strip()}")
            return True
        except FileNotFoundError:
            logger.error("✗ Jupyter not found. Install with: pip install jupyter")
            return False
    
    def execute_notebook(self, notebook_path, timeout=3600):
        """Execute a single Jupyter notebook"""
        if not os.path.exists(notebook_path):
            logger.warning(f"✗ Notebook not found: {notebook_path}")
            return False
        
        logger.info(f"Executing: {notebook_path}")
        
        try:
            # Convert and execute notebook
            cmd = [
                'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                '--ExecutePreprocessor.timeout={}'.format(timeout),
                notebook_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully executed: {notebook_path}")
                return True
            else:
                logger.error(f"✗ Failed to execute: {notebook_path}")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"✗ Error executing {notebook_path}: {str(e)}")
            return False
    
    def run_pipeline(self, skip_on_error=True):
        """Execute all notebooks in sequence"""
        logger.info("="*80)
        logger.info("Starting Notebook Pipeline Execution")
        logger.info("="*80)
        
        if not self.check_dependencies():
            logger.error("Required dependencies not found. Aborting.")
            return
        
        results = []
        
        for i, notebook in enumerate(self.notebooks, 1):
            logger.info(f"\n[STEP {i}/{len(self.notebooks)}] {notebook['name']}")
            logger.info(f"Description: {notebook['description']}")
            logger.info(f"File: {notebook['file']}")
            
            success = self.execute_notebook(notebook['file'])
            results.append({
                'notebook': notebook['name'],
                'file': notebook['file'],
                'success': success
            })
            
            if not success and not skip_on_error:
                logger.error("Pipeline stopped due to error.")
                break
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Pipeline Execution Summary")
        logger.info("="*80)
        
        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)
        
        for result in results:
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            logger.info(f"{status}: {result['notebook']}")
        
        logger.info(f"\nCompleted: {success_count}/{total_count} notebooks")
        
        if success_count == total_count:
            logger.info("🎉 All notebooks executed successfully!")
        else:
            logger.warning(f"⚠ {total_count - success_count} notebook(s) failed")
    
    def run_streamlit_app(self):
        """Launch the Streamlit application"""
        logger.info("\n" + "="*80)
        logger.info("Launching Streamlit Application")
        logger.info("="*80)
        
        app_file = 'streamlit_app.py'
        
        if not os.path.exists(app_file):
            logger.error(f"✗ Streamlit app not found: {app_file}")
            return
        
        logger.info(f"Starting: {app_file}")
        logger.info("Press Ctrl+C to stop the app")
        
        try:
            subprocess.run(['streamlit', 'run', app_file])
        except KeyboardInterrupt:
            logger.info("\nStreamlit app stopped")
        except FileNotFoundError:
            logger.error("✗ Streamlit not found. Install with: pip install streamlit")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Corporate Credit Rating Pipeline')
    parser.add_argument('--mode', choices=['pipeline', 'app', 'both'], 
                       default='pipeline',
                       help='Run pipeline, app, or both')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue pipeline even if a notebook fails')
    
    args = parser.parse_args()
    
    pipeline = NotebookPipeline()
    
    if args.mode in ['pipeline', 'both']:
        pipeline.run_pipeline(skip_on_error=args.continue_on_error)
    
    if args.mode in ['app', 'both']:
        pipeline.run_streamlit_app()


if __name__ == '__main__':
    main()
