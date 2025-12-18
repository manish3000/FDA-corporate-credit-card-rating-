# sec_scraper.py - Advanced SEC data scraping module
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import re
import json
import time
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

class SECScraper:
    """Advanced SEC data scraper with XBRL parsing"""
    
    def __init__(self, user_agent="Company Analytics contact@company.com"):
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate'
        }
        self.base_url = "https://www.sec.gov"
        
    def get_cik_from_ticker(self, ticker):
        """Get CIK from ticker symbol"""
        try:
            # Try local mapping first
            cik_map = self._load_cik_mapping()
            if ticker in cik_map:
                return cik_map[ticker]
            
            # Try SEC API
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            data = response.json()
            
            for company in data.values():
                if company['ticker'] == ticker:
                    return str(company['cik_str']).zfill(10)
                    
        except Exception as e:
            print(f"Error getting CIK: {e}")
        
        return None
    
    def _load_cik_mapping(self):
        """Load local CIK mapping"""
        return {
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
    
    def get_company_filings(self, cik, filing_type="10-K", years=3):
        """Get company filings from SEC"""
        
        # Get submissions
        submissions_url = f"{self.base_url}/submissions/CIK{cik}.json"
        response = requests.get(submissions_url, headers=self.headers)
        
        if response.status_code != 200:
            return []
        
        submissions = response.json()
        filings = submissions.get('filings', {}).get('recent', {})
        
        # Extract filing information
        forms = filings.get('form', [])
        accession_numbers = filings.get('accessionNumber', [])
        filing_dates = filings.get('filingDate', [])
        primary_docs = filings.get('primaryDocument', [])
        
        # Filter by type and date
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
                        'primary_document': primary_docs[i]
                    })
        
        return target_filings[:5]  # Limit to 5 most recent
    
    def get_filing_data(self, cik, accession_number, primary_document):
        """Get filing data and extract financial information"""
        
        # Construct filing URL
        accession_clean = accession_number.replace('-', '')
        filing_url = f"{self.base_url}/Archives/edgar/data/{cik}/{accession_clean}/{primary_document}"
        
        response = requests.get(filing_url, headers=self.headers)
        
        if response.status_code != 200:
            return None
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find XBRL data
        xbrl_data = self._extract_xbrl_data(soup)
        
        if xbrl_data:
            return self._parse_xbrl_financials(xbrl_data)
        else:
            # Fall back to HTML parsing
            return self._parse_html_financials(soup)
    
    def _extract_xbrl_data(self, soup):
        """Extract XBRL data from filing"""
        
        # Look for XBRL instance documents
        xbrl_links = soup.find_all('a', href=re.compile(r'.*\.xml$'))
        
        for link in xbrl_links:
            if 'ixbrl' in link.get('href', '').lower() or 'xbrl' in link.get('href', '').lower():
                xbrl_url = link['href']
                if not xbrl_url.startswith('http'):
                    xbrl_url = self.base_url + xbrl_url
                
                try:
                    response = requests.get(xbrl_url, headers=self.headers)
                    if response.status_code == 200:
                        return response.content
                except:
                    continue
        
        return None
    
    def _parse_xbrl_financials(self, xbrl_content):
        """Parse XBRL financial data"""
        
        try:
            # Parse XML
            root = ET.fromstring(xbrl_content)
            
            # Namespace handling
            ns = {
                'xbrli': 'http://www.xbrl.org/2003/instance',
                'us-gaap': 'http://fasb.org/us-gaap/2021-01-31'
            }
            
            financials = {}
            
            # Extract key financial metrics
            metrics = {
                'Assets': 'Assets',
                'Liabilities': 'Liabilities',
                'StockholdersEquity': 'StockholdersEquity',
                'RevenueFromContractWithCustomerExcludingAssessedTax': 'Revenue',
                'NetIncomeLoss': 'NetIncome',
                'OperatingIncomeLoss': 'OperatingIncome',
                'CashAndCashEquivalentsAtCarryingValue': 'Cash',
                'CurrentAssets': 'CurrentAssets',
                'CurrentLiabilities': 'CurrentLiabilities',
                'LongTermDebt': 'LongTermDebt',
                'ShortTermDebt': 'ShortTermDebt',
                'ResearchAndDevelopmentExpense': 'R&D',
                'SellingGeneralAndAdministrativeExpense': 'SG&A'
            }
            
            for gaap_tag, metric_name in metrics.items():
                elements = root.findall(f'.//us-gaap:{gaap_tag}', ns)
                if elements:
                    # Take the most recent value
                    for element in elements:
                        context_ref = element.get('contextRef')
                        if context_ref and 'instant' in context_ref:  # Balance sheet item
                            try:
                                value = float(element.text.replace(',', ''))
                                financials[metric_name] = value
                                break
                            except:
                                pass
            
            return financials
            
        except Exception as e:
            print(f"Error parsing XBRL: {e}")
            return None
    
    def _parse_html_financials(self, soup):
        """Parse financial data from HTML tables"""
        
        financials = {}
        
        # Look for financial statement tables
        tables = soup.find_all('table')
        
        for table in tables:
            # Try to identify financial tables
            table_text = table.get_text().lower()
            
            # Check for balance sheet
            if any(keyword in table_text for keyword in ['balance sheet', 'statement of financial position']):
                financials.update(self._parse_balance_sheet(table))
            
            # Check for income statement
            elif any(keyword in table_text for keyword in ['income statement', 'statement of operations']):
                financials.update(self._parse_income_statement(table))
        
        return financials
    
    def _parse_balance_sheet(self, table):
        """Parse balance sheet table"""
        data = {}
        
        try:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].get_text().strip().lower()
                    value_text = cells[1].get_text().strip()
                    
                    # Clean value
                    value_text = value_text.replace(',', '').replace('$', '').replace('(', '-').replace(')', '')
                    
                    if value_text:
                        try:
                            value = float(value_text)
                            
                            # Map to standard names
                            if 'total assets' in label:
                                data['Assets'] = value
                            elif 'total liabilities' in label:
                                data['Liabilities'] = value
                            elif 'stockholders' in label or 'shareholders' in label:
                                data['StockholdersEquity'] = value
                            elif 'current assets' in label:
                                data['CurrentAssets'] = value
                            elif 'current liabilities' in label:
                                data['CurrentLiabilities'] = value
                            elif 'cash' in label:
                                data['Cash'] = value
                                
                        except:
                            pass
        except:
            pass
        
        return data
    
    def _parse_income_statement(self, table):
        """Parse income statement table"""
        data = {}
        
        try:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].get_text().strip().lower()
                    value_text = cells[1].get_text().strip()
                    
                    value_text = value_text.replace(',', '').replace('$', '').replace('(', '-').replace(')', '')
                    
                    if value_text:
                        try:
                            value = float(value_text)
                            
                            if 'revenue' in label or 'sales' in label:
                                data['Revenue'] = value
                            elif 'net income' in label:
                                data['NetIncome'] = value
                            elif 'operating income' in label:
                                data['OperatingIncome'] = value
                                
                        except:
                            pass
        except:
            pass
        
        return data
    
    def calculate_financial_ratios(self, financials):
        """Calculate financial ratios from extracted data"""
        
        ratios = {}
        
        # Liquidity Ratios
        if 'CurrentAssets' in financials and 'CurrentLiabilities' in financials:
            ratios['currentRatio'] = financials['CurrentAssets'] / financials['CurrentLiabilities']
        
        if 'Cash' in financials and 'CurrentLiabilities' in financials:
            ratios['cashRatio'] = financials['Cash'] / financials['CurrentLiabilities']
        
        # Profitability Ratios
        if 'NetIncome' in financials and 'Revenue' in financials:
            ratios['netProfitMargin'] = financials['NetIncome'] / financials['Revenue']
        
        if 'OperatingIncome' in financials and 'Revenue' in financials:
            ratios['operatingProfitMargin'] = financials['OperatingIncome'] / financials['Revenue']
        
        # Leverage Ratios
        if 'Liabilities' in financials and 'StockholdersEquity' in financials:
            ratios['debtEquityRatio'] = financials['Liabilities'] / financials['StockholdersEquity']
        
        # Return Ratios
        if 'NetIncome' in financials and 'Assets' in financials:
            ratios['returnOnAssets'] = financials['NetIncome'] / financials['Assets']
        
        if 'NetIncome' in financials and 'StockholdersEquity' in financials:
            ratios['returnOnEquity'] = financials['NetIncome'] / financials['StockholdersEquity']
        
        return ratios