import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Corporate Credit & Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)
@st.cache_data
def load_data():
    return pd.read_csv("credit_ratings_multimodal.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading 'credit_ratings_multimodal.csv'. Please ensure the file is in the same directory as this script.\n\nError details: {e}")
    st.stop()

st.sidebar.header("Filter Dashboard")
if 'Sector' in df.columns:
    all_sectors = sorted(df['Sector'].astype(str).unique())
    selected_sectors = st.sidebar.multiselect("Select Sector", all_sectors, default=all_sectors)
else:
    selected_sectors = []
    st.sidebar.warning("Column 'Sector' not found.")

# Rating Filter
if 'Rating_Merged' in df.columns:
    all_ratings = sorted(df['Rating_Merged'].astype(str).unique())
    selected_ratings = st.sidebar.multiselect("Select Rating", all_ratings, default=all_ratings)
else:
    selected_ratings = []
    st.sidebar.warning("Column 'Rating_Merged' not found.")

df_filtered = df if not selected_sectors or not selected_ratings else df[
    (df['Sector'].isin(selected_sectors)) & 
    (df['Rating_Merged'].isin(selected_ratings))
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data Points:** {len(df_filtered)}")
if 'Ticker' in df.columns:
    st.sidebar.markdown(f"**Companies:** {df_filtered['Ticker'].nunique()}")

st.title("📊 Corporate Financial & NLP Analysis")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Companies", len(df_filtered))

if 'netProfitMargin' in df_filtered.columns:
    col2.metric("Avg Net Profit Margin", f"{df_filtered['netProfitMargin'].mean()*100:.2f}%")
else:
    col2.metric("Avg Net Profit Margin", "N/A")

if 'debtEquityRatio' in df_filtered.columns:
    col3.metric("Avg Debt/Equity", f"{df_filtered['debtEquityRatio'].mean():.2f}")
else:
    col3.metric("Avg Debt/Equity", "N/A")

if 'nlp_sentiment' in df_filtered.columns:
    col4.metric("Avg NLP Sentiment", f"{df_filtered['nlp_sentiment'].mean():.2f}")
else:
    col4.metric("Avg NLP Sentiment", "N/A")

tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Financial Deep Dive", "NLP Sentiment Analysis", "Company Finder"])
with tab1:
    st.subheader("Market Composition")
    
    c1, c2 = st.columns(2)
    
    with c1:
        if 'Sector' in df_filtered.columns:
            st.markdown("**Sector Distribution**")
            fig_sector = px.pie(df_filtered, names='Sector', title='Companies by Sector', hole=0.4)
            st.plotly_chart(fig_sector, use_container_width=True)
        
    with c2:
        if 'Rating_Merged' in df_filtered.columns:
            st.markdown("**Credit Rating Distribution**")
            rating_counts = df_filtered['Rating_Merged'].value_counts().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            fig_rating = px.bar(rating_counts, x='Rating', y='Count', color='Rating', title='Count of Ratings')
            st.plotly_chart(fig_rating, use_container_width=True)

    st.markdown("**Financial Heatmap (Average Metrics by Sector)**")
    heatmap_cols = ['currentRatio', 'netProfitMargin', 'debtEquityRatio', 'returnOnEquity', 'nlp_sentiment']
    existing_heatmap_cols = [c for c in heatmap_cols if c in df_filtered.columns]
    
    if 'Sector' in df_filtered.columns and existing_heatmap_cols:
        sector_stats = df_filtered.groupby('Sector')[existing_heatmap_cols].mean().reset_index()
        
        fig_heat = px.imshow(
            sector_stats.set_index('Sector').T, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Average Metrics by Sector"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Insufficient data for Heatmap.")

with tab2:
    st.subheader("Financial Ratios Analysis")
    
    financial_options = ['netProfitMargin', 'returnOnEquity', 'debtEquityRatio', 
                         'currentRatio', 'quickRatio', 'assetTurnover']
    available_fin_options = [c for c in financial_options if c in df_filtered.columns]
    
    if available_fin_options:
        selected_metric = st.selectbox("Select Metric to Analyze:", available_fin_options, index=0)
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            if 'Rating_Merged' in df_filtered.columns:
                st.markdown(f"**{selected_metric} by Rating Category**")
                fig_box = px.box(
                    df_filtered, 
                    x='Rating_Merged', 
                    y=selected_metric, 
                    color='Rating_Merged',
                    points="outliers",
                    title=f"Distribution of {selected_metric}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
        with row1_col2:
            if 'returnOnAssets' in df_filtered.columns and 'Sector' in df_filtered.columns:
                st.markdown(f"**{selected_metric} vs. Return on Assets**")
                fig_scatter = px.scatter(
                    df_filtered,
                    x='returnOnAssets',
                    y=selected_metric,
                    color='Sector',
                    hover_data=['Name', 'Rating_Merged'] if 'Name' in df_filtered.columns else None,
                    title=f"Correlation: ROA vs {selected_metric}"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Selected financial metrics not found in dataset.")

with tab3:
    st.subheader("NLP & Risk Analytics")
    st.info("Visualizing insights extracted from MD&A reports (Sentiment, Risk, Uncertainty).")
    
    required_nlp = ['nlp_sentiment', 'nlp_risk', 'Rating_Merged', 'nlp_readability', 'Sector', 'Name']
    available_nlp = [c for c in required_nlp if c in df_filtered.columns]
    
    if len(available_nlp) == len(required_nlp):
        nlp_df = df_filtered.dropna(subset=['nlp_sentiment', 'nlp_risk'])
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Sentiment vs. Risk Score**")
            fig_nlp = px.scatter(
                nlp_df,
                x='nlp_risk',
                y='nlp_sentiment',
                color='Rating_Merged',
                size='nlp_readability',
                hover_data=['Name', 'Sector'],
                title="Does Higher Risk Language correlate with Sentiment?",
                labels={'nlp_risk': 'Risk Score (High = Risky)', 'nlp_sentiment': 'Sentiment Score'}
            )
            st.plotly_chart(fig_nlp, use_container_width=True)
            
        with c2:
            st.markdown("**Linguistic Attributes by Sector**")
            nlp_attributes = ['nlp_certainty', 'nlp_uncertainty', 'nlp_litigiousness', 'nlp_fraud']
            existing_nlp_attrs = [a for a in nlp_attributes if a in nlp_df.columns]
            
            if existing_nlp_attrs:
                nlp_melted = nlp_df.groupby('Sector')[existing_nlp_attrs].mean().reset_index().melt(id_vars='Sector')
                
                fig_radar = px.bar(
                    nlp_melted, 
                    x='Sector', 
                    y='value', 
                    color='variable', 
                    barmode='group',
                    title="Average Language Attributes by Sector"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
        if 'netProfitMargin' in nlp_df.columns:
            st.markdown("**Word Complexity (Readability) vs Financial Performance**")
            fig_readability = px.density_heatmap(
                nlp_df, 
                x='nlp_readability', 
                y='netProfitMargin', 
                nbinsx=20, 
                nbinsy=20, 
                title="Density: Readability vs Profit Margin"
            )
            st.plotly_chart(fig_readability, use_container_width=True)
    else:
        st.warning("Some NLP columns are missing from the dataset. Please check CSV headers.")

with tab4:
    st.subheader("Single Company Drill-down")
    
    if 'Ticker' in df.columns:
        selected_ticker = st.selectbox("Search Company by Ticker:", df['Ticker'].unique())
        company_data = df[df['Ticker'] == selected_ticker].iloc[0]
        name = company_data.get('Name', 'Unknown')
        ticker = company_data.get('Ticker', 'Unknown')
        sector = company_data.get('Sector', 'Unknown')
        rating = company_data.get('Rating_Merged', 'Unknown')
        agency = company_data.get('Rating Agency Name', 'Unknown')
        
        st.markdown(f"### {name} ({ticker})")
        st.markdown(f"**Sector:** {sector} | **Rating:** {rating} | **Agency:** {agency}")
        
        dc1, dc2, dc3 = st.columns(3)
        
        with dc1:
            st.markdown("#### 💰 Liquidity")
            liq_data = {}
            if 'currentRatio' in company_data: liq_data['Current Ratio'] = company_data['currentRatio']
            if 'quickRatio' in company_data: liq_data['Quick Ratio'] = company_data['quickRatio']
            if 'cashRatio' in company_data: liq_data['Cash Ratio'] = company_data['cashRatio']
            
            if liq_data:
                st.dataframe(pd.DataFrame(list(liq_data.items()), columns=['Metric', 'Value']).set_index('Metric'))

        with dc2:
            st.markdown("#### 📈 Profitability")
            prof_data = {}
            if 'netProfitMargin' in company_data: prof_data['Net Profit Margin'] = company_data['netProfitMargin']
            if 'grossProfitMargin' in company_data: prof_data['Gross Margin'] = company_data['grossProfitMargin']
            if 'returnOnEquity' in company_data: prof_data['ROE'] = company_data['returnOnEquity']

            if prof_data:
                st.dataframe(pd.DataFrame(list(prof_data.items()), columns=['Metric', 'Value']).set_index('Metric'))

        with dc3:
            st.markdown("#### 🧠 NLP Analysis")
            sentiment_val = company_data.get('nlp_sentiment', np.nan)
            risk_val = company_data.get('nlp_risk', np.nan)
            
            st.metric("Sentiment Score", f"{sentiment_val:.4f}" if pd.notna(sentiment_val) else "N/A")
            st.metric("Risk Score", f"{risk_val:.4f}" if pd.notna(risk_val) else "N/A")

        st.markdown("#### 📝 Raw Data Record")
        st.dataframe(company_data.to_frame().T)
    else:
        st.error("Ticker column not found to perform search.")

st.markdown("---")
with st.expander("View & Download Filtered Data"):
    st.dataframe(df_filtered)
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download CSV",
        csv,
        "filtered_financial_data.csv",
        "text/csv",
        key='download-csv'
    )