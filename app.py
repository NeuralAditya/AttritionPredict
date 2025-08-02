import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from data_processor import DataProcessor
from visualizations import Visualizations
from enhanced_visualizations import EnhancedVisualizations
from advanced_analytics import AdvancedAnalytics
from utils import Utils
import io

# Page configuration
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

def main():
    st.title("🏢 HR Analytics Dashboard")
    st.markdown("### Employee Attrition Analysis - IBM Dataset")
    
    # Sidebar for file upload and filters
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload HR Dataset CSV",
            type=['csv'],
            help="Upload the IBM HR Analytics Employee Attrition dataset"
        )
        
        # Load data
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"Data loaded successfully! {len(df)} records found.")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
        elif not st.session_state.data_loaded:
            st.info("Please upload the HR dataset to begin analysis.")
            st.markdown("""
            **Expected Dataset Format:**
            - CSV file with employee data
            - Must include 'Attrition' column
            - Should contain demographic and job-related fields
            """)
            return
    
    # Main dashboard
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # Initialize processors
        processor = DataProcessor(df)
        viz = Visualizations(df)
        utils = Utils()
        
        # Process data
        processed_data = processor.process_data()
        
        # Sidebar filters
        with st.sidebar:
            st.header("🔍 Filters")
            
            # Department filter
            departments = ['All'] + sorted(df['Department'].unique().tolist())
            selected_dept = st.selectbox("Department", departments)
            
            # Job Role filter
            job_roles = ['All'] + sorted(df['JobRole'].unique().tolist())
            selected_role = st.selectbox("Job Role", job_roles)
            
            # Age group filter
            age_groups = ['All', '18-25', '26-35', '36-45', '46-55', '55+']
            selected_age = st.selectbox("Age Group", age_groups)
            
            # Education level filter
            education_levels = ['All'] + sorted(df['Education'].unique().tolist())
            selected_education = st.selectbox("Education Level", education_levels)
            
            # Gender filter
            genders = ['All'] + sorted(df['Gender'].unique().tolist())
            selected_gender = st.selectbox("Gender", genders)
            
            # Apply filters
            filtered_df = processor.apply_filters(
                df, selected_dept, selected_role, selected_age, 
                selected_education, selected_gender
            )
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Executive Dashboard", "📈 Overview", "👥 Demographics", "💼 Job Analysis", 
            "😊 Satisfaction", "🔗 Advanced Analytics", "🧠 ML Insights", "📋 Reports"
        ])
        
        # Initialize enhanced visualizations and advanced analytics
        enhanced_viz = EnhancedVisualizations(filtered_df)
        advanced_analytics = AdvancedAnalytics(filtered_df)
        
        with tab1:
            show_executive_dashboard(filtered_df, enhanced_viz, utils)
        
        with tab2:
            show_overview(filtered_df, viz, utils)
        
        with tab3:
            show_demographics(filtered_df, viz, utils)
        
        with tab4:
            show_job_analysis(filtered_df, viz, utils)
        
        with tab5:
            show_satisfaction_analysis(filtered_df, viz, utils)
        
        with tab6:
            show_advanced_analytics(filtered_df, enhanced_viz, utils)
        
        with tab7:
            show_ml_insights(filtered_df, advanced_analytics, utils)
        
        with tab8:
            show_reports(filtered_df, utils)

def show_overview(df, viz, utils):
    """Display overview metrics and key insights"""
    st.header("📊 Key Metrics Overview")
    
    # Calculate key metrics
    total_employees = len(df)
    attrition_count = len(df[df['Attrition'] == 'Yes'])
    attrition_rate = (attrition_count / total_employees) * 100 if total_employees > 0 else 0
    avg_age = df['Age'].mean()
    avg_tenure = df['YearsAtCompany'].mean()
    avg_income = df['MonthlyIncome'].mean()
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Employees", f"{total_employees:,}")
    
    with col2:
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    
    with col3:
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    with col4:
        st.metric("Average Tenure", f"{avg_tenure:.1f} years")
    
    with col5:
        st.metric("Average Income", f"${avg_income:,.0f}")
    
    st.divider()
    
    # Attrition trend visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Department")
        fig = viz.create_attrition_by_department(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Export button
        if st.button("📥 Export Department Chart", key="export_dept"):
            utils.export_chart(fig, "attrition_by_department.html")
    
    with col2:
        st.subheader("Attrition Distribution")
        fig = viz.create_attrition_donut(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Export button
        if st.button("📥 Export Donut Chart", key="export_donut"):
            utils.export_chart(fig, "attrition_distribution.html")
    
    # Monthly income analysis
    st.subheader("Monthly Income Analysis")
    fig = viz.create_income_analysis(df)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("📥 Export Income Analysis", key="export_income"):
        utils.export_chart(fig, "income_analysis.html")

def show_demographics(df, viz, utils):
    """Display demographic analysis"""
    st.header("👥 Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Age Group")
        fig = viz.create_age_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export Age Analysis", key="export_age"):
            utils.export_chart(fig, "age_analysis.html")
    
    with col2:
        st.subheader("Attrition by Gender")
        fig = viz.create_gender_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export Gender Analysis", key="export_gender"):
            utils.export_chart(fig, "gender_analysis.html")
    
    # Education analysis
    st.subheader("Education Level Impact")
    fig = viz.create_education_analysis(df)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("📥 Export Education Analysis", key="export_education"):
        utils.export_chart(fig, "education_analysis.html")
    
    # Distance from home analysis
    st.subheader("Distance from Home Impact")
    fig = viz.create_distance_analysis(df)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("📥 Export Distance Analysis", key="export_distance"):
        utils.export_chart(fig, "distance_analysis.html")

def show_job_analysis(df, viz, utils):
    """Display job-related analysis"""
    st.header("💼 Job-Related Analysis")
    
    # Job role analysis
    st.subheader("Attrition by Job Role")
    fig = viz.create_job_role_analysis(df)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("📥 Export Job Role Analysis", key="export_jobrole"):
        utils.export_chart(fig, "job_role_analysis.html")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Job Level Distribution")
        fig = viz.create_job_level_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export Job Level Analysis", key="export_joblevel"):
            utils.export_chart(fig, "job_level_analysis.html")
    
    with col2:
        st.subheader("Overtime Impact")
        fig = viz.create_overtime_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export Overtime Analysis", key="export_overtime"):
            utils.export_chart(fig, "overtime_analysis.html")
    
    # Years at company analysis
    st.subheader("Tenure Analysis")
    fig = viz.create_tenure_analysis(df)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("📥 Export Tenure Analysis", key="export_tenure"):
        utils.export_chart(fig, "tenure_analysis.html")

def show_satisfaction_analysis(df, viz, utils):
    """Display satisfaction-related analysis"""
    st.header("😊 Employee Satisfaction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Job Satisfaction Impact")
        fig = viz.create_job_satisfaction_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export Job Satisfaction", key="export_jobsat"):
            utils.export_chart(fig, "job_satisfaction_analysis.html")
    
    with col2:
        st.subheader("Work-Life Balance Impact")
        fig = viz.create_worklife_balance_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export Work-Life Balance", key="export_worklife"):
            utils.export_chart(fig, "worklife_balance_analysis.html")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Environment Satisfaction")
        fig = viz.create_environment_satisfaction_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export Environment Satisfaction", key="export_envsat"):
            utils.export_chart(fig, "environment_satisfaction_analysis.html")
    
    with col4:
        st.subheader("Job Involvement")
        fig = viz.create_job_involvement_analysis(df)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export Job Involvement", key="export_jobinv"):
            utils.export_chart(fig, "job_involvement_analysis.html")

def show_correlation_analysis(df, viz, utils):
    """Display correlation analysis"""
    st.header("🔗 Correlation Analysis")
    
    # Create binary attrition column for correlation
    df_corr = df.copy()
    df_corr['Attrition_Binary'] = (df_corr['Attrition'] == 'Yes').astype(int)
    
    # Select numeric columns for correlation
    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    if 'Attrition_Binary' in numeric_cols:
        numeric_cols.remove('Attrition_Binary')
    
    # Correlation with attrition
    st.subheader("Correlation with Attrition")
    correlations = df_corr[numeric_cols].corrwith(df_corr['Attrition_Binary']).sort_values(key=abs, ascending=False)
    
    fig = viz.create_correlation_chart(correlations)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("📥 Export Correlation Analysis", key="export_corr"):
        utils.export_chart(fig, "correlation_analysis.html")
    
    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    
    # Select top correlated features for heatmap
    top_features = correlations.head(10).index.tolist()
    top_features.append('Attrition_Binary')
    
    correlation_matrix = df_corr[top_features].corr()
    fig = viz.create_correlation_heatmap(correlation_matrix)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("📥 Export Correlation Heatmap", key="export_heatmap"):
        utils.export_chart(fig, "correlation_heatmap.html")
    
    # Top correlations table
    st.subheader("Top Correlations with Attrition")
    corr_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values
    })
    st.dataframe(corr_df, use_container_width=True)

def show_executive_dashboard(df, enhanced_viz, utils):
    """Display executive dashboard with advanced metrics and visualizations"""
    st.header("🏢 Executive Dashboard - Strategic HR Insights")
    
    # Interactive KPI Dashboard
    st.subheader("📊 Key Performance Indicators")
    fig_metrics = enhanced_viz.create_interactive_dashboard_metrics(df)
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    if st.button("📥 Export Executive Metrics", key="export_exec_metrics"):
        utils.export_chart(fig_metrics, "executive_dashboard_metrics.html")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 3D Employee Analysis")
        fig_3d = enhanced_viz.create_3d_scatter_analysis(df)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        if st.button("📥 Export 3D Analysis", key="export_3d"):
            utils.export_chart(fig_3d, "3d_employee_analysis.html")
    
    with col2:
        st.subheader("🌅 Organizational Hierarchy")
        fig_sunburst = enhanced_viz.create_sunburst_hierarchy(df)
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        if st.button("📥 Export Hierarchy Chart", key="export_sunburst"):
            utils.export_chart(fig_sunburst, "organizational_hierarchy.html")
    
    # Employee Profile Comparison
    st.subheader("🎭 Employee Profile Comparison")
    fig_radar = enhanced_viz.create_radar_chart_comparison(df)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    if st.button("📥 Export Profile Comparison", key="export_radar"):
        utils.export_chart(fig_radar, "employee_profile_comparison.html")
    
    # Retention Impact Analysis
    st.subheader("💧 Retention Impact Analysis")
    fig_waterfall = enhanced_viz.create_waterfall_chart(df)
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    if st.button("📥 Export Waterfall Analysis", key="export_waterfall"):
        utils.export_chart(fig_waterfall, "retention_impact_analysis.html")

def show_advanced_analytics(df, enhanced_viz, utils):
    """Display advanced analytics and visualizations"""
    st.header("🔗 Advanced Analytics & Multi-Dimensional Insights")
    
    # Parallel Coordinates
    st.subheader("🎼 Multi-Dimensional Analysis")
    st.markdown("Explore relationships between multiple employee attributes simultaneously")
    fig_parallel = enhanced_viz.create_parallel_coordinates(df)
    st.plotly_chart(fig_parallel, use_container_width=True)
    
    if st.button("📥 Export Parallel Coordinates", key="export_parallel"):
        utils.export_chart(fig_parallel, "parallel_coordinates_analysis.html")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Employee Distribution Map")
        fig_treemap = enhanced_viz.create_treemap_visualization(df)
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        if st.button("📥 Export Treemap", key="export_treemap"):
            utils.export_chart(fig_treemap, "employee_distribution_treemap.html")
    
    with col2:
        st.subheader("🔥 Enhanced Correlation Matrix")
        fig_heatmap = enhanced_viz.create_heatmap_correlation_matrix(df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        if st.button("📥 Export Correlation Matrix", key="export_enhanced_corr"):
            utils.export_chart(fig_heatmap, "enhanced_correlation_matrix.html")
    
    # Animated Timeline
    st.subheader("⏰ Employee Journey Timeline")
    st.markdown("Interactive timeline showing employee patterns across different age groups")
    fig_timeline = enhanced_viz.create_animated_timeline(df)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    if st.button("📥 Export Timeline", key="export_timeline"):
        utils.export_chart(fig_timeline, "employee_journey_timeline.html")
    
    # Advanced Statistical Analysis
    st.subheader("📈 Advanced Statistical Distributions")
    fig_advanced_box = enhanced_viz.create_advanced_box_plots(df)
    st.plotly_chart(fig_advanced_box, use_container_width=True)
    
    if st.button("📥 Export Statistical Analysis", key="export_advanced_stats"):
        utils.export_chart(fig_advanced_box, "advanced_statistical_analysis.html")

def show_ml_insights(df, advanced_analytics, utils):
    """Display machine learning insights and predictive analytics"""
    st.header("🧠 Machine Learning Insights & Predictive Analytics")
    
    # Feature Importance Analysis
    st.subheader("🎯 Feature Importance Analysis")
    st.markdown("Discover which factors most strongly predict employee attrition using Random Forest algorithm")
    
    try:
        fig_importance, feature_df, accuracy = advanced_analytics.feature_importance_analysis()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            st.subheader("Top 5 Predictors")
            for i, row in feature_df.head(5).iterrows():
                st.metric(row['feature'], f"{row['importance']:.3f}")
        
        if st.button("📥 Export Feature Importance", key="export_feature_importance"):
            utils.export_chart(fig_importance, "feature_importance_analysis.html")
            utils.export_data(feature_df, "feature_importance_data.csv")
    
    except Exception as e:
        st.error(f"Error in feature importance analysis: {str(e)}")
        st.info("This analysis requires sufficient data variety to work properly.")
    
    # PCA Analysis
    st.subheader("🔍 Principal Component Analysis")
    st.markdown("Dimensionality reduction to identify key patterns in employee data")
    
    try:
        fig_pca, explained_variance = advanced_analytics.perform_pca_analysis()
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Show explained variance
        st.info(f"First two components explain {sum(explained_variance[:2]):.1%} of the total variance")
        
        if st.button("📥 Export PCA Analysis", key="export_pca"):
            utils.export_chart(fig_pca, "pca_analysis.html")
    
    except Exception as e:
        st.error(f"Error in PCA analysis: {str(e)}")
        st.info("PCA analysis requires numeric data and sufficient samples.")
    
    # Clustering Analysis
    st.subheader("👥 Employee Clustering Analysis")
    st.markdown("Identify natural groups of employees with similar characteristics")
    
    try:
        fig_clustering, clustered_df = advanced_analytics.perform_clustering_analysis()
        st.plotly_chart(fig_clustering, use_container_width=True)
        
        # Cluster insights
        st.subheader("Cluster Insights")
        cluster_summary = clustered_df.groupby('Cluster').agg({
            'Attrition_Binary': ['count', 'mean'],
            'Age': 'mean',
            'MonthlyIncome': 'mean'
        }).round(2)
        
        st.dataframe(cluster_summary, use_container_width=True)
        
        if st.button("📥 Export Clustering Analysis", key="export_clustering"):
            utils.export_chart(fig_clustering, "clustering_analysis.html")
            utils.export_data(clustered_df, "clustered_employee_data.csv")
    
    except Exception as e:
        st.error(f"Error in clustering analysis: {str(e)}")
        st.info("Clustering analysis requires sufficient data points to identify patterns.")
    
    # Survival Analysis
    st.subheader("📈 Employee Survival Analysis")
    st.markdown("Analyze employee retention patterns and risk factors over time")
    
    try:
        fig_survival, survival_data = advanced_analytics.survival_analysis()
        st.plotly_chart(fig_survival, use_container_width=True)
        
        # Survival insights
        st.subheader("Survival Rate by Tenure")
        st.dataframe(survival_data, use_container_width=True)
        
        if st.button("📥 Export Survival Analysis", key="export_survival"):
            utils.export_chart(fig_survival, "employee_survival_analysis.html")
            utils.export_data(survival_data, "survival_analysis_data.csv")
    
    except Exception as e:
        st.error(f"Error in survival analysis: {str(e)}")
        st.info("Survival analysis requires tenure and attrition data.")

def show_reports(df, utils):
    """Display comprehensive reports and export options"""
    st.header("📋 Comprehensive HR Analytics Reports")
    
    # Executive Summary
    st.subheader("📊 Executive Summary")
    
    total_employees = len(df)
    attrition_count = len(df[df['Attrition'] == 'Yes'])
    attrition_rate = (attrition_count / total_employees) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", f"{total_employees:,}")
    with col2:
        st.metric("Employees Left", f"{attrition_count:,}")
    with col3:
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    with col4:
        avg_tenure = df['YearsAtCompany'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} years")
    
    # Key Insights
    insights_text = utils.create_insights_text(df)
    st.markdown(insights_text)
    
    # Departmental Analysis
    st.subheader("🏢 Departmental Analysis")
    dept_analysis = df.groupby('Department').agg({
        'Attrition': ['count', lambda x: (x == 'Yes').sum()],
        'MonthlyIncome': 'mean',
        'Age': 'mean',
        'YearsAtCompany': 'mean'
    }).round(2)
    
    dept_analysis.columns = ['Total_Employees', 'Attrition_Count', 'Avg_Income', 'Avg_Age', 'Avg_Tenure']
    dept_analysis['Attrition_Rate'] = (dept_analysis['Attrition_Count'] / dept_analysis['Total_Employees'] * 100).round(2)
    dept_analysis = dept_analysis.sort_values('Attrition_Rate', ascending=False)
    
    st.dataframe(dept_analysis, use_container_width=True)
    
    # Job Role Analysis
    st.subheader("💼 Job Role Analysis")
    role_analysis = df.groupby('JobRole').agg({
        'Attrition': ['count', lambda x: (x == 'Yes').sum()],
        'MonthlyIncome': 'mean',
        'JobSatisfaction': 'mean'
    }).round(2)
    
    role_analysis.columns = ['Total_Employees', 'Attrition_Count', 'Avg_Income', 'Avg_Job_Satisfaction']
    role_analysis['Attrition_Rate'] = (role_analysis['Attrition_Count'] / role_analysis['Total_Employees'] * 100).round(2)
    role_analysis = role_analysis.sort_values('Attrition_Rate', ascending=False)
    
    st.dataframe(role_analysis, use_container_width=True)
    
    # Data Export Section
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Full Dataset", key="export_full_data"):
            utils.export_data(df, "hr_analytics_full_dataset.csv")
    
    with col2:
        if st.button("🏢 Export Department Analysis", key="export_dept_analysis"):
            utils.export_data(dept_analysis.reset_index(), "department_analysis.csv")
    
    with col3:
        if st.button("💼 Export Job Role Analysis", key="export_role_analysis"):
            utils.export_data(role_analysis.reset_index(), "job_role_analysis.csv")
    
    # Recommendations
    st.subheader("💡 Strategic Recommendations")
    
    # Calculate recommendations based on data
    high_risk_dept = dept_analysis.index[0]
    high_risk_role = role_analysis.index[0]
    
    recommendations = f"""
    ### 🎯 Priority Actions
    
    **1. Focus on {high_risk_dept} Department**
    - Attrition rate: {dept_analysis.loc[high_risk_dept, 'Attrition_Rate']:.1f}%
    - Implement targeted retention strategies
    - Conduct exit interviews to understand specific issues
    
    **2. Address {high_risk_role} Role Challenges** 
    - Highest attrition rate: {role_analysis.loc[high_risk_role, 'Attrition_Rate']:.1f}%
    - Review job responsibilities and workload
    - Consider career development opportunities
    
    **3. Income and Satisfaction Correlation**
    - Monitor employees with below-median income
    - Implement regular satisfaction surveys
    - Create clear career progression paths
    
    **4. Work-Life Balance Initiative**
    - Reduce overtime requirements where possible
    - Implement flexible working arrangements
    - Provide mental health and wellness support
    """
    
    st.markdown(recommendations)

if __name__ == "__main__":
    main()
