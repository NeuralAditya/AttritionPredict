import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

class AdvancedAnalytics:
    def __init__(self, df):
        self.df = df.copy()
        self.processed_df = None
        
    def prepare_data_for_ml(self):
        """Prepare data for machine learning analysis"""
        df = self.df.copy()
        
        # Create binary target
        df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)
        
        # Select numeric features for ML
        numeric_features = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 
                          'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate',
                          'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                          'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
                          'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
                          'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                          'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
                          'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        # Filter only existing columns
        available_features = [col for col in numeric_features if col in df.columns]
        
        self.processed_df = df[available_features + ['Attrition_Binary']].dropna()
        return self.processed_df
        
    def perform_pca_analysis(self):
        """Perform Principal Component Analysis"""
        df_ml = self.prepare_data_for_ml()
        
        # Prepare features
        X = df_ml.drop(['Attrition_Binary'], axis=1)
        y = df_ml['Attrition_Binary']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PCA Scatter Plot', 'Explained Variance Ratio', 
                          'Cumulative Explained Variance', 'Feature Importance in PC1 & PC2'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # PCA scatter plot
        colors = ['red' if x == 1 else 'blue' for x in y]
        fig.add_trace(
            go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1], 
                      mode='markers', 
                      marker=dict(color=colors, opacity=0.6),
                      name='Employees',
                      text=[f'Attrition: {"Yes" if x == 1 else "No"}' for x in y]),
            row=1, col=1
        )
        
        # Explained variance ratio
        fig.add_trace(
            go.Bar(x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                   y=pca.explained_variance_ratio_,
                   name='Explained Variance'),
            row=1, col=2
        )
        
        # Cumulative explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        fig.add_trace(
            go.Scatter(x=list(range(1, len(cumsum_var) + 1)),
                      y=cumsum_var,
                      mode='lines+markers',
                      name='Cumulative Variance'),
            row=2, col=1
        )
        
        # Feature importance in first two components
        feature_names = X.columns
        pc1_importance = abs(pca.components_[0])
        pc2_importance = abs(pca.components_[1])
        
        fig.add_trace(
            go.Bar(x=feature_names, y=pc1_importance, name='PC1 Importance'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Principal Component Analysis")
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="Component", row=1, col=2)
        fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=2)
        fig.update_xaxes(title_text="Component", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Variance", row=2, col=1)
        fig.update_xaxes(title_text="Features", row=2, col=2)
        fig.update_yaxes(title_text="Importance", row=2, col=2)
        
        return fig, pca.explained_variance_ratio_
    
    def perform_clustering_analysis(self):
        """Perform K-means clustering analysis"""
        df_ml = self.prepare_data_for_ml()
        
        # Prepare features
        X = df_ml.drop(['Attrition_Binary'], axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Perform clustering with optimal k (let's use 4)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        df_clustered = df_ml.copy()
        df_clustered['Cluster'] = clusters
        
        # Create clustering visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Elbow Method', 'Cluster Distribution', 
                          'Clusters vs Attrition', 'Cluster Characteristics'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # Elbow method
        fig.add_trace(
            go.Scatter(x=list(K_range), y=inertias, 
                      mode='lines+markers', name='Inertia'),
            row=1, col=1
        )
        
        # Cluster distribution
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=cluster_counts.index, y=cluster_counts.values, 
                   name='Cluster Size'),
            row=1, col=2
        )
        
        # Clusters vs Attrition
        cluster_attrition = df_clustered.groupby(['Cluster', 'Attrition_Binary']).size().unstack(fill_value=0)
        cluster_attrition_rate = (cluster_attrition[1] / (cluster_attrition[0] + cluster_attrition[1]) * 100).round(2)
        
        fig.add_trace(
            go.Bar(x=cluster_attrition_rate.index, y=cluster_attrition_rate.values,
                   name='Attrition Rate %'),
            row=2, col=1
        )
        
        # Cluster characteristics heatmap
        cluster_means = df_clustered.groupby('Cluster')[X.columns[:10]].mean()  # Top 10 features
        fig.add_trace(
            go.Heatmap(z=cluster_means.values.T,
                      x=cluster_means.index,
                      y=cluster_means.columns,
                      colorscale='RdBu'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Employee Clustering Analysis")
        fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_xaxes(title_text="Cluster", row=1, col=2)
        fig.update_yaxes(title_text="Number of Employees", row=1, col=2)
        fig.update_xaxes(title_text="Cluster", row=2, col=1)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=2, col=1)
        
        return fig, df_clustered
    
    def feature_importance_analysis(self):
        """Perform feature importance analysis using Random Forest"""
        df_ml = self.prepare_data_for_ml()
        
        # Prepare data
        X = df_ml.drop(['Attrition_Binary'], axis=1)
        y = df_ml['Attrition_Binary']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Top 15 Most Important Features', 'Feature Importance Distribution'),
            specs=[[{"type": "bar"}], [{"type": "histogram"}]]
        )
        
        # Top features
        top_features = feature_importance.head(15)
        fig.add_trace(
            go.Bar(x=top_features['importance'], y=top_features['feature'],
                   orientation='h', name='Feature Importance'),
            row=1, col=1
        )
        
        # Distribution
        fig.add_trace(
            go.Histogram(x=feature_importance['importance'], nbinsx=20,
                        name='Importance Distribution'),
            row=2, col=1
        )
        
        fig.update_layout(height=700, title_text="Feature Importance Analysis")
        fig.update_xaxes(title_text="Importance Score", row=1, col=1)
        fig.update_yaxes(title_text="Features", row=1, col=1)
        fig.update_xaxes(title_text="Importance Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        # Get predictions and model performance
        y_pred = rf.predict(X_test)
        accuracy = rf.score(X_test, y_test)
        
        return fig, feature_importance, accuracy
    
    def survival_analysis(self):
        """Perform employee tenure survival analysis"""
        df = self.df.copy()
        
        # Create tenure bins
        tenure_bins = [0, 1, 2, 5, 10, 20, 50]
        tenure_labels = ['0-1yr', '1-2yr', '2-5yr', '5-10yr', '10-20yr', '20+yr']
        df['TenureBin'] = pd.cut(df['YearsAtCompany'], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
        
        # Calculate survival rates
        survival_data = df.groupby('TenureBin').agg({
            'Attrition': ['count', lambda x: (x == 'No').sum()]
        }).round(2)
        
        survival_data.columns = ['Total', 'Survived']
        survival_data['SurvivalRate'] = (survival_data['Survived'] / survival_data['Total'] * 100).round(2)
        survival_data = survival_data.reset_index()
        
        # Create survival curve
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Employee Survival Rate by Tenure', 'Attrition Risk by Department',
                          'Monthly Income vs Tenure (Survivors)', 'Job Satisfaction Impact'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "violin"}]]
        )
        
        # Survival curve
        fig.add_trace(
            go.Scatter(x=survival_data['TenureBin'], y=survival_data['SurvivalRate'],
                      mode='lines+markers', name='Survival Rate',
                      line=dict(width=3, color='green')),
            row=1, col=1
        )
        
        # Attrition risk by department
        dept_risk = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100).sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=dept_risk.index, y=dept_risk.values,
                   name='Attrition Risk %', marker_color='red'),
            row=1, col=2
        )
        
        # Income vs Tenure for survivors
        survivors = df[df['Attrition'] == 'No']
        fig.add_trace(
            go.Scatter(x=survivors['YearsAtCompany'], y=survivors['MonthlyIncome'],
                      mode='markers', name='Survivors', opacity=0.6,
                      marker=dict(color='blue')),
            row=2, col=1
        )
        
        # Job satisfaction impact
        fig.add_trace(
            go.Violin(x=df['Attrition'], y=df['JobSatisfaction'],
                     name='Job Satisfaction Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Employee Survival Analysis")
        fig.update_xaxes(title_text="Tenure Range", row=1, col=1)
        fig.update_yaxes(title_text="Survival Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="Department", row=1, col=2)
        fig.update_yaxes(title_text="Attrition Risk (%)", row=1, col=2)
        fig.update_xaxes(title_text="Years at Company", row=2, col=1)
        fig.update_yaxes(title_text="Monthly Income", row=2, col=1)
        fig.update_xaxes(title_text="Attrition Status", row=2, col=2)
        fig.update_yaxes(title_text="Job Satisfaction", row=2, col=2)
        
        return fig, survival_data
    
    def cohort_analysis(self):
        """Perform cohort analysis based on employee hiring patterns"""
        df = self.df.copy()
        
        # Create hiring year cohorts
        df['HiringYear'] = 2024 - df['YearsAtCompany']  # Assuming current year is 2024
        df['CohortYear'] = pd.cut(df['HiringYear'], 
                                 bins=[2000, 2010, 2015, 2020, 2024], 
                                 labels=['2000-2010', '2011-2015', '2016-2020', '2021-2024'])
        
        # Cohort analysis
        cohort_data = df.groupby(['CohortYear', 'Attrition']).size().unstack(fill_value=0)
        cohort_data['Total'] = cohort_data.sum(axis=1)
        cohort_data['AttritionRate'] = (cohort_data.get('Yes', 0) / cohort_data['Total'] * 100).round(2)
        cohort_data['RetentionRate'] = (cohort_data.get('No', 0) / cohort_data['Total'] * 100).round(2)
        
        # Create cohort visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cohort Retention Rates', 'Cohort Size Distribution',
                          'Average Income by Cohort', 'Job Satisfaction by Cohort'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Retention rates
        fig.add_trace(
            go.Bar(x=cohort_data.index, y=cohort_data['RetentionRate'],
                   name='Retention Rate %', marker_color='green'),
            row=1, col=1
        )
        
        # Cohort sizes
        fig.add_trace(
            go.Pie(labels=cohort_data.index, values=cohort_data['Total'],
                   name='Cohort Sizes'),
            row=1, col=2
        )
        
        # Average income by cohort
        avg_income = df.groupby('CohortYear')['MonthlyIncome'].mean()
        fig.add_trace(
            go.Bar(x=avg_income.index, y=avg_income.values,
                   name='Average Income', marker_color='blue'),
            row=2, col=1
        )
        
        # Job satisfaction by cohort
        for cohort in df['CohortYear'].dropna().unique():
            cohort_data_subset = df[df['CohortYear'] == cohort]['JobSatisfaction']
            fig.add_trace(
                go.Box(y=cohort_data_subset, name=str(cohort)),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Employee Cohort Analysis")
        
        return fig, cohort_data