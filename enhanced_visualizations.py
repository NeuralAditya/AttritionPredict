import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class EnhancedVisualizations:
    def __init__(self, df):
        self.df = df
        self.color_palette = px.colors.qualitative.Set3
        self.attrition_colors = {'Yes': '#FF6B6B', 'No': '#4ECDC4'}
        
    def create_interactive_dashboard_metrics(self, df):
        """Create an interactive metrics dashboard with gauges and KPIs"""
        total_employees = len(df)
        attrition_count = len(df[df['Attrition'] == 'Yes'])
        attrition_rate = (attrition_count / total_employees) * 100 if total_employees > 0 else 0
        
        # Create subplot with gauges and metrics
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Attrition Rate', 'Average Age', 'Average Tenure',
                          'Monthly Income Range', 'Job Satisfaction', 'Work-Life Balance'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Attrition Rate Gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = attrition_rate,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Attrition Rate (%)"},
            delta = {'reference': 15},  # Industry average
            gauge = {'axis': {'range': [None, 50]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 10], 'color': "lightgray"},
                        {'range': [10, 20], 'color': "yellow"},
                        {'range': [20, 50], 'color': "red"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 20}}
        ), row=1, col=1)
        
        # Average Age Gauge
        avg_age = df['Age'].mean()
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = avg_age,
            title = {'text': "Average Age (years)"},
            gauge = {'axis': {'range': [20, 70]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [20, 30], 'color': "lightblue"},
                        {'range': [30, 50], 'color': "blue"},
                        {'range': [50, 70], 'color': "darkblue"}]}
        ), row=1, col=2)
        
        # Average Tenure Gauge
        avg_tenure = df['YearsAtCompany'].mean()
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = avg_tenure,
            title = {'text': "Average Tenure (years)"},
            gauge = {'axis': {'range': [0, 25]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 15], 'color': "green"},
                        {'range': [15, 25], 'color': "darkgreen"}]}
        ), row=1, col=3)
        
        # Income distribution
        income_bins = pd.cut(df['MonthlyIncome'], bins=5)
        income_dist = pd.Series(income_bins).value_counts().sort_index()
        fig.add_trace(go.Bar(
            x=[f"${int(interval.left/1000)}K-${int(interval.right/1000)}K" for interval in income_dist.index],
            y=income_dist.values,
            name="Income Distribution",
            marker_color='lightblue'
        ), row=2, col=1)
        
        # Job satisfaction distribution
        job_sat_dist = df['JobSatisfaction'].value_counts().sort_index()
        satisfaction_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        fig.add_trace(go.Bar(
            x=[satisfaction_labels.get(x, str(x)) for x in job_sat_dist.index],
            y=job_sat_dist.values,
            name="Job Satisfaction",
            marker_color='orange'
        ), row=2, col=2)
        
        # Work-life balance distribution
        wlb_dist = df['WorkLifeBalance'].value_counts().sort_index()
        wlb_labels = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
        fig.add_trace(go.Bar(
            x=[wlb_labels.get(x, str(x)) for x in wlb_dist.index],
            y=wlb_dist.values,
            name="Work-Life Balance",
            marker_color='purple'
        ), row=2, col=3)
        
        fig.update_layout(height=700, title_text="Executive Dashboard - Key HR Metrics")
        return fig
    
    def create_3d_scatter_analysis(self, df):
        """Create 3D scatter plot for multi-dimensional analysis"""
        fig = go.Figure(data=[go.Scatter3d(
            x=df['Age'],
            y=df['MonthlyIncome'],
            z=df['YearsAtCompany'],
            mode='markers',
            marker=dict(
                size=5,
                color=df['Attrition'].map({'Yes': 1, 'No': 0}),
                colorscale=[[0, '#4ECDC4'], [1, '#FF6B6B']],
                opacity=0.8,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            text=[f"Age: {age}<br>Income: ${income:,}<br>Tenure: {tenure}yr<br>Attrition: {attr}" 
                  for age, income, tenure, attr in zip(df['Age'], df['MonthlyIncome'], 
                                                      df['YearsAtCompany'], df['Attrition'])],
            hovertemplate="%{text}<extra></extra>"
        )])
        
        fig.update_layout(
            title="3D Employee Analysis: Age vs Income vs Tenure",
            scene=dict(
                xaxis_title='Age',
                yaxis_title='Monthly Income ($)',
                zaxis_title='Years at Company'
            ),
            height=600
        )
        
        return fig
    
    def create_sunburst_hierarchy(self, df):
        """Create sunburst chart for hierarchical data exploration"""
        # Prepare data for sunburst
        df_sunburst = df.copy()
        df_sunburst['Count'] = 1
        
        # Create hierarchy: Department -> JobRole -> Attrition
        hierarchy_data = df_sunburst.groupby(['Department', 'JobRole', 'Attrition'])['Count'].sum().reset_index()
        
        fig = go.Figure(go.Sunburst(
            labels=hierarchy_data['Department'].tolist() + 
                   hierarchy_data['JobRole'].tolist() + 
                   hierarchy_data['Attrition'].tolist(),
            parents=[''] * len(hierarchy_data['Department'].unique()) +
                    hierarchy_data['Department'].tolist() +
                    hierarchy_data['JobRole'].tolist(),
            values=[1] * len(hierarchy_data['Department'].unique()) +
                   [1] * len(hierarchy_data) +
                   hierarchy_data['Count'].tolist(),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
            maxdepth=3
        ))
        
        fig.update_layout(
            title="Organizational Hierarchy: Department → Job Role → Attrition Status",
            height=600
        )
        
        return fig
    
    def create_animated_timeline(self, df):
        """Create animated timeline showing trends over tenure"""
        # Create tenure-based timeline
        df_timeline = df.copy()
        
        # Create age groups for animation
        df_timeline['AgeGroup'] = pd.cut(df_timeline['Age'], 
                                       bins=[0, 25, 35, 45, 55, 100],
                                       labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Aggregate data by tenure and age group
        timeline_data = df_timeline.groupby(['YearsAtCompany', 'AgeGroup', 'Attrition']).size().reset_index(name='Count')
        
        fig = px.scatter(timeline_data, 
                        x='YearsAtCompany', 
                        y='Count',
                        size='Count',
                        color='Attrition',
                        animation_frame='AgeGroup',
                        hover_name='AgeGroup',
                        title="Employee Attrition Timeline by Age Groups",
                        labels={'YearsAtCompany': 'Years at Company', 'Count': 'Number of Employees'},
                        color_discrete_map=self.attrition_colors)
        
        fig.update_layout(height=500)
        return fig
    
    def create_radar_chart_comparison(self, df):
        """Create radar chart comparing different employee segments"""
        # Create comparison groups
        stayed = df[df['Attrition'] == 'No']
        left = df[df['Attrition'] == 'Yes']
        
        # Calculate averages for radar chart
        metrics = ['Age', 'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction', 
                  'JobInvolvement', 'RelationshipSatisfaction']
        
        stayed_avg = [stayed[metric].mean() for metric in metrics]
        left_avg = [left[metric].mean() for metric in metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=stayed_avg,
            theta=metrics,
            fill='toself',
            name='Employees Who Stayed',
            line_color='rgba(78, 205, 196, 0.8)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=left_avg,
            theta=metrics,
            fill='toself',
            name='Employees Who Left',
            line_color='rgba(255, 107, 107, 0.8)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(stayed_avg), max(left_avg)) * 1.1]
                )),
            showlegend=True,
            title="Employee Profile Comparison: Stayed vs Left",
            height=500
        )
        
        return fig
    
    def create_parallel_coordinates(self, df):
        """Create parallel coordinates plot for multi-dimensional analysis"""
        # Select relevant numeric columns
        numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 
                       'WorkLifeBalance', 'EnvironmentSatisfaction', 'JobLevel']
        
        # Prepare data
        df_parallel = df[numeric_cols + ['Attrition']].copy()
        df_parallel['AttritionBinary'] = (df_parallel['Attrition'] == 'Yes').astype(int)
        
        # Normalize data for better visualization
        for col in numeric_cols:
            df_parallel[f'{col}_norm'] = (df_parallel[col] - df_parallel[col].min()) / (df_parallel[col].max() - df_parallel[col].min())
        
        # Create dimensions for parallel coordinates
        dimensions = []
        for col in numeric_cols:
            dimensions.append(dict(
                range=[df_parallel[col].min(), df_parallel[col].max()],
                label=col.replace('_', ' '),
                values=df_parallel[col]
            ))
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=df_parallel['AttritionBinary'],
                     colorscale=[[0, '#4ECDC4'], [1, '#FF6B6B']],
                     showscale=True,
                     colorbar=dict(title="Attrition<br>(0=No, 1=Yes)")),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title="Multi-Dimensional Employee Analysis - Parallel Coordinates",
            height=600
        )
        
        return fig
    
    def create_heatmap_correlation_matrix(self, df):
        """Create enhanced correlation heatmap with clustering"""
        # Select numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Add binary attrition column
        numeric_df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create enhanced heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Enhanced Correlation Matrix - Employee Attributes",
            height=700,
            width=700
        )
        
        return fig
    
    def create_waterfall_chart(self, df):
        """Create waterfall chart showing attrition factors impact"""
        # Calculate base retention rate
        base_retention = len(df[df['Attrition'] == 'No']) / len(df) * 100
        
        # Calculate impact of different factors
        factors = []
        impacts = []
        
        # High job satisfaction impact
        high_satisfaction = df[df['JobSatisfaction'] >= 3]
        high_sat_retention = len(high_satisfaction[high_satisfaction['Attrition'] == 'No']) / len(high_satisfaction) * 100
        factors.append('High Job Satisfaction')
        impacts.append(high_sat_retention - base_retention)
        
        # Work-life balance impact
        good_wlb = df[df['WorkLifeBalance'] >= 3]
        wlb_retention = len(good_wlb[good_wlb['Attrition'] == 'No']) / len(good_wlb) * 100
        factors.append('Good Work-Life Balance')
        impacts.append(wlb_retention - base_retention)
        
        # Overtime impact
        no_overtime = df[df['OverTime'] == 'No']
        no_ot_retention = len(no_overtime[no_overtime['Attrition'] == 'No']) / len(no_overtime) * 100
        factors.append('No Overtime')
        impacts.append(no_ot_retention - base_retention)
        
        # High income impact
        high_income = df[df['MonthlyIncome'] > df['MonthlyIncome'].median()]
        high_inc_retention = len(high_income[high_income['Attrition'] == 'No']) / len(high_income) * 100
        factors.append('Above Median Income')
        impacts.append(high_inc_retention - base_retention)
        
        fig = go.Figure(go.Waterfall(
            name="Retention Impact",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(factors),
            x=["Base Retention"] + factors,
            textposition="outside",
            text=[f"{base_retention:.1f}%"] + [f"{impact:+.1f}%" for impact in impacts],
            y=[base_retention] + impacts,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Employee Retention Impact Analysis - Waterfall Chart",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_treemap_visualization(self, df):
        """Create treemap for hierarchical data visualization"""
        # Create hierarchical data
        df_tree = df.groupby(['Department', 'JobRole', 'Attrition']).size().reset_index(name='Count')
        
        fig = px.treemap(df_tree, 
                        path=['Department', 'JobRole', 'Attrition'], 
                        values='Count',
                        color='Count',
                        color_continuous_scale='RdYlBu',
                        title="Employee Distribution Treemap: Department → Job Role → Attrition")
        
        fig.update_layout(height=600)
        return fig
    
    def create_advanced_box_plots(self, df):
        """Create advanced box plots with statistical annotations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Monthly Income by Department', 'Age Distribution by Job Level',
                          'Years at Company by Education', 'Job Satisfaction by Attrition'],
            specs=[[{"type": "box"}, {"type": "box"}],
                   [{"type": "box"}, {"type": "violin"}]]
        )
        
        # Monthly Income by Department
        departments = df['Department'].unique()
        for dept in departments:
            dept_data = df[df['Department'] == dept]['MonthlyIncome']
            fig.add_trace(go.Box(y=dept_data, name=dept, showlegend=False), row=1, col=1)
        
        # Age by Job Level
        job_levels = sorted(df['JobLevel'].unique())
        for level in job_levels:
            level_data = df[df['JobLevel'] == level]['Age']
            fig.add_trace(go.Box(y=level_data, name=f'Level {level}', showlegend=False), row=1, col=2)
        
        # Years at Company by Education
        education_mapping = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
        for edu_level in sorted(df['Education'].unique()):
            edu_data = df[df['Education'] == edu_level]['YearsAtCompany']
            fig.add_trace(go.Box(y=edu_data, name=education_mapping.get(edu_level, str(edu_level)), 
                               showlegend=False), row=2, col=1)
        
        # Job Satisfaction by Attrition (Violin Plot)
        for attrition in ['No', 'Yes']:
            attr_data = df[df['Attrition'] == attrition]['JobSatisfaction']
            fig.add_trace(go.Violin(y=attr_data, name=f'Attrition: {attrition}', 
                                  side='positive' if attrition == 'No' else 'negative',
                                  showlegend=False), row=2, col=2)
        
        fig.update_layout(height=800, title_text="Advanced Statistical Analysis - Distribution Comparisons")
        return fig