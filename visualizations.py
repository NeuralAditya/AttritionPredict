import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizations:
    def __init__(self, df):
        self.df = df
        self.color_palette = px.colors.qualitative.Set3
    
    def create_attrition_donut(self, df):
        """Create a donut chart for attrition distribution"""
        attrition_counts = df['Attrition'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=attrition_counts.index,
            values=attrition_counts.values,
            hole=0.4,
            marker_colors=['#FF6B6B', '#4ECDC4']
        )])
        
        fig.update_layout(
            title="Employee Attrition Distribution",
            annotations=[dict(text='Attrition', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig
    
    def create_attrition_by_department(self, df):
        """Create bar chart for attrition by department"""
        dept_attrition = df.groupby(['Department', 'Attrition']).size().unstack(fill_value=0)
        dept_attrition['Total'] = dept_attrition.sum(axis=1)
        dept_attrition['Attrition_Rate'] = (dept_attrition['Yes'] / dept_attrition['Total'] * 100).round(2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Stayed',
            x=dept_attrition.index,
            y=dept_attrition['No'],
            marker_color='#4ECDC4'
        ))
        
        fig.add_trace(go.Bar(
            name='Left',
            x=dept_attrition.index,
            y=dept_attrition['Yes'],
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title="Employee Attrition by Department",
            xaxis_title="Department",
            yaxis_title="Number of Employees",
            barmode='stack'
        )
        
        return fig
    
    def create_income_analysis(self, df):
        """Create box plot for income analysis by attrition"""
        fig = px.box(
            df, 
            x='Attrition', 
            y='MonthlyIncome',
            title="Monthly Income Distribution by Attrition Status",
            color='Attrition',
            color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
        )
        
        fig.update_layout(
            xaxis_title="Attrition Status",
            yaxis_title="Monthly Income ($)"
        )
        
        return fig
    
    def create_age_analysis(self, df):
        """Create age distribution analysis"""
        age_bins = [18, 25, 35, 45, 55, 65]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
        df_temp = df.copy()
        df_temp['AgeGroup'] = pd.cut(df_temp['Age'], bins=age_bins, labels=age_labels)
        
        age_attrition = df_temp.groupby(['AgeGroup', 'Attrition']).size().unstack(fill_value=0)
        age_attrition['Total'] = age_attrition.sum(axis=1)
        age_attrition['Attrition_Rate'] = (age_attrition['Yes'] / age_attrition['Total'] * 100).round(2)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                name='Stayed',
                x=age_attrition.index,
                y=age_attrition['No'],
                marker_color='#4ECDC4'
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Bar(
                name='Left',
                x=age_attrition.index,
                y=age_attrition['Yes'],
                marker_color='#FF6B6B'
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                name='Attrition Rate %',
                x=age_attrition.index,
                y=age_attrition['Attrition_Rate'],
                mode='lines+markers',
                marker_color='#45B7D1',
                line=dict(width=3)
            ),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Age Group")
        fig.update_yaxes(title_text="Number of Employees", secondary_y=False)
        fig.update_yaxes(title_text="Attrition Rate (%)", secondary_y=True)
        fig.update_layout(title="Attrition Analysis by Age Group", barmode='stack')
        
        return fig
    
    def create_gender_analysis(self, df):
        """Create gender analysis chart"""
        gender_attrition = df.groupby(['Gender', 'Attrition']).size().unstack(fill_value=0)
        gender_attrition['Total'] = gender_attrition.sum(axis=1)
        gender_attrition['Attrition_Rate'] = (gender_attrition['Yes'] / gender_attrition['Total'] * 100).round(2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Stayed',
            x=gender_attrition.index,
            y=gender_attrition['No'],
            marker_color='#4ECDC4'
        ))
        
        fig.add_trace(go.Bar(
            name='Left',
            x=gender_attrition.index,
            y=gender_attrition['Yes'],
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title="Employee Attrition by Gender",
            xaxis_title="Gender",
            yaxis_title="Number of Employees",
            barmode='stack'
        )
        
        return fig
    
    def create_education_analysis(self, df):
        """Create education level analysis"""
        education_mapping = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
        df_temp = df.copy()
        df_temp['EducationLevel'] = df_temp['Education'].map(education_mapping)
        
        edu_attrition = df_temp.groupby(['EducationLevel', 'Attrition']).size().unstack(fill_value=0)
        edu_attrition['Total'] = edu_attrition.sum(axis=1)
        edu_attrition['Attrition_Rate'] = (edu_attrition['Yes'] / edu_attrition['Total'] * 100).round(2)
        
        fig = px.bar(
            x=edu_attrition.index,
            y=edu_attrition['Attrition_Rate'],
            title="Attrition Rate by Education Level",
            labels={'x': 'Education Level', 'y': 'Attrition Rate (%)'},
            color=edu_attrition['Attrition_Rate'],
            color_continuous_scale='RdYlBu_r'
        )
        
        return fig
    
    def create_distance_analysis(self, df):
        """Create distance from home analysis"""
        distance_bins = [0, 5, 10, 20, 30]
        distance_labels = ['0-5 km', '6-10 km', '11-20 km', '20+ km']
        df_temp = df.copy()
        df_temp['DistanceGroup'] = pd.cut(df_temp['DistanceFromHome'], bins=distance_bins, labels=distance_labels)
        
        dist_attrition = df_temp.groupby(['DistanceGroup', 'Attrition']).size().unstack(fill_value=0)
        dist_attrition['Total'] = dist_attrition.sum(axis=1)
        dist_attrition['Attrition_Rate'] = (dist_attrition['Yes'] / dist_attrition['Total'] * 100).round(2)
        
        fig = px.line(
            x=dist_attrition.index,
            y=dist_attrition['Attrition_Rate'],
            title="Attrition Rate by Distance from Home",
            labels={'x': 'Distance Group', 'y': 'Attrition Rate (%)'},
            markers=True
        )
        
        return fig
    
    def create_job_role_analysis(self, df):
        """Create job role analysis"""
        role_attrition = df.groupby(['JobRole', 'Attrition']).size().unstack(fill_value=0)
        role_attrition['Total'] = role_attrition.sum(axis=1)
        role_attrition['Attrition_Rate'] = (role_attrition['Yes'] / role_attrition['Total'] * 100).round(2)
        role_attrition = role_attrition.sort_values('Attrition_Rate', ascending=True)
        
        fig = px.bar(
            x=role_attrition['Attrition_Rate'],
            y=role_attrition.index,
            title="Attrition Rate by Job Role",
            labels={'x': 'Attrition Rate (%)', 'y': 'Job Role'},
            orientation='h',
            color=role_attrition['Attrition_Rate'],
            color_continuous_scale='RdYlBu_r'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_job_level_analysis(self, df):
        """Create job level analysis"""
        level_attrition = df.groupby(['JobLevel', 'Attrition']).size().unstack(fill_value=0)
        level_attrition['Total'] = level_attrition.sum(axis=1)
        level_attrition['Attrition_Rate'] = (level_attrition['Yes'] / level_attrition['Total'] * 100).round(2)
        
        fig = px.bar(
            x=level_attrition.index,
            y=level_attrition['Attrition_Rate'],
            title="Attrition Rate by Job Level",
            labels={'x': 'Job Level', 'y': 'Attrition Rate (%)'},
            color=level_attrition['Attrition_Rate'],
            color_continuous_scale='RdYlBu_r'
        )
        
        return fig
    
    def create_overtime_analysis(self, df):
        """Create overtime analysis"""
        overtime_attrition = df.groupby(['OverTime', 'Attrition']).size().unstack(fill_value=0)
        overtime_attrition['Total'] = overtime_attrition.sum(axis=1)
        overtime_attrition['Attrition_Rate'] = (overtime_attrition['Yes'] / overtime_attrition['Total'] * 100).round(2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Stayed',
            x=overtime_attrition.index,
            y=overtime_attrition['No'],
            marker_color='#4ECDC4'
        ))
        
        fig.add_trace(go.Bar(
            name='Left',
            x=overtime_attrition.index,
            y=overtime_attrition['Yes'],
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title="Employee Attrition by Overtime Status",
            xaxis_title="Overtime",
            yaxis_title="Number of Employees",
            barmode='stack'
        )
        
        return fig
    
    def create_tenure_analysis(self, df):
        """Create tenure analysis"""
        fig = px.histogram(
            df,
            x='YearsAtCompany',
            color='Attrition',
            title="Employee Tenure Distribution by Attrition Status",
            labels={'YearsAtCompany': 'Years at Company', 'count': 'Number of Employees'},
            nbins=20,
            color_discrete_map={'Yes': '#FF6B6B', 'No': '#4ECDC4'}
        )
        
        return fig
    
    def create_job_satisfaction_analysis(self, df):
        """Create job satisfaction analysis"""
        satisfaction_mapping = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        df_temp = df.copy()
        df_temp['JobSatisfactionLevel'] = df_temp['JobSatisfaction'].map(satisfaction_mapping)
        
        sat_attrition = df_temp.groupby(['JobSatisfactionLevel', 'Attrition']).size().unstack(fill_value=0)
        sat_attrition['Total'] = sat_attrition.sum(axis=1)
        sat_attrition['Attrition_Rate'] = (sat_attrition['Yes'] / sat_attrition['Total'] * 100).round(2)
        
        fig = px.bar(
            x=sat_attrition.index,
            y=sat_attrition['Attrition_Rate'],
            title="Attrition Rate by Job Satisfaction Level",
            labels={'x': 'Job Satisfaction Level', 'y': 'Attrition Rate (%)'},
            color=sat_attrition['Attrition_Rate'],
            color_continuous_scale='RdYlBu_r'
        )
        
        return fig
    
    def create_worklife_balance_analysis(self, df):
        """Create work-life balance analysis"""
        worklife_mapping = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
        df_temp = df.copy()
        df_temp['WorkLifeBalanceLevel'] = df_temp['WorkLifeBalance'].map(worklife_mapping)
        
        wlb_attrition = df_temp.groupby(['WorkLifeBalanceLevel', 'Attrition']).size().unstack(fill_value=0)
        wlb_attrition['Total'] = wlb_attrition.sum(axis=1)
        wlb_attrition['Attrition_Rate'] = (wlb_attrition['Yes'] / wlb_attrition['Total'] * 100).round(2)
        
        fig = px.bar(
            x=wlb_attrition.index,
            y=wlb_attrition['Attrition_Rate'],
            title="Attrition Rate by Work-Life Balance Level",
            labels={'x': 'Work-Life Balance Level', 'y': 'Attrition Rate (%)'},
            color=wlb_attrition['Attrition_Rate'],
            color_continuous_scale='RdYlBu_r'
        )
        
        return fig
    
    def create_environment_satisfaction_analysis(self, df):
        """Create environment satisfaction analysis"""
        satisfaction_mapping = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        df_temp = df.copy()
        df_temp['EnvironmentSatisfactionLevel'] = df_temp['EnvironmentSatisfaction'].map(satisfaction_mapping)
        
        env_attrition = df_temp.groupby(['EnvironmentSatisfactionLevel', 'Attrition']).size().unstack(fill_value=0)
        env_attrition['Total'] = env_attrition.sum(axis=1)
        env_attrition['Attrition_Rate'] = (env_attrition['Yes'] / env_attrition['Total'] * 100).round(2)
        
        fig = px.bar(
            x=env_attrition.index,
            y=env_attrition['Attrition_Rate'],
            title="Attrition Rate by Environment Satisfaction Level",
            labels={'x': 'Environment Satisfaction Level', 'y': 'Attrition Rate (%)'},
            color=env_attrition['Attrition_Rate'],
            color_continuous_scale='RdYlBu_r'
        )
        
        return fig
    
    def create_job_involvement_analysis(self, df):
        """Create job involvement analysis"""
        involvement_mapping = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        df_temp = df.copy()
        df_temp['JobInvolvementLevel'] = df_temp['JobInvolvement'].map(involvement_mapping)
        
        inv_attrition = df_temp.groupby(['JobInvolvementLevel', 'Attrition']).size().unstack(fill_value=0)
        inv_attrition['Total'] = inv_attrition.sum(axis=1)
        inv_attrition['Attrition_Rate'] = (inv_attrition['Yes'] / inv_attrition['Total'] * 100).round(2)
        
        fig = px.bar(
            x=inv_attrition.index,
            y=inv_attrition['Attrition_Rate'],
            title="Attrition Rate by Job Involvement Level",
            labels={'x': 'Job Involvement Level', 'y': 'Attrition Rate (%)'},
            color=inv_attrition['Attrition_Rate'],
            color_continuous_scale='RdYlBu_r'
        )
        
        return fig
    
    def create_correlation_chart(self, correlations):
        """Create correlation bar chart"""
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            title="Feature Correlation with Attrition",
            labels={'x': 'Correlation Coefficient', 'y': 'Features'},
            orientation='h',
            color=correlations.values,
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix):
        """Create correlation heatmap"""
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            aspect="auto",
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(height=600)
        
        return fig
