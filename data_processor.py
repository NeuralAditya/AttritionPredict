import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def process_data(self):
        """Process and clean the HR dataset"""
        df = self.df.copy()
        
        # Create age groups
        df['AgeGroup'] = pd.cut(df['Age'], 
                               bins=[0, 25, 35, 45, 55, 100], 
                               labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Create tenure groups
        df['TenureGroup'] = pd.cut(df['YearsAtCompany'],
                                  bins=[0, 2, 5, 10, 50],
                                  labels=['0-2 years', '3-5 years', '6-10 years', '10+ years'])
        
        # Create income groups
        df['IncomeGroup'] = pd.cut(df['MonthlyIncome'],
                                  bins=[0, 3000, 6000, 10000, 20000],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Map education levels
        education_mapping = {
            1: 'Below College',
            2: 'College',
            3: 'Bachelor',
            4: 'Master',
            5: 'Doctor'
        }
        df['EducationLevel'] = df['Education'].map(education_mapping)
        
        # Map satisfaction levels
        satisfaction_mapping = {
            1: 'Low',
            2: 'Medium',
            3: 'High',
            4: 'Very High'
        }
        
        df['JobSatisfactionLevel'] = df['JobSatisfaction'].map(satisfaction_mapping)
        df['EnvironmentSatisfactionLevel'] = df['EnvironmentSatisfaction'].map(satisfaction_mapping)
        df['RelationshipSatisfactionLevel'] = df['RelationshipSatisfaction'].map(satisfaction_mapping)
        
        # Map work-life balance
        worklife_mapping = {
            1: 'Bad',
            2: 'Good',
            3: 'Better',
            4: 'Best'
        }
        df['WorkLifeBalanceLevel'] = df['WorkLifeBalance'].map(worklife_mapping)
        
        # Map job involvement
        involvement_mapping = {
            1: 'Low',
            2: 'Medium',
            3: 'High',
            4: 'Very High'
        }
        df['JobInvolvementLevel'] = df['JobInvolvement'].map(involvement_mapping)
        
        # Map performance rating
        performance_mapping = {
            1: 'Low',
            2: 'Good',
            3: 'Excellent',
            4: 'Outstanding'
        }
        df['PerformanceRatingLevel'] = df['PerformanceRating'].map(performance_mapping)
        
        return df
    
    def apply_filters(self, df, department, job_role, age_group, education, gender):
        """Apply filters to the dataframe"""
        filtered_df = df.copy()
        
        if department != 'All':
            filtered_df = filtered_df[filtered_df['Department'] == department]
        
        if job_role != 'All':
            filtered_df = filtered_df[filtered_df['JobRole'] == job_role]
        
        if age_group != 'All':
            if age_group == '18-25':
                filtered_df = filtered_df[filtered_df['Age'] <= 25]
            elif age_group == '26-35':
                filtered_df = filtered_df[(filtered_df['Age'] >= 26) & (filtered_df['Age'] <= 35)]
            elif age_group == '36-45':
                filtered_df = filtered_df[(filtered_df['Age'] >= 36) & (filtered_df['Age'] <= 45)]
            elif age_group == '46-55':
                filtered_df = filtered_df[(filtered_df['Age'] >= 46) & (filtered_df['Age'] <= 55)]
            elif age_group == '55+':
                filtered_df = filtered_df[filtered_df['Age'] > 55]
        
        if education != 'All':
            filtered_df = filtered_df[filtered_df['Education'] == education]
        
        if gender != 'All':
            filtered_df = filtered_df[filtered_df['Gender'] == gender]
        
        return filtered_df
    
    def get_attrition_rate(self, df, group_by_column):
        """Calculate attrition rate by a specific column"""
        if group_by_column not in df.columns:
            return pd.DataFrame()
        
        result = df.groupby(group_by_column).agg({
            'Attrition': ['count', lambda x: (x == 'Yes').sum()]
        }).round(2)
        
        result.columns = ['Total', 'Attrition_Count']
        result['Attrition_Rate'] = (result['Attrition_Count'] / result['Total'] * 100).round(2)
        result = result.reset_index()
        
        return result
    
    def get_summary_statistics(self, df):
        """Get summary statistics for the dataset"""
        stats = {
            'total_employees': len(df),
            'attrition_count': len(df[df['Attrition'] == 'Yes']),
            'attrition_rate': len(df[df['Attrition'] == 'Yes']) / len(df) * 100 if len(df) > 0 else 0,
            'avg_age': df['Age'].mean(),
            'avg_monthly_income': df['MonthlyIncome'].mean(),
            'avg_years_at_company': df['YearsAtCompany'].mean(),
            'departments': df['Department'].nunique(),
            'job_roles': df['JobRole'].nunique()
        }
        return stats
