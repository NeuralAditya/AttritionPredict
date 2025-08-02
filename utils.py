import streamlit as st
import plotly.graph_objects as go
import io
import pandas as pd

class Utils:
    def __init__(self):
        pass
    
    def export_chart(self, fig, filename):
        """Export chart as HTML file"""
        try:
            # Create HTML string
            html_string = fig.to_html(include_plotlyjs='cdn')
            
            # Create download button
            st.download_button(
                label="Download Chart",
                data=html_string,
                file_name=filename,
                mime="text/html"
            )
            
            st.success(f"Chart exported as {filename}")
            
        except Exception as e:
            st.error(f"Error exporting chart: {str(e)}")
    
    def export_data(self, df, filename):
        """Export dataframe as CSV"""
        try:
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download Data",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
            st.success(f"Data exported as {filename}")
            
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
    
    def format_number(self, number, format_type="standard"):
        """Format numbers for display"""
        if format_type == "percentage":
            return f"{number:.1f}%"
        elif format_type == "currency":
            return f"${number:,.0f}"
        elif format_type == "standard":
            return f"{number:,.0f}"
        else:
            return str(number)
    
    def create_summary_card(self, title, value, delta=None, delta_color=None):
        """Create a summary card display"""
        if delta is not None and delta_color is not None:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color
            )
        elif delta is not None:
            st.metric(
                label=title,
                value=value,
                delta=delta
            )
        else:
            st.metric(label=title, value=value)
    
    def validate_data(self, df):
        """Validate the uploaded dataset"""
        required_columns = [
            'Age', 'Attrition', 'Department', 'DistanceFromHome',
            'Education', 'Gender', 'JobRole', 'JobSatisfaction',
            'MonthlyIncome', 'OverTime', 'YearsAtCompany'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check if Attrition column has correct values
        attrition_values = df['Attrition'].unique()
        if not all(val in ['Yes', 'No'] for val in attrition_values):
            return False, "Attrition column must contain only 'Yes' and 'No' values"
        
        return True, "Data validation successful"
    
    def get_data_info(self, df):
        """Get basic information about the dataset"""
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'numeric_columns': len(df.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
        return info
    
    def create_insights_text(self, df):
        """Generate insights text based on data analysis"""
        total_employees = len(df)
        attrition_count = len(df[df['Attrition'] == 'Yes'])
        attrition_rate = (attrition_count / total_employees) * 100
        
        # Department with highest attrition
        dept_attrition = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
        highest_dept = dept_attrition.idxmax()
        highest_dept_rate = dept_attrition.max()
        
        # Job role with highest attrition
        role_attrition = df.groupby('JobRole')['Attrition'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
        highest_role = role_attrition.idxmax()
        highest_role_rate = role_attrition.max()
        
        insights = f"""
        ## Key Insights
        
        📊 **Overall Attrition Rate**: {attrition_rate:.1f}% ({attrition_count} out of {total_employees} employees)
        
        🏢 **Department Analysis**: {highest_dept} has the highest attrition rate at {highest_dept_rate:.1f}%
        
        👔 **Job Role Analysis**: {highest_role} role shows the highest attrition rate at {highest_role_rate:.1f}%
        
        💰 **Income Impact**: Employees who left had an average monthly income of ${df[df['Attrition'] == 'Yes']['MonthlyIncome'].mean():,.0f}
        
        ⏰ **Tenure Pattern**: Average tenure of employees who left is {df[df['Attrition'] == 'Yes']['YearsAtCompany'].mean():.1f} years
        """
        
        return insights
