"""
Custom risk meter component
"""

import streamlit as st
import plotly.graph_objects as go

def create_risk_meter(probability, title="Recurrence Risk"):
    """
    Create a gauge chart for risk visualization
    
    Args:
        probability: Risk probability (0-1)
        title: Chart title
    """
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "#E74C3C"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#FF6B6B", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#2ECC71'},
                {'range': [30, 60], 'color': '#F39C12'},
                {'range': [60, 100], 'color': '#E74C3C'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig