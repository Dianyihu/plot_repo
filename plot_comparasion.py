import plotly.graph_objects as go
import numpy as np
import pandas as pd

def create_split_violin_plot(df, days, title="Distribution by Day and Group", 
                           y_label="Value", group_col="smoker", value_col="total_bill",
                           pos_group="Yes", neg_group="No",
                           pos_color='rgb(65, 105, 225)', neg_color='rgb(211, 211, 211)',
                           width=1200, height=800):
    """
    Create a split violin plot comparing two groups across different days.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the data
    days : list
        List of days to plot
    title : str
        Title of the plot
    y_label : str
        Label for y-axis
    group_col : str
        Column name for the grouping variable
    value_col : str
        Column name for the values to plot
    pos_group : str
        Name of the group to plot on positive side
    neg_group : str
        Name of the group to plot on negative side
    pos_color : str
        Color for positive side group
    neg_color : str
        Color for negative side group
    width : int
        Width of the figure
    height : int
        Height of the figure
    """
    
    # Create figure
    fig = go.Figure()
    
    # Calculate max y value for annotation positioning
    max_y = df[value_col].max()
    
    # Add violin plots for each day
    for day in days:
        # Add violin for positive side group
        pos_data = df[(df['day'] == day) & (df[group_col] == pos_group)][value_col]
        fig.add_trace(go.Violin(
            x=[day] * len(pos_data),
            y=pos_data,
            name=pos_group,
            side='positive',
            line_color=pos_color,
            showlegend=True if day == days[0] else False,
            box_visible=True,
            meanline_visible=True,
            points=False,
            box=dict(
                fillcolor='rgba(255,255,255,0)',
                line_color=pos_color,
                width=0.6,
                line=dict(width=2)
            ),
            meanline=dict(visible=True, color=pos_color)
        ))
        
        # Add violin for negative side group
        neg_data = df[(df['day'] == day) & (df[group_col] == neg_group)][value_col]
        fig.add_trace(go.Violin(
            x=[day] * len(neg_data),
            y=neg_data,
            name=neg_group,
            side='negative',
            line_color=neg_color,
            showlegend=True if day == days[0] else False,
            box_visible=True,
            meanline_visible=True,
            points=False,
            box=dict(
                fillcolor='rgba(255,255,255,0)',
                line_color=neg_color,
                width=0.6,
                line=dict(width=2)
            ),
            meanline=dict(visible=True, color=neg_color)
        ))
    
    # Add annotations for statistics
    annotations = []
    y_spacing = 10
    
    for day in days:
        # Stats for positive side group
        pos_data = df[(df['day'] == day) & (df[group_col] == pos_group)][value_col]
        pos_stats = {
            'n': len(pos_data),
            'mean': np.mean(pos_data),
            'median': np.median(pos_data),
            'std': np.std(pos_data)
        }
        
        # Stats for negative side group
        neg_data = df[(df['day'] == day) & (df[group_col] == neg_group)][value_col]
        neg_stats = {
            'n': len(neg_data),
            'mean': np.mean(neg_data),
            'median': np.median(neg_data),
            'std': np.std(neg_data)
        }
        
        # Add annotation for positive side
        annotations.append(dict(
            x=day,
            y=max_y + y_spacing,
            xanchor='left',
            yanchor='bottom',
            xshift=5,
            text=f"n={pos_stats['n']}<br>μ={pos_stats['mean']:.1f}<br>m={pos_stats['median']:.1f}<br>σ={pos_stats['std']:.1f}",
            showarrow=False,
            font=dict(size=14, color=pos_color),
            align='left'
        ))
        
        # Add annotation for negative side
        annotations.append(dict(
            x=day,
            y=max_y + y_spacing,
            xanchor='right',
            yanchor='bottom',
            xshift=-5,
            text=f"n={neg_stats['n']}<br>μ={neg_stats['mean']:.1f}<br>m={neg_stats['median']:.1f}<br>σ={neg_stats['std']:.1f}",
            showarrow=False,
            font=dict(size=14, color=neg_color),
            align='right'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24)
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(size=18)
            ),
            gridcolor='lightgray',
            gridwidth=0.5,
            range=[df[value_col].min() - 5, max_y + 25]
        ),
        xaxis=dict(
            title=dict(
                text='Day',
                font=dict(size=18)
            ),
            tickfont=dict(size=16),
            categoryarray=days,
            categoryorder='array',
            showgrid=False,
            range=[-0.5, len(days)-0.5]
        ),
        violinmode='overlay',
        violingap=0,
        showlegend=True,
        plot_bgcolor='white',
        margin=dict(l=100, r=200, t=120, b=100),
        width=width,
        height=height,
        legend=dict(
            title=dict(
                text=group_col.title(),
                font=dict(size=16)
            ),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        annotations=annotations
    )
    
    return fig

# Generate sample data
np.random.seed(42)
days = ['Thur', 'Fri',]
data = []

# Generate random data for each day
for day in days:
    # Simulate different patterns for smokers and non-smokers
    smoker_data = np.random.normal(25, 10, 100)
    non_smoker_data = np.random.normal(20, 8, 150)
    
    # Add data for smokers
    data.extend([{
        'day': day,
        'total_bill': v,
        'smoker': 'Yes'
    } for v in smoker_data])
    
    # Add data for non-smokers
    data.extend([{
        'day': day,
        'total_bill': v,
        'smoker': 'No'
    } for v in non_smoker_data])

# Create DataFrame
df = pd.DataFrame(data)

# Create and show the plot
fig = create_split_violin_plot(
    df, 
    days,
    title="Total Bill Distribution by Day and Smoking Status",
    y_label="Total Bill"
)

# Show the plot
fig.show()

# Save as PNG with high resolution
fig.write_image("violin_plot.png", scale=2)
