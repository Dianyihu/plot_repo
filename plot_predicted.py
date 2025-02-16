import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence  # Updated import
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd

def plot_prediction_results(y_true, y_pred, features, feature_names, model, X):
    """
    Plot comprehensive model prediction results
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    features : array-like
        Feature matrix
    feature_names : list
        Names of the features
    model : sklearn model object
        Trained model
    X : array-like
        Training data
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Actual vs Predicted Plot
    ax1 = plt.subplot(2, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    
    # 2. Feature Importance Plot
    ax2 = plt.subplot(2, 2, 2)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_)
    
    importance_df = pd.DataFrame({
        'features': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    sns.barplot(y='features', x='importance', data=importance_df)  # Changed from barh to barplot
    plt.title('Feature Importance')
    
    # 3. Partial Dependence Plots
    
    # Single feature partial dependence
    fig2 = plt.figure(figsize=(15, 5))
    for i, feature in enumerate(range(n_features)):
        ax = plt.subplot(1, n_features, i+1)
        display = PartialDependenceDisplay.from_estimator(
            model, 
            X, 
            [feature],
            feature_names=feature_names,
            ax=ax
        )
        plt.ylabel('Partial dependence')
        plt.title(f'Response of {feature_names[i]}')

    fig2.show()
    
def two_feature_inter(y_true, y_pred, features, feature_names, model, X):
    # Two-feature interaction partial dependence
    n_features = len(feature_names)

    for i in range(n_features-1):
        fig = plt.figure(figsize=(15, 5))

        features = [i, i+1]  # Get pairs of consecutive features
        
        # Calculate partial dependence values
        pdp = partial_dependence(
            model, 
            X, 
            features,
            feature_names=feature_names,
            kind='average',
            grid_resolution=50
        )
        
        XX, YY = np.meshgrid(pdp["grid_values"][0], pdp["grid_values"][1])
        Z = pdp.average[0].T

        ax = fig.add_subplot(projection="3d")
        fig.add_axes(ax)

        surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor="k")
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        # fig.suptitle(
        #     "PD of number of bike rentals on\nthe temperature and humidity GBDT model",
        #     fontsize=16,
        # )
        # pretty init view
        ax.view_init(elev=22, azim=122)
        clb = plt.colorbar(surf, pad=0.08, shrink=0.6, aspect=10)
        # clb.ax.set_title("Partial\ndependence")
        plt.show()


def two_feature_inter_plotly(feature_names, model, X):
    """Plot two-feature interaction using plotly for interactive 3D visualization"""
    n_features = len(feature_names)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=n_features-1,
        specs=[[{'type': 'surface'} for _ in range(n_features-1)]],
        subplot_titles=[f'Interaction: {feature_names[i]} vs {feature_names[i+1]}' 
                       for i in range(n_features-1)],
        horizontal_spacing=0.05
    )
    
    # Store Z values to calculate global color scale
    all_z_values = []
    
    # First pass: collect all Z values
    for i in range(n_features-1):
        features = [i, i+1]
        pdp = partial_dependence(model, X, features, grid_resolution=50)
        Z = pdp.average[0].T
        all_z_values.append(Z)
    
    # Calculate global min and max for consistent color scale
    z_min = min(np.min(z) for z in all_z_values)
    z_max = max(np.max(z) for z in all_z_values)
    
    # Second pass: create plots with consistent color scale
    for i in range(n_features-1):
        features = [i, i+1]
        pdp = partial_dependence(model, X, features, grid_resolution=50)
        XX, YY = np.meshgrid(pdp["grid_values"][0], pdp["grid_values"][1])
        Z = all_z_values[i]
        
        # Add surface plot to subplot
        fig.add_trace(
            go.Surface(
                x=XX, y=YY, z=Z,
                colorscale='Viridis',
                cmin=z_min,
                cmax=z_max,
                colorbar=dict(
                    title='Partial Dependence',
                    titleside='right',
                    x=1.0,  # Fixed position for all colorbars
                    len=0.8,
                )
            ),
            row=1, col=i+1
        )
        
        # Update scene for this subplot
        fig.update_layout({
            f'scene{i+1}': dict(
                xaxis=dict(title=feature_names[i]),
                yaxis=dict(title=feature_names[i+1]),
                zaxis=dict(title='Partial Dependence'),
                aspectmode='cube',  # Force same size for all plots
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        })
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1000,  # Fixed width
        showlegend=False,
        margin=dict(t=60, l=0, r=100)  # Adjust margins
    )
    
    fig.show()

# Example usage:
"""
# Assuming you have:
model = your_trained_model
X = your_feature_matrix
y_true = actual_values
y_pred = predicted_values
feature_names = ['Feature1', 'Feature2', 'Feature3']

plot_prediction_results(y_true, y_pred, X, feature_names, model, X)
"""