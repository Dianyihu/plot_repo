import numpy as np
from sklearn.ensemble import RandomForestRegressor
from plot_predicted import plot_prediction_results, two_feature_inter, two_feature_inter_plotly

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# Create features with different relationships
X = np.zeros((n_samples, 3))
X[:, 0] = np.random.normal(100, 10, n_samples)  # Temperature: normal distribution
X[:, 1] = np.random.uniform(5, 15, n_samples)   # Pressure: uniform distribution
X[:, 2] = np.random.exponential(5, n_samples)   # Flow rate: exponential distribution

# Create target with non-linear relationships
y = (0.3 * X[:, 0] + 
     2.0 * np.sin(X[:, 1]) + 
     0.5 * np.square(X[:, 2]) + 
     np.random.normal(0, 5, n_samples))  # Add some noise

# Train a random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Define feature names
feature_names = ['Temperature', 'Pressure', 'Flow_Rate']

# Plot the results
two_feature_inter_plotly(feature_names, model, X)