import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import pandas as pd
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)

# Generate reproducible synthetic dataset for gradient boosting regression


def generate_gb_regression_data(n_samples=1000, n_features=10, n_informative=8,
                                noise=0.1, random_state=42):
    """
    Generate synthetic dataset with nonlinear relationships and interactions
    suitable for gradient boosting regression.
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )

    # Add nonlinear transformations and interactions (fixed to avoid NaN values)
    X[:, 0] = np.sin(X[:, 0]) * 2  # Nonlinear transformation

    # Safe power transformation - use absolute value and restore sign
    X[:, 1] = np.sign(X[:, 1]) * (np.abs(X[:, 1]) ** 1.5)

    # Safe log transformation - use log1p and ensure positive values
    X[:, 2] = np.log1p(np.abs(X[:, 2])) * np.sign(X[:, 2])

    # Add feature interactions
    y = y + 3 * X[:, 3] * X[:, 4] - 2 * X[:, 5] * np.sin(X[:, 6])

    # Add some categorical-like structure through thresholding
    X[:, 7] = (X[:, 7] > 0).astype(float)

    feature_names = [f'Feature_{i+1}' for i in range(n_features)]

    # Ensure no NaN values
    assert not np.any(np.isnan(X)), "X contains NaN values"
    assert not np.any(np.isnan(y)), "y contains NaN values"

    return X, y, feature_names


# Generate dataset with complex relationships
n_features = 12
n_informative = 9
X, y, feature_names = generate_gb_regression_data(
    n_samples=1000,
    n_features=n_features,
    n_informative=n_informative,
    noise=2.0
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Number of features: {n_features}")
print(f"Number of informative features: {n_informative}")
print(f"X contains NaN: {np.any(np.isnan(X))}")
print(f"y contains NaN: {np.any(np.isnan(y))}")

# Initialize and train Gradient Boosting Regressor
print("\n" + "="*60)
print("GRADIENT BOOSTING REGRESSOR TRAINING")
print("="*60)

# Basic Gradient Boosting model
gb_basic = GradientBoostingRegressor(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Shrinks the contribution of each tree
    max_depth=3,           # Maximum depth of individual trees
    min_samples_split=2,   # Minimum samples required to split a node
    min_samples_leaf=1,    # Minimum samples required at a leaf node
    subsample=1.0,         # Fraction of samples used for fitting
    random_state=42,
    loss='squared_error'   # Loss function to optimize
)

gb_basic.fit(X_train, y_train)

# Make predictions
y_train_pred = gb_basic.predict(X_train)
y_test_pred = gb_basic.predict(X_test)

# Calculate performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"Training Performance:")
print(f"  MSE: {train_mse:.3f}, R²: {train_r2:.3f}, MAE: {train_mae:.3f}")
print(f"Test Performance:")
print(f"  MSE: {test_mse:.3f}, R²: {test_r2:.3f}, MAE: {test_mae:.3f}")

# Feature Importance Analysis
feature_importances = gb_basic.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print(f"\nTop 5 Most Important Features:")
for i, row in feature_importance_df.head().iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Visualization 1: Actual vs Predicted values
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [
         y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Gradient Boosting: Actual vs Predicted\nTest R² = {test_r2:.3f}')
plt.grid(True, alpha=0.3)

# Visualization 2: Feature Importance
plt.subplot(1, 3, 2)
sns.barplot(data=feature_importance_df, x='Importance',
            y='Feature', palette='viridis')
plt.title('Feature Importance Scores')
plt.xlabel('Importance')

# Visualization 3: Training history (loss curve)
plt.subplot(1, 3, 3)
train_score = gb_basic.train_score_  # Training loss at each iteration
plt.plot(range(1, len(train_score) + 1), train_score, 'b-', linewidth=2)
plt.xlabel('Number of Boosting Iterations')
plt.ylabel('Training Loss (MSE)')
plt.title('Training Loss Convergence')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare different numbers of estimators
print("\n" + "="*60)
print("COMPARING DIFFERENT NUMBERS OF TREES")
print("="*60)

n_estimators_list = [50, 100, 200, 500]
estimator_results = []

for n_est in n_estimators_list:
    gb_temp = GradientBoostingRegressor(
        n_estimators=n_est,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    gb_temp.fit(X_train, y_train)
    y_pred_temp = gb_temp.predict(X_test)
    test_r2_temp = r2_score(y_test, y_pred_temp)
    train_r2_temp = r2_score(y_train, gb_temp.predict(X_train))

    estimator_results.append({
        'n_estimators': n_est,
        'train_r2': train_r2_temp,
        'test_r2': test_r2_temp,
        'gap': train_r2_temp - test_r2_temp  # Overfitting measure
    })

    print(f"Trees: {n_est:3d} | Train R²: {train_r2_temp:.3f} | "
          f"Test R²: {test_r2_temp:.3f} | Gap: {train_r2_temp - test_r2_temp:.3f}")

# Compare different learning rates
print("\n" + "="*60)
print("COMPARING DIFFERENT LEARNING RATES")
print("="*60)

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
lr_results = []

for lr in learning_rates:
    gb_lr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )

    gb_lr.fit(X_train, y_train)
    y_pred_lr = gb_lr.predict(X_test)
    test_r2_lr = r2_score(y_test, y_pred_lr)

    lr_results.append({
        'learning_rate': lr,
        'test_r2': test_r2_lr
    })

    print(f"Learning Rate: {lr:5.2f} | Test R²: {test_r2_lr:.3f}")

# Plot comparison of different parameters
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
n_est_vals = [result['n_estimators'] for result in estimator_results]
train_r2_vals = [result['train_r2'] for result in estimator_results]
test_r2_vals = [result['test_r2'] for result in estimator_results]

plt.plot(n_est_vals, train_r2_vals, 'o-', label='Train R²', linewidth=2)
plt.plot(n_est_vals, test_r2_vals, 's-', label='Test R²', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('R² Score')
plt.title('Performance vs Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
lr_vals = [result['learning_rate'] for result in lr_results]
test_r2_lr_vals = [result['test_r2'] for result in lr_results]

plt.semilogx(lr_vals, test_r2_lr_vals, 'o-', linewidth=2)
plt.xlabel('Learning Rate')
plt.ylabel('Test R² Score')
plt.title('Performance vs Learning Rate')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Hyperparameter tuning with GridSearchCV
print("\n" + "="*60)
print("HYPERPARAMETER TUNING WITH GRID SEARCH")
print("="*60)

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [2, 3, 4],
    'subsample': [0.8, 1.0]
}

# Use smaller subset for faster grid search
X_train_sub, _, y_train_sub, _ = train_test_split(
    X_train, y_train, test_size=0.7, random_state=42
)

gb_grid = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    gb_grid, param_grid, cv=3, scoring='r2',
    n_jobs=-1, verbose=1
)

print("Performing grid search... (this may take a moment)")
grid_search.fit(X_train_sub, y_train_sub)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation R²: {grid_search.best_score_:.3f}")

# Train final model with best parameters
best_gb = grid_search.best_estimator_
best_gb.fit(X_train, y_train)
y_test_pred_best = best_gb.predict(X_test)
best_test_r2 = r2_score(y_test, y_test_pred_best)

print(f"Test R² with tuned model: {best_test_r2:.3f}")

# Compare with basic model
improvement = best_test_r2 - test_r2
print(f"Improvement from tuning: {improvement:.3f}")

# Permutation importance for more robust feature importance
print("\n" + "="*60)
print("PERMUTATION FEATURE IMPORTANCE")
print("="*60)

perm_importance = permutation_importance(
    best_gb, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("Top features by permutation importance:")
for i, row in perm_importance_df.head().iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")

# Compare different loss functions
print("\n" + "="*60)
print("COMPARING DIFFERENT LOSS FUNCTIONS")
print("="*60)

loss_functions = ['squared_error', 'absolute_error', 'huber']
loss_results = []

for loss in loss_functions:
    gb_loss = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        loss=loss,
        random_state=42
    )

    gb_loss.fit(X_train, y_train)
    y_pred_loss = gb_loss.predict(X_test)
    test_r2_loss = r2_score(y_test, y_pred_loss)
    test_mae_loss = mean_absolute_error(y_test, y_pred_loss)

    loss_results.append({
        'loss': loss,
        'test_r2': test_r2_loss,
        'test_mae': test_mae_loss
    })

    print(
        f"Loss: {loss:15} | Test R²: {test_r2_loss:.3f} | Test MAE: {test_mae_loss:.3f}")

# Final comparison visualization
plt.figure(figsize=(15, 5))

# Plot 1: Compare basic vs tuned model
plt.subplot(1, 3, 1)
models = ['Basic GB', 'Tuned GB']
r2_scores = [test_r2, best_test_r2]

bars = plt.bar(models, r2_scores, color=['lightblue', 'lightgreen'], alpha=0.7)
plt.ylabel('Test R² Score')
plt.title('Basic vs Tuned Gradient Boosting')
plt.grid(True, alpha=0.3)

for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom')

# Plot 2: Compare feature importance methods
plt.subplot(1, 3, 2)
top_n = 6
top_features = feature_importance_df.head(top_n)['Feature'].values

gb_importances = []
perm_importances = []

for feature in top_features:
    gb_idx = feature_importance_df[feature_importance_df['Feature']
                                   == feature].index[0]
    perm_idx = perm_importance_df[perm_importance_df['Feature']
                                  == feature].index[0]

    gb_importances.append(feature_importance_df.loc[gb_idx, 'Importance'])
    perm_importances.append(perm_importance_df.loc[perm_idx, 'Importance'])

x_pos = np.arange(len(top_features))
width = 0.35

plt.bar(x_pos - width/2, gb_importances, width,
        label='GB Importance', alpha=0.7)
plt.bar(x_pos + width/2, perm_importances, width,
        label='Permutation Importance', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance Comparison')
plt.xticks(x_pos, [f'F{i+1}' for i in range(top_n)], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Compare loss functions
plt.subplot(1, 3, 3)
loss_names = [result['loss'] for result in loss_results]
loss_r2_scores = [result['test_r2'] for result in loss_results]

plt.bar(loss_names, loss_r2_scores, color=[
        'skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
plt.ylabel('Test R² Score')
plt.title('Performance by Loss Function')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

for i, score in enumerate(loss_r2_scores):
    plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print(f"\nKey Insights:")
print(
    f"- Gradient Boosting achieved test R² of {best_test_r2:.3f} with proper tuning")
print(
    f"- Hyperparameter tuning improved performance by {improvement:.3f} in R²")
print(f"- Optimal number of trees is typically between 100-200 for this dataset")
print(f"- Learning rate of 0.1 provided good balance between speed and performance")
print(f"- Feature importance helps identify the most predictive variables")
print(f"- Permutation importance provides more robust feature ranking")
print(f"- Different loss functions offer trade-offs between robustness and performance")
