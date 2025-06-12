import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RegressionComparison:
    def __init__(self, data, target_column='score', test_size=0.1, val_size=0.1, random_state=42):

        self.data = data
        self.target_column = target_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        # Separate features and target
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # First split: separate test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Second split: separate train and validation from remaining data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=self.val_size, random_state=self.random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data split completed:")
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Validation set: {self.X_val.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
    def initialize_models(self):
        """Initialize all regression models"""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=self.random_state),
            'Support Vector Regression': SVR(kernel='rbf'),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                         random_state=self.random_state, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20)
        }
        
    def calculate_metrics(self, y_true, y_pred):
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred)
        }
        
    def train_and_evaluate(self):
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Determine if model needs scaled features
            needs_scaling = name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net', 
                                   'Support Vector Regression', 'K-Nearest Neighbors', 'Neural Network']
            
            if needs_scaling:
                X_train_use = self.X_train_scaled
                X_val_use = self.X_val_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train
                X_val_use = self.X_val
                X_test_use = self.X_test
            
            # Train model
            model.fit(X_train_use, self.y_train)
            
            # Make predictions
            train_pred = model.predict(X_train_use)
            val_pred = model.predict(X_val_use)
            test_pred = model.predict(X_test_use)
            
            # Calculate metrics
            self.results[name] = {
                'train_metrics': self.calculate_metrics(self.y_train, train_pred),
                'val_metrics': self.calculate_metrics(self.y_val, val_pred),
                'test_metrics': self.calculate_metrics(self.y_test, test_pred),
                'model': model
            }
            
    def display_results(self):
        print("\n" + "="*80)
        print("REGRESSION MODEL COMPARISON RESULTS")
        print("="*80)
        
        # Create results DataFrame for easy viewing
        results_data = []
        for name, metrics in self.results.items():
            results_data.append({
                'Model': name,
                'Train R²': f"{metrics['train_metrics']['R²']:.4f}",
                'Val R²': f"{metrics['val_metrics']['R²']:.4f}",
                'Test R²': f"{metrics['test_metrics']['R²']:.4f}",
                'Train RMSE': f"{metrics['train_metrics']['RMSE']:.4f}",
                'Val RMSE': f"{metrics['val_metrics']['RMSE']:.4f}",
                'Test RMSE': f"{metrics['test_metrics']['RMSE']:.4f}",
            })
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        # Find best model based on validation R²
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['val_metrics']['R²'])
        
        print(f"\n🏆 Best Model (by Validation R²): {best_model_name}")
        print(f"   Validation R²: {self.results[best_model_name]['val_metrics']['R²']:.4f}")
        print(f"   Test R²: {self.results[best_model_name]['test_metrics']['R²']:.4f}")
        
    def plot_learning_curves(self):
        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, (name, model) in enumerate(self.models.items()):
            print(f"Generating learning curve for {name}...")
            
            # Determine if model needs scaled features
            needs_scaling = name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net', 
                                   'Support Vector Regression', 'K-Nearest Neighbors', 'Neural Network']
            
            if needs_scaling:
                X_use = np.vstack([self.X_train_scaled, self.X_val_scaled])
            else:
                X_use = pd.concat([self.X_train, self.X_val], axis=0).values
                
            y_use = np.concatenate([self.y_train, self.y_val])
            
            # Generate learning curve
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_use, y_use, 
                train_sizes=train_sizes, 
                cv=5, 
                scoring='neg_root_mean_squared_error',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Calculate means and stds
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # Convert negative RMSE back to positive RMSE
            train_mean = -train_mean
            val_mean = -val_mean
            train_std = np.abs(train_std)
            val_std = np.abs(val_std) 
            
            # Plot
            ax = axes[idx]
            ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('RMSE')
            ax.set_title(f'{name} - Learning Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add overfitting indicator
            final_gap = val_mean[-1] - train_mean[-1]
            if final_gap > 0.1:
                ax.text(0.05, 0.95, 'Potential Overfitting', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.5),
                       verticalalignment='top')
            elif final_gap < 0.05:
                ax.text(0.05, 0.95, 'Good Fit', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.5),
                       verticalalignment='top')
        
        # Hide empty subplots
        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.savefig('learning_curvess.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_validation_curves(self):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Models with hyperparameters to tune
        param_configs = [
            ('Ridge Regression', 'alpha', np.logspace(-4, 2, 20)),
            ('Lasso Regression', 'alpha', np.logspace(-4, 2, 20)),
            ('Decision Tree', 'max_depth', range(1, 21)),
            ('Random Forest', 'n_estimators', range(10, 201, 20)),
            ('Gradient Boosting', 'n_estimators', range(10, 201, 20)),
            ('K-Nearest Neighbors', 'n_neighbors', range(1, 31))
        ]
        
        for idx, (model_name, param_name, param_range) in enumerate(param_configs):
            if model_name not in self.models:
                continue
                
            print(f"Generating validation curve for {model_name} - {param_name}...")
            
            model = self.models[model_name]
            needs_scaling = model_name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net', 
                                         'Support Vector Regression', 'K-Nearest Neighbors', 'Neural Network']
            
            if needs_scaling:
                X_use = np.vstack([self.X_train_scaled, self.X_val_scaled])
            else:
                X_use = pd.concat([self.X_train, self.X_val], axis=0).values
                
            y_use = np.concatenate([self.y_train, self.y_val])
            
            train_scores, val_scores = validation_curve(
                model, X_use, y_use, 
                param_name=param_name, 
                param_range=param_range,
                cv=5, 
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            # Convert negative RMSE back to positive RMSE
            train_mean = -train_mean
            val_mean = -val_mean
            train_std = np.abs(train_std)
            val_std = np.abs(val_std) 
            
            ax = axes[idx]
            ax.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            ax.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('RMSE')
            ax.set_title(f'{model_name} - Validation Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if param_name == 'alpha':
                ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('validation_curvess.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_neural_network_training(self):
        if 'Neural Network' in self.results:
            model = self.results['Neural Network']['model']
            if hasattr(model, 'loss_curve_'):
                plt.figure(figsize=(10, 6))
                plt.plot(model.loss_curve_, 'b-', label='Training Loss', linewidth=2)
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Neural Network Training Loss Over Epochs')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add early stopping indicator if applicable
                if hasattr(model, 'n_iter_'):
                    plt.axvline(x=model.n_iter_, color='red', linestyle='--', 
                               label=f'Early Stopping (Epoch {model.n_iter_})')
                    plt.legend()
                        
                plt.savefig('neural_network_training_results.png', dpi=300, bbox_inches='tight')
                plt.show()
            else:
                print("Neural Network training history not available")
    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        
        # R² Score comparison
        train_r2 = [self.results[m]['train_metrics']['R²'] for m in models]
        val_r2 = [self.results[m]['val_metrics']['R²'] for m in models]
        test_r2 = [self.results[m]['test_metrics']['R²'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        axes[0,0].bar(x - width, train_r2, width, label='Train', alpha=0.8)
        axes[0,0].bar(x, val_r2, width, label='Validation', alpha=0.8)
        axes[0,0].bar(x + width, test_r2, width, label='Test', alpha=0.8)
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_title('R² Score Comparison')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(models, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # RMSE comparison
        train_rmse = [self.results[m]['train_metrics']['RMSE'] for m in models]
        val_rmse = [self.results[m]['val_metrics']['RMSE'] for m in models]
        test_rmse = [self.results[m]['test_metrics']['RMSE'] for m in models]
        
        axes[0,1].bar(x - width, train_rmse, width, label='Train', alpha=0.8)
        axes[0,1].bar(x, val_rmse, width, label='Validation', alpha=0.8)
        axes[0,1].bar(x + width, test_rmse, width, label='Test', alpha=0.8)
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].set_title('RMSE Comparison')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(models, rotation=45, ha='right')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Overfitting analysis (Train vs Validation R²)
        axes[1,0].scatter(train_r2, val_r2, alpha=0.7, s=100)
        for i, model in enumerate(models):
            axes[1,0].annotate(model, (train_r2[i], val_r2[i]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add diagonal line for reference
        min_r2 = min(min(train_r2), min(val_r2))
        max_r2 = max(max(train_r2), max(val_r2))
        axes[1,0].plot([min_r2, max_r2], [min_r2, max_r2], 'r--', alpha=0.5)
        axes[1,0].set_xlabel('Training R²')
        axes[1,0].set_ylabel('Validation R²')
        axes[1,0].set_title('Overfitting Analysis')
        axes[1,0].grid(True, alpha=0.3)
        
        # Model ranking by validation performance
        sorted_models = sorted(models, key=lambda x: self.results[x]['val_metrics']['R²'], reverse=True)
        val_scores = [self.results[m]['val_metrics']['R²'] for m in sorted_models]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_models)))
        bars = axes[1,1].barh(range(len(sorted_models)), val_scores, color=colors)
        axes[1,1].set_yticks(range(len(sorted_models)))
        axes[1,1].set_yticklabels(sorted_models)
        axes[1,1].set_xlabel('Validation R²')
        axes[1,1].set_title('Model Ranking (by Validation R²)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_best_model(self):
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['val_metrics']['R²'])
        return best_model_name, self.results[best_model_name]['model']
    
    def run_complete_analysis(self):
        print("Starting Regression Model Comparison...")
        self.prepare_data()
        self.initialize_models()
        self.train_and_evaluate()
        self.display_results()
        self.plot_results()
        self.plot_learning_curves()
        self.plot_neural_network_training()
        self.plot_validation_curves()
        return self.get_best_model()

def main():
    print("Loading dataset...")
    df = pd.read_csv('/WAVE/projects/CSEN-140-Sp25/team_10/prepoccesed_wave_forecast.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize and run regression comparison
    regressor = RegressionComparison(df, target_column='score')
    best_model_name, best_model = regressor.run_complete_analysis()
    
    return regressor, best_model_name, best_model

if __name__ == "__main__":
    regressor, best_model_name, best_model = main()