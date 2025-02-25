pipeline {
    agent any
    
    options {
        // Add timestamps to console output
        timestamps()
        // Set timeout for the entire pipeline
        timeout(time: 30, unit: 'MINUTES')
    }
    
    // Define environment variables
    environment {
        PYTHON_VERSION = '3.9'
        MODEL_ENV = 'churn-model-env'
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python virtual environment...'
                sh '''
                # Create a virtual environment if it doesn't exist
                if [ ! -d "${MODEL_ENV}" ]; then
                    python3 -m venv ${MODEL_ENV}
                fi
                
                # Activate the virtual environment and install dependencies
                . ${MODEL_ENV}/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }
        
        stage('Data Preparation') {
            steps {
                echo 'Checking data files...'
                sh '''
                . ${MODEL_ENV}/bin/activate
                
                # Check if data files exist, if not download sample data
                if [ ! -f "churn-bigml-80.csv" ] || [ ! -f "churn-bigml-20.csv" ]; then
                    echo "Data files not found, downloading sample data..."
                    # Create a Python script to generate sample data
                    cat > generate_sample_data.py << 'EOL'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
n_samples = 5000

data = {
    'State': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n_samples),
    'Account Length': np.random.randint(1, 200, n_samples),
    'Area Code': np.random.randint(100, 999, n_samples),
    'International Plan': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]),
    'Voice Mail Plan': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
    'Number Vmail Messages': np.random.randint(0, 50, n_samples),
    'Total Day Minutes': np.random.uniform(0, 400, n_samples),
    'Total Day Calls': np.random.randint(0, 200, n_samples),
    'Total Day Charge': np.random.uniform(0, 100, n_samples),
    'Total Eve Minutes': np.random.uniform(0, 400, n_samples),
    'Total Eve Calls': np.random.randint(0, 200, n_samples),
    'Total Eve Charge': np.random.uniform(0, 100, n_samples),
    'Total Night Minutes': np.random.uniform(0, 400, n_samples),
    'Total Night Calls': np.random.randint(0, 200, n_samples),
    'Total Night Charge': np.random.uniform(0, 100, n_samples),
    'Total Intl Minutes': np.random.uniform(0, 60, n_samples),
    'Total Intl Calls': np.random.randint(0, 20, n_samples),
    'Total Intl Charge': np.random.uniform(0, 20, n_samples),
    'Customer Service Calls': np.random.randint(0, 10, n_samples)
}

# Create churn based on some rules
total_minutes = data['Total Day Minutes'] + data['Total Eve Minutes'] + data['Total Night Minutes']
intl_plan = np.array([1 if x == 'yes' else 0 for x in data['International Plan']])
cs_calls = data['Customer Service Calls']

# Churn is more likely with high total minutes, international plan, and many customer service calls
churn_prob = 0.05 + 0.1 * (total_minutes > 300) + 0.1 * intl_plan + 0.1 * (cs_calls > 3)
data['Churn'] = np.random.binomial(1, churn_prob)
data['Churn'] = ['True' if x == 1 else 'False' for x in data['Churn']]

# Create DataFrame
df = pd.DataFrame(data)

# Split into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV
train_df.to_csv('churn-bigml-80.csv', index=False)
test_df.to_csv('churn-bigml-20.csv', index=False)
EOL
                    
                    # Run the script to generate data
                    python generate_sample_data.py
                fi
                
                # Verify data files exist
                if [ -f "churn-bigml-80.csv" ] && [ -f "churn-bigml-20.csv" ]; then
                    echo "Data files verified."
                    # Display data file sizes
                    ls -la churn-bigml-*.csv
                else
                    echo "Data files not found. Exiting."
                    exit 1
                fi
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                echo 'Training decision tree model...'
                sh '''
                . ${MODEL_ENV}/bin/activate
                
                # Run the training script
                python main.py --train_file churn-bigml-80.csv --test_file churn-bigml-20.csv --max_depth 5 --save
                '''
            }
        }
        
        stage('Evaluate Model') {
            steps {
                echo 'Evaluating model performance...'
                sh '''
                . ${MODEL_ENV}/bin/activate
                
                # Check if model file exists
                if [ -f "decision_tree_model.pkl" ]; then
                    echo "Model file found. Size: $(ls -la decision_tree_model.pkl | awk '{print $5}') bytes"
                else
                    echo "Model file not found. Training likely failed."
                    exit 1
                fi
                
                # Create a directory for reports if it doesn't exist
                mkdir -p reports
                '''
            }
            post {
                success {
                    echo 'Model evaluation completed successfully'
                    archiveArtifacts artifacts: 'decision_tree_model.pkl,reports/*.png', fingerprint: true
                }
            }
        }
        
        stage('Generate Report') {
            steps {
                echo 'Generating model performance report...'
                sh '''
                . ${MODEL_ENV}/bin/activate
                
                # Create a simple HTML report
                cat > create_report.py << 'EOL'
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

def create_report():
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Load model and test data
    model = joblib.load("decision_tree_model.pkl")
    test_df = pd.read_csv("churn-bigml-20.csv")
    
    # Separate features and target
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    
    # Encode categorical features
    for col in X_test.columns:
        if X_test[col].dtype == 'object':
            X_test[col] = X_test[col].astype('category').cat.codes
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Visualize the tree (limited depth for visibility)
    plt.figure(figsize=(20, 10))
    plot_tree(model, max_depth=3, feature_names=X_test.columns, filled=True, rounded=True)
    plt.savefig("reports/decision_tree.png", bbox_inches="tight")
    
    # Create HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Churn Prediction Model Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .metric-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .image-container {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Churn Prediction Model Report</h1>
            <p>Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Model Performance</h2>
            <div class="metrics">
                <div class="metric-box">
                    <h3>Accuracy</h3>
                    <p style="font-size: 24px; font-weight: bold;">{accuracy:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Precision</h3>
                    <p style="font-size: 24px; font-weight: bold;">{class_report['True']['precision']:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>Recall</h3>
                    <p style="font-size: 24px; font-weight: bold;">{class_report['True']['recall']:.4f}</p>
                </div>
                <div class="metric-box">
                    <h3>F1 Score</h3>
                    <p style="font-size: 24px; font-weight: bold;">{class_report['True']['f1-score']:.4f}</p>
                </div>
            </div>
            
            <h2>Feature Importance</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Importance</th>
                </tr>
                {''.join(f'<tr><td>{row.Feature}</td><td>{row.Importance:.4f}</td></tr>' for _, row in feature_importance.head(10).iterrows())}
            </table>
            
            <h2>Decision Tree Visualization</h2>
            <div class="image-container">
                <img src="decision_tree.png" alt="Decision Tree Visualization">
            </div>
            
            <h2>Confusion Matrix</h2>
            <div class="image-container">
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("reports/model_report.html", "w") as f:
        f.write(html)
    
    print("Report generated successfully: reports/model_report.html")

if __name__ == "__main__":
    create_report()
EOL
                
                # Run the report generation script
                python create_report.py
                '''
            }
            post {
                success {
                    echo 'Report generated successfully'
                    archiveArtifacts artifacts: 'reports/model_report.html,reports/*.png', fingerprint: true
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'reports',
                        reportFiles: 'model_report.html',
                        reportName: 'Model Performance Report',
                        reportTitles: 'Churn Model Report'
                    ])
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                echo 'Cleaning up workspace...'
                sh '''
                # Keep model and reports but clean temporary files
                if [ -d "${MODEL_ENV}" ]; then
                    echo "Preserved virtual environment for future builds"
                fi
                
                # Remove any temporary files
                rm -f generate_sample_data.py create_report.py 2>/dev/null || true
                
                echo "Workspace cleaned"
                '''
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
        always {
            echo 'Pipeline execution completed.'
        }
    }
}
