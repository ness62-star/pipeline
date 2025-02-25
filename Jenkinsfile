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
                bat '''
                @echo off
                rem Create a virtual environment if it doesn't exist
                if not exist "%MODEL_ENV%" (
                    python -m venv %MODEL_ENV%
                )
                
                rem Activate the virtual environment and install dependencies
                call %MODEL_ENV%\\Scripts\\activate.bat
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }
        
        stage('Data Preparation') {
            steps {
                echo 'Checking data files...'
                bat '''
                @echo off
                call %MODEL_ENV%\\Scripts\\activate.bat
                
                rem Check if data files exist, if not download sample data
                if not exist "churn-bigml-80.csv" (
                    echo Data files not found, generating sample data...
                    
                    rem Create a Python script to generate sample data
                    echo import pandas as pd > generate_sample_data.py
                    echo import numpy as np >> generate_sample_data.py
                    echo from sklearn.model_selection import train_test_split >> generate_sample_data.py
                    echo. >> generate_sample_data.py
                    echo # Generate sample data >> generate_sample_data.py
                    echo np.random.seed(42) >> generate_sample_data.py
                    echo n_samples = 5000 >> generate_sample_data.py
                    echo. >> generate_sample_data.py
                    echo data = { >> generate_sample_data.py
                    echo     'State': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n_samples), >> generate_sample_data.py
                    echo     'Account Length': np.random.randint(1, 200, n_samples), >> generate_sample_data.py
                    echo     'Area Code': np.random.randint(100, 999, n_samples), >> generate_sample_data.py
                    echo     'International Plan': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]), >> generate_sample_data.py
                    echo     'Voice Mail Plan': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]), >> generate_sample_data.py
                    echo     'Number Vmail Messages': np.random.randint(0, 50, n_samples), >> generate_sample_data.py
                    echo     'Total Day Minutes': np.random.uniform(0, 400, n_samples), >> generate_sample_data.py
                    echo     'Total Day Calls': np.random.randint(0, 200, n_samples), >> generate_sample_data.py
                    echo     'Total Day Charge': np.random.uniform(0, 100, n_samples), >> generate_sample_data.py
                    echo     'Total Eve Minutes': np.random.uniform(0, 400, n_samples), >> generate_sample_data.py
                    echo     'Total Eve Calls': np.random.randint(0, 200, n_samples), >> generate_sample_data.py
                    echo     'Total Eve Charge': np.random.uniform(0, 100, n_samples), >> generate_sample_data.py
                    echo     'Total Night Minutes': np.random.uniform(0, 400, n_samples), >> generate_sample_data.py
                    echo     'Total Night Calls': np.random.randint(0, 200, n_samples), >> generate_sample_data.py
                    echo     'Total Night Charge': np.random.uniform(0, 100, n_samples), >> generate_sample_data.py
                    echo     'Total Intl Minutes': np.random.uniform(0, 60, n_samples), >> generate_sample_data.py
                    echo     'Total Intl Calls': np.random.randint(0, 20, n_samples), >> generate_sample_data.py
                    echo     'Total Intl Charge': np.random.uniform(0, 20, n_samples), >> generate_sample_data.py
                    echo     'Customer Service Calls': np.random.randint(0, 10, n_samples) >> generate_sample_data.py
                    echo } >> generate_sample_data.py
                    echo. >> generate_sample_data.py
                    echo # Create churn based on some rules >> generate_sample_data.py
                    echo total_minutes = data['Total Day Minutes'] + data['Total Eve Minutes'] + data['Total Night Minutes'] >> generate_sample_data.py
                    echo intl_plan = np.array([1 if x == 'yes' else 0 for x in data['International Plan']]) >> generate_sample_data.py
                    echo cs_calls = data['Customer Service Calls'] >> generate_sample_data.py
                    echo. >> generate_sample_data.py
                    echo # Churn is more likely with high total minutes, international plan, and many customer service calls >> generate_sample_data.py
                    echo churn_prob = 0.05 + 0.1 * (total_minutes ^> 300) + 0.1 * intl_plan + 0.1 * (cs_calls ^> 3) >> generate_sample_data.py
                    echo data['Churn'] = np.random.binomial(1, churn_prob) >> generate_sample_data.py
                    echo data['Churn'] = ['True' if x == 1 else 'False' for x in data['Churn']] >> generate_sample_data.py
                    echo. >> generate_sample_data.py
                    echo # Create DataFrame >> generate_sample_data.py
                    echo df = pd.DataFrame(data) >> generate_sample_data.py
                    echo. >> generate_sample_data.py
                    echo # Split into training and testing sets >> generate_sample_data.py
                    echo train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) >> generate_sample_data.py
                    echo. >> generate_sample_data.py
                    echo # Save to CSV >> generate_sample_data.py
                    echo train_df.to_csv('churn-bigml-80.csv', index=False) >> generate_sample_data.py
                    echo test_df.to_csv('churn-bigml-20.csv', index=False) >> generate_sample_data.py
                    
                    rem Fix the comparison operators (Windows bat escaping)
                    powershell -Command "(Get-Content generate_sample_data.py) -replace '\\^>', '>' | Set-Content generate_sample_data.py"
                    
                    rem Run the script to generate data
                    python generate_sample_data.py
                )
                
                rem Verify data files exist
                if exist "churn-bigml-80.csv" if exist "churn-bigml-20.csv" (
                    echo Data files verified.
                    dir churn-bigml-*.csv
                ) else (
                    echo Data files not found. Exiting.
                    exit 1
                )
                '''
            }
        }
        
        stage('Train Model') {
            steps {
                echo 'Training decision tree model...'
                bat '''
                @echo off
                call %MODEL_ENV%\\Scripts\\activate.bat
                
                rem Run the training script
                python main.py --train_file churn-bigml-80.csv --test_file churn-bigml-20.csv --max_depth 5 --save
                '''
            }
        }
        
        stage('Evaluate Model') {
            steps {
                echo 'Evaluating model performance...'
                bat '''
                @echo off
                call %MODEL_ENV%\\Scripts\\activate.bat
                
                rem Check if model file exists
                if exist "decision_tree_model.pkl" (
                    echo Model file found.
                    dir decision_tree_model.pkl
                ) else (
                    echo Model file not found. Training likely failed.
                    exit 1
                )
                
                rem Create a directory for reports if it doesn't exist
                if not exist reports mkdir reports
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
        bat '''
        @echo off
        call %MODEL_ENV%\\Scripts\\activate.bat
        
        rem Create a Python script for report generation
        echo import joblib > create_report.py
        echo import pandas as pd >> create_report.py
        echo import numpy as np >> create_report.py
        echo from sklearn.metrics import accuracy_score, classification_report, confusion_matrix >> create_report.py
        echo import matplotlib.pyplot as plt >> create_report.py
        echo from sklearn.tree import plot_tree >> create_report.py
        echo import os >> create_report.py
        echo. >> create_report.py
        echo def create_report(): >> create_report.py
        echo     # Create reports directory if it doesn't exist >> create_report.py
        echo     os.makedirs("reports", exist_ok=True) >> create_report.py
        echo. >> create_report.py
        echo     # Load model and test data >> create_report.py
        echo     model = joblib.load("decision_tree_model.pkl") >> create_report.py
        echo     test_df = pd.read_csv("churn-bigml-20.csv") >> create_report.py
        echo. >> create_report.py
        echo     # Separate features and target >> create_report.py
        echo     X_test = test_df.drop(columns=["Churn"]) >> create_report.py
        echo     y_test = test_df["Churn"] >> create_report.py
        echo. >> create_report.py
        echo     # Encode categorical features >> create_report.py
        echo     for col in X_test.columns: >> create_report.py
        echo         if X_test[col].dtype == 'object': >> create_report.py
        echo             X_test[col] = X_test[col].astype('category').cat.codes >> create_report.py
        echo. >> create_report.py
        echo     # Make predictions >> create_report.py
        echo     y_pred = model.predict(X_test) >> create_report.py
        echo. >> create_report.py
        echo     # Calculate metrics >> create_report.py
        echo     accuracy = accuracy_score(y_test, y_pred) >> create_report.py
        echo     class_report = classification_report(y_test, y_pred, output_dict=True) >> create_report.py
        echo. >> create_report.py
        echo     # Feature importance >> create_report.py
        echo     feature_importance = pd.DataFrame({ >> create_report.py
        echo         'Feature': X_test.columns, >> create_report.py
        echo         'Importance': model.feature_importances_ >> create_report.py
        echo     }).sort_values('Importance', ascending=False) >> create_report.py
        echo. >> create_report.py
        echo     # Visualize the tree (limited depth for visibility) >> create_report.py
        echo     plt.figure(figsize=(20, 10)) >> create_report.py
        echo     plot_tree(model, max_depth=3, feature_names=X_test.columns, filled=True, rounded=True) >> create_report.py
        echo     plt.savefig("reports/decision_tree.png", bbox_inches="tight") >> create_report.py
        echo. >> create_report.py
        echo     # Create HTML report with escaped curly braces for CSS >> create_report.py
        echo     html = f"""^<!DOCTYPE html^>^<html^>^<head^>^<title^>Churn Prediction Model Report^</title^>^<style^>body {{ font-family: Arial, sans-serif; margin: 20px; }} h1, h2 {{ color: #2c3e50; }} .container {{ max-width: 1200px; margin: 0 auto; }} .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }} .metric-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }} table {{ border-collapse: collapse; width: 100%%; margin: 20px 0; }} th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }} th {{ background-color: #f2f2f2; }} tr:nth-child(even) {{ background-color: #f9f9f9; }} .image-container {{ text-align: center; margin: 20px 0; }} img {{ max-width: 100%%; height: auto; }}^</style^>^</head^>^<body^>^<div class="container"^>^<h1^>Churn Prediction Model Report^</h1^>^<p^>Date: {pd.Timestamp.now().strftime('%%Y-%%m-%%d %%H:%%M:%%S')}^</p^>^<h2^>Model Performance^</h2^>^<div class="metrics"^>^<div class="metric-box"^>^<h3^>Accuracy^</h3^>^<p style="font-size: 24px; font-weight: bold;"^>{accuracy:.4f}^</p^>^</div^>^<div class="metric-box"^>^<h3^>Precision^</h3^>^<p style="font-size: 24px; font-weight: bold;"^>{class_report['True']['precision']:.4f}^</p^>^</div^>^<div class="metric-box"^>^<h3^>Recall^</h3^>^<p style="font-size: 24px; font-weight: bold;"^>{class_report['True']['recall']:.4f}^</p^>^</div^>^<div class="metric-box"^>^<h3^>F1 Score^</h3^>^<p style="font-size: 24px; font-weight: bold;"^>{class_report['True']['f1-score']:.4f}^</p^>^</div^>^</div^>^<h2^>Feature Importance^</h2^>^<table^>^<tr^>^<th^>Feature^</th^>^<th^>Importance^</th^>^</tr^>{''.join(f'^<tr^>^<td^>{row.Feature}^</td^>^<td^>{row.Importance:.4f}^</td^>^</tr^>' for _, row in feature_importance.head(10).iterrows())}^</table^>^<h2^>Decision Tree Visualization^</h2^>^<div class="image-container"^>^<img src="decision_tree.png" alt="Decision Tree Visualization"^>^</div^>^<h2^>Confusion Matrix^</h2^>^<div class="image-container"^>^<img src="confusion_matrix.png" alt="Confusion Matrix"^>^</div^>^</div^>^</body^>^</html^>""" >> create_report.py
        echo. >> create_report.py
        echo     with open("reports/model_report.html", "w") as f: >> create_report.py
        echo         f.write(html) >> create_report.py
        echo. >> create_report.py
        echo     print("Report generated successfully: reports/model_report.html") >> create_report.py
        echo. >> create_report.py
        echo if __name__ == "__main__": >> create_report.py
        echo     create_report() >> create_report.py
        
        rem Fix HTML escaping for Windows batch (remove ^ characters)
        powershell -Command "(Get-Content create_report.py) -replace '\\^', '' | Set-Content create_report.py"
        
        rem Run the report generation script
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
                bat '''
                @echo off
                rem Keep model and reports but clean temporary files
                if exist "generate_sample_data.py" del /q generate_sample_data.py
                if exist "create_report.py" del /q create_report.py
                
                echo Workspace cleaned
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
