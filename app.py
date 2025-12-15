from streamlit import config as st_config
# Set Streamlit configuration to avoid CORS/XSRF warning
st_config.set_option('server.enableCORS', True)
st_config.set_option('server.enableXsrfProtection', False)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc)
# type: ignore
from mlxtend.frequent_patterns import apriori, association_rules
# type: ignore
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Academic Performance Analysis System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'le_target' not in st.session_state:
    st.session_state.le_target = None
if 'kmeans_model' not in st.session_state:
    st.session_state.kmeans_model = None

# Title
st.markdown('<h1 class="main-header">📊 Web-Based Interactive Data Mining System</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #6B7280;">Academic Performance Analysis and Prediction</h3>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["🏠 Home", "📁 Data Upload", "🧹 Data Preprocessing", "🌲 Classification", 
     "📊 Clustering", "🔗 Association Rules", "📈 Visualization", "📋 Performance Evaluation"]
)

# Home Page
if page == "🏠 Home":
    st.markdown("""
    ## 🎯 Project Overview
    
    This project aims to develop a web-based interactive data mining system for analyzing student academic data and predicting academic performance.
    
    ### ✨ Main Features:
    
    1. **📁 Data Upload** - Upload CSV datasets or use sample data
    2. **🧹 Data Preprocessing** - Data cleaning, transformation, and feature engineering
    3. **🌲 Classification** - Predict student performance using Decision Tree
    4. **📊 Clustering** - Group students using K-Means clustering
    5. **🔗 Association Rules** - Discover patterns using Apriori algorithm
    6. **📈 Visualization** - Interactive charts and dashboards
    7. **📋 Performance Evaluation** - Model evaluation metrics
    
    ### 🛠️ Technologies Used:
    
    - Python, Pandas, NumPy
    - Scikit-learn (Machine Learning)
    - MLxtend (Association Rule Mining)
    - Streamlit (Web Application Framework)
    - Matplotlib, Plotly, Seaborn (Visualization)
    
    ### 📊 Data Mining Techniques:
    
    - Classification (Decision Tree)
    - Clustering (K-Means)
    - Association Rule Mining (Apriori)
    """)
    
    # Load sample data
    if st.button("📥 Load Sample Dataset", type="primary"):
        try:
            # Create sample data from the provided CSV content
            sample_data = {
                'gender': ['female', 'female', 'female', 'male', 'male'],
                'race/ethnicity': ['group B', 'group C', 'group B', 'group A', 'group C'],
                'parental level of education': ["bachelor's degree", "some college", "master's degree", 
                                                "associate's degree", "some college"],
                'lunch': ['standard', 'standard', 'standard', 'free/reduced', 'standard'],
                'test preparation course': ['none', 'completed', 'none', 'none', 'none'],
                'math score': [72, 69, 90, 47, 76],
                'reading score': [72, 90, 95, 57, 78],
                'writing score': [74, 88, 93, 44, 75]
            }
            df = pd.DataFrame(sample_data)
            
            # Create derived columns
            df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
            df['average_score'] = df['total_score'] / 3
            df['passed'] = (df['average_score'] >= 60).astype(int)
            
            st.session_state.dataset = df
            st.session_state.preprocessed_data = df.copy()
            
            st.success("✅ Sample dataset loaded successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", len(df))
            with col2:
                st.metric("Pass Rate", f"{(df['passed'].mean()*100):.1f}%")
            with col3:
                st.metric("Average Score", f"{df['average_score'].mean():.1f}")
            with col4:
                st.metric("Features", len(df.columns))
                
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            st.info("You can also upload your own dataset in the Data Upload page.")

# Data Upload Page
elif page == "📁 Data Upload":
    st.markdown('<h2 class="sub-header">📁 Data Upload</h2>', unsafe_allow_html=True)
    
    # Option 1: Upload CSV file
    uploaded_file = st.file_uploader("Upload your own CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.dataset = df
            st.success("✅ Data uploaded successfully!")
            
            # Display data info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Preview")
                st.dataframe(df.head())
            
            with col2:
                st.subheader("📋 Data Information")
                
                info_df = pd.DataFrame({
                    'Feature': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Missing Values': df.isnull().sum().values
                })
                st.dataframe(info_df)
            
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe())
            
            st.info(f"📐 Data Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Option 2: Load sample dataset
    st.markdown("---")
    st.subheader("📥 Load Sample Dataset")
    
    if st.button("📊 Load StudentsPerformance.csv"):
        try:
            # Try to load from data folder
            df = pd.read_csv('data/StudentsPerformance.csv')
            
            # Create derived columns
            df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
            df['average_score'] = df['total_score'] / 3
            df['passed'] = (df['average_score'] >= 60).astype(int)
            
            st.session_state.dataset = df
            st.session_state.preprocessed_data = df.copy()
            
            st.success(f"✅ StudentsPerformance.csv loaded! ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", len(df))
            with col2:
                st.metric("Pass Rate", f"{(df['passed'].mean()*100):.1f}%")
            with col3:
                avg_math = df['math score'].mean()
                st.metric("Avg Math", f"{avg_math:.1f}")
            with col4:
                st.metric("Features", len(df.columns))
            
            # Show preview
            with st.expander("👀 View Dataset Preview"):
                st.dataframe(df.head(10))
            
        except FileNotFoundError:
            st.error("❌ File not found. Please ensure 'data/StudentsPerformance.csv' exists.")
            st.info("You can upload your own CSV file above.")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
    
    # Display current dataset status
    if st.session_state.dataset is not None:
        st.markdown("---")
        st.subheader("📋 Current Dataset Status")
        df = st.session_state.dataset
        st.success(f"✅ Dataset is loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("Available for: Data Preprocessing, Classification, Clustering, etc.")

# Data Preprocessing Page
elif page == "🧹 Data Preprocessing":
    st.markdown('<h2 class="sub-header">🧹 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first from the Data Upload page!")
        st.stop()
    
    df = st.session_state.dataset.copy()
    
    st.subheader("📋 Original Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # 1. Handle Missing Values
    st.subheader("1️⃣ Handle Missing Values")
    missing_cols = df.columns[df.isnull().any()].tolist()
    
    if missing_cols:
        st.write(f"📌 Columns with missing values: {missing_cols}")
        
        for col in missing_cols:
            missing_rate = df[col].isnull().mean() * 100
            st.write(f"- **{col}**: {missing_rate:.1f}% missing")
            
        fill_method = st.selectbox(
            "Select filling method:",
            ["Drop rows", "Mean fill (numeric)", "Median fill (numeric)", "Mode fill (categorical)", "Custom value"]
        )
        
        if st.button("Apply Missing Value Handling", key="missing_values"):
            if fill_method == "Drop rows":
                df = df.dropna()
                st.success(f"✅ Dropped rows with missing values. Remaining: {len(df)} rows")
            elif fill_method == "Mean fill (numeric)":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in missing_cols:
                        df[col] = df[col].fillna(df[col].mean())
                st.success("✅ Filled numeric columns with mean values")
            elif fill_method == "Median fill (numeric)":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in missing_cols:
                        df[col] = df[col].fillna(df[col].median())
                st.success("✅ Filled numeric columns with median values")
            elif fill_method == "Mode fill (categorical)":
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if col in missing_cols:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                st.success("✅ Filled categorical columns with mode values")
            else:
                custom_value = st.text_input("Enter custom value:", "0")
                df = df.fillna(custom_value)
                st.success(f"✅ Filled missing values with: {custom_value}")
    else:
        st.success("✅ No missing values found in data!")
    
    # 2. Feature Encoding
    st.subheader("2️⃣ Feature Encoding")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        st.write(f"📌 Categorical features: {categorical_cols}")
        
        encoding_method = st.selectbox(
            "Select encoding method:",
            ["Label Encoding", "One-Hot Encoding"]
        )
        
        encode_cols = st.multiselect("Select columns to encode:", categorical_cols, default=categorical_cols[:min(3, len(categorical_cols))])
        
        if st.button("Apply Feature Encoding", key="encoding"):
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                for col in encode_cols:
                    df[col] = le.fit_transform(df[col])
                st.success(f"✅ Applied Label Encoding to {len(encode_cols)} columns")
            else:
                df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
                st.success(f"✅ Applied One-Hot Encoding. New features: {len(df.columns)}")
    else:
        st.info("ℹ️ No categorical features found.")
    
    # 3. Feature Scaling
    st.subheader("3️⃣ Feature Scaling")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.write(f"📌 Numeric features: {numeric_cols}")
        
        scale_cols = st.multiselect("Select columns to scale:", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))])
        
        if st.button("Apply Feature Scaling", key="scaling"):
            scaler = StandardScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            st.success("✅ Feature scaling complete!")
            st.session_state.scaler = scaler
    
    # 4. Feature Engineering
    st.subheader("4️⃣ Feature Engineering")
    
    if st.checkbox("Create derived features"):
        # Check if we have score columns
        score_cols = [col for col in df.columns if 'score' in col.lower() and df[col].dtype in [np.int64, np.float64]]
        
        if len(score_cols) >= 2:
            # Create total score if not exists
            if 'total_score' not in df.columns:
                total_score_col = st.selectbox("Select columns to sum for total score:", 
                                              score_cols, 
                                              default=score_cols[:min(3, len(score_cols))])
                if st.button("Create Total Score"):
                    df['total_score'] = df[total_score_col].sum(axis=1)
                    st.success("✅ Created total_score column")
            
            # Create pass/fail column
            if 'total_score' in df.columns and 'passed' not in df.columns:
                threshold = st.slider("Passing threshold (total score):", 0, 300, 180)
                if st.button("Create Pass/Fail Column"):
                    df['passed'] = (df['total_score'] >= threshold).astype(int)
                    st.success(f"✅ Created 'passed' column (threshold: {threshold})")
    
    # Convert object columns to string to avoid Arrow serialization warnings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    
    # Save preprocessed data
    if st.button("💾 Save Preprocessed Data", type="primary"):
        st.session_state.preprocessed_data = df
        st.success("✅ Preprocessed data saved to session state!")
    
    # Display preprocessed data
    st.subheader("📊 Preprocessed Data Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head())
    with col2:
        st.write("**Data Summary:**")
        st.write(f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.write(f"- Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
        st.write(f"- Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
        st.write(f"- Missing values: {df.isnull().sum().sum()}")

# Classification Page
elif page == "🌲 Classification":
    st.markdown('<h2 class="sub-header">🌲 Classification Analysis (Decision Tree)</h2>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first from the Data Upload page!")
        st.stop()
    
    df = st.session_state.preprocessed_data if st.session_state.preprocessed_data is not None else st.session_state.dataset.copy()
    
    st.subheader("🎯 Create Target Variable")
    
    # Check if we have score columns
    score_cols = [col for col in df.columns if 'score' in col.lower() and df[col].dtype in [np.int64, np.float64]]
    
    # Create total score if not exists
    if 'total_score' not in df.columns and len(score_cols) >= 1:
        if st.button("Calculate Total Score"):
            df['total_score'] = df[score_cols].sum(axis=1)
            st.success(f"✅ Created total_score from {len(score_cols)} score columns")
    
    # Create target variable options
    target_option = st.radio(
        "Choose target variable to predict:",
        ["Pass/Fail (Binary)", "Math Score Level", "Reading Score Level", "Writing Score Level", "Total Score Level"]
    )
    
    if target_option == "Pass/Fail (Binary)":
        if 'total_score' in df.columns:
            threshold = st.slider("Passing threshold (total score):", 
                                 int(df['total_score'].min()), 
                                 int(df['total_score'].max()), 
                                 int(df['total_score'].median()))
            df['target'] = (df['total_score'] >= threshold).astype(int)
            target_name = f"passed (total_score ≥ {threshold})"
        else:
            st.error("❌ Need total_score column. Please create it first or select different target.")
            st.stop()
    
    elif "Level" in target_option:
        # Extract score column name
        score_type = target_option.replace(" Score Level", "").lower()
        score_col = f"{score_type} score" if f"{score_type} score" in df.columns else None
        
        if not score_col:
            # Try to find similar column
            for col in df.columns:
                if score_type in col.lower():
                    score_col = col
                    break
        
        if score_col:
            bins = st.slider(f"Number of bins for {score_type} score:", 3, 10, 4)
            df['target'] = pd.qcut(df[score_col], q=bins, labels=[f'Level_{i}' for i in range(bins)])
            target_name = f"{score_type}_level"
        else:
            st.error(f"❌ Could not find {score_type} score column")
            st.stop()
    
    # Show dataset with target
    with st.expander("📋 View dataset with target variable"):
        st.dataframe(df[['target'] + list(df.columns[:5])].head())
    
    st.subheader("⚙️ Select Features for Classification")
    
    # Exclude target and ID columns from features
    exclude_cols = ['target', 'student_id', 'id', 'name', 'email']
    available_features = [col for col in df.columns if col not in exclude_cols and df[col].nunique() > 1]
    
    selected_features = st.multiselect(
        "Select features to use for prediction:",
        available_features,
        default=available_features[:min(5, len(available_features))]
    )
    
    if not selected_features:
        st.error("❌ Please select at least one feature.")
        st.stop()
    
    # Model parameters
    st.subheader("🌳 Decision Tree Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    with col2:
        max_depth = st.slider("Max Depth", 2, 20, 5)
    with col3:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
    
    if st.button("🚀 Train Decision Tree Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                X = df[selected_features].copy()
                y = df['target']
                
                # Encode categorical features
                le_dict = {}
                X_encoded = X.copy()
                for col in X_encoded.select_dtypes(include=['object']).columns:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col])
                    le_dict[col] = le
                
                # Encode target if categorical
                if y.dtype == 'object':
                    le_target = LabelEncoder()
                    y_encoded = le_target.fit_transform(y)
                    st.session_state.le_target = le_target
                else:
                    y_encoded = y
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_encoded, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded if len(np.unique(y_encoded)) < 10 else None
                )
                
                # Train model
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store in session state
                st.session_state.trained_model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.accuracy = accuracy
                st.session_state.target_name = target_name
                st.session_state.selected_features = selected_features
                st.session_state.feature_encoders = le_dict
                
                # Display results
                st.success("✅ Model trained successfully!")
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("Training Samples", len(X_train))
                with col3:
                    st.metric("Test Samples", len(X_test))
                
                # Confusion Matrix
                st.subheader("📊 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=np.unique(y_test) if y.dtype != 'object' else le_target.classes_,
                           yticklabels=np.unique(y_test) if y.dtype != 'object' else le_target.classes_,
                           ax=ax)
                plt.title('Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
                
                # Feature Importance
                st.subheader("📈 Feature Importance")
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(range(len(feature_importance)), 
                                  feature_importance['Importance'],
                                  color='skyblue')
                    
                    ax.set_yticks(range(len(feature_importance)))
                    ax.set_yticklabels(feature_importance['Feature'])
                    ax.set_xlabel('Importance Score')
                    ax.set_title('Feature Importance in Decision Tree')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Classification Report
                st.subheader("📋 Classification Report")
                if y.dtype == 'object':
                    report = classification_report(y_test, y_pred, target_names=le_target.classes_)
                else:
                    report = classification_report(y_test, y_pred)
                st.text(report)
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
    
    # Prediction interface
    if 'trained_model' in st.session_state and st.session_state.trained_model is not None:
        st.subheader("🔮 Make Predictions")
        
        if selected_features:
            input_data = {}
            cols = st.columns(2)
            
            for i, feature in enumerate(selected_features):
                col_idx = i % 2
                with cols[col_idx]:
                    # FIXED: Check if feature_encoders exists and contains the feature
                    if (st.session_state.feature_encoders is not None and 
                        feature in st.session_state.feature_encoders):
                        # For categorical features
                        encoder = st.session_state.feature_encoders[feature]
                        unique_vals = encoder.classes_
                        input_data[feature] = st.selectbox(feature, unique_vals)
                    elif df[feature].dtype in [np.int64, np.float64]:
                        # For numeric features
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        default_val = float(df[feature].median())
                        input_data[feature] = st.number_input(feature, min_val, max_val, default_val)
                    else:
                        # For other types
                        unique_vals = df[feature].unique()
                        input_data[feature] = st.selectbox(feature, unique_vals)
            
            if st.button("Predict"):
                model = st.session_state.trained_model
                
                # Prepare input for prediction
                input_df = pd.DataFrame([input_data])
                
                # Encode using the same encoders
                for col in selected_features:
                    if (st.session_state.feature_encoders is not None and 
                        col in st.session_state.feature_encoders):
                        encoder = st.session_state.feature_encoders[col]
                        if input_data[col] in encoder.classes_:
                            input_df[col] = encoder.transform([input_data[col]])[0]
                        else:
                            # Handle unseen categories
                            input_df[col] = -1  # Or some default value
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Decode if categorical target
                if st.session_state.le_target is not None:
                    result = st.session_state.le_target.inverse_transform([prediction])[0]
                else:
                    result = "PASS" if prediction == 1 else "FAIL"
                
                st.success(f"🎯 Prediction: **{result}**")
                
                # Show prediction probability if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)[0]
                    st.info(f"Prediction probabilities: {proba}")
    else:
        st.info("ℹ️ Please train a model first to enable predictions.")

# Clustering Page
elif page == "📊 Clustering":
    st.markdown('<h2 class="sub-header">📊 Clustering Analysis (K-Means)</h2>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first from the Data Upload page!")
        st.stop()
    
    df = st.session_state.preprocessed_data if st.session_state.preprocessed_data is not None else st.session_state.dataset
    
    # Parameters
    st.sidebar.subheader("K-Means Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    max_iter = st.sidebar.slider("Max Iterations", 100, 500, 300)
    
    # Select features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric features available for clustering. Please preprocess data first.")
        st.stop()
    
    selected_features = st.multiselect("Select clustering features:", numeric_cols, 
                                       default=numeric_cols[:min(3, len(numeric_cols))])
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features for clustering")
        st.stop()
    
    if st.button("🔍 Perform K-Means Clustering", type="primary"):
        # Prepare data
        X = df[selected_features].dropna()
        
        if len(X) < n_clusters:
            st.error(f"❌ Not enough data points ({len(X)}) for {n_clusters} clusters")
            st.stop()
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        df_clustered = df.copy()
        df_clustered = df_clustered.loc[X.index]
        df_clustered['cluster'] = clusters
        
        # Visualization
        st.subheader("📊 Clustering Results")
        
        # 2D Scatter plot
        if len(selected_features) >= 2:
            fig = px.scatter(
                df_clustered,
                x=selected_features[0],
                y=selected_features[1],
                color='cluster',
                title=f'K-Means Clustering Results ({n_clusters} clusters)',
                hover_data=selected_features,
                color_continuous_scale=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Scatter plot if we have 3+ features
        if len(selected_features) >= 3:
            fig = px.scatter_3d(
                df_clustered,
                x=selected_features[0],
                y=selected_features[1],
                z=selected_features[2],
                color='cluster',
                title='3D Clustering Visualization',
                hover_data=selected_features
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Centers
        st.subheader("🎯 Cluster Centers")
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)
        
        centers_df = pd.DataFrame(
            centers_original,
            columns=selected_features,
            index=[f'Cluster {i}' for i in range(n_clusters)]
        )
        st.dataframe(centers_df.style.format("{:.2f}"))
        
        # Cluster Sizes
        st.subheader("📈 Cluster Distribution")
        cluster_sizes = pd.DataFrame(df_clustered['cluster'].value_counts().sort_index())
        cluster_sizes.columns = ['Count']
        cluster_sizes['Percentage'] = (cluster_sizes['Count'] / len(df_clustered) * 100).round(1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(cluster_sizes)
        with col2:
            fig = px.pie(cluster_sizes, values='Count', names=cluster_sizes.index, 
                        title='Cluster Size Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Save results
        st.session_state.clusters = df_clustered
        st.session_state.kmeans_model = kmeans
        
        st.success(f"✅ Clustering complete! Created {n_clusters} clusters.")

# Association Rules Page
elif page == "🔗 Association Rules":
    st.markdown('<h2 class="sub-header">🔗 Association Rule Mining (Apriori)</h2>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first from the Data Upload page!")
        st.stop()
    
    df = st.session_state.dataset.copy()
    
    st.subheader("🛠️ Data Preparation for Association Rules")
    
    # Create bins for numerical scores to make them categorical
    score_cols = [col for col in df.columns if 'score' in col.lower() and df[col].dtype in [np.int64, np.float64]]
    
    for score_col in score_cols[:3]:  # Process up to 3 score columns
        if score_col not in df.columns:
            continue
            
        col_name = score_col.replace(' ', '_').replace('/', '_').lower()
        if 'math' in col_name:
            df['math_level'] = pd.cut(df[score_col], 
                                     bins=[0, 50, 70, 90, 101], 
                                     labels=['Math_Low', 'Math_Medium', 'Math_High', 'Math_Excellent'])
        elif 'read' in col_name:
            df['reading_level'] = pd.cut(df[score_col], 
                                        bins=[0, 50, 70, 90, 101], 
                                        labels=['Reading_Low', 'Reading_Medium', 'Reading_High', 'Reading_Excellent'])
        elif 'writ' in col_name:
            df['writing_level'] = pd.cut(df[score_col], 
                                        bins=[0, 50, 70, 90, 101], 
                                        labels=['Writing_Low', 'Writing_Medium', 'Writing_High', 'Writing_Excellent'])
    
    # Create total score level if we have total_score
    if 'total_score' in df.columns:
        df['total_level'] = pd.cut(df['total_score'], 
                                  bins=[0, 150, 210, 270, 301], 
                                  labels=['Total_Low', 'Total_Medium', 'Total_High', 'Total_Excellent'])
    
    st.subheader("📋 Select Features for Association Rule Mining")
    
    # Available features for association rules
    categorical_features = [col for col in df.columns if df[col].dtype == 'object']
    
    numerical_level_features = [col for col in df.columns if 'level' in col.lower()]
    
    all_features = categorical_features + numerical_level_features
    
    if not all_features:
        st.error("❌ No categorical features found for association rule mining.")
        st.info("Please ensure your dataset has categorical columns or create level columns from scores.")
        st.stop()
    
    selected_features = st.multiselect(
        "Select features to analyze:",
        all_features,
        default=all_features[:min(5, len(all_features))]
    )
    
    if len(selected_features) < 2:
        st.error("❌ Please select at least 2 features for association rule mining.")
        st.stop()
    
    # Parameter settings
    st.subheader("⚙️ Set Association Rule Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01,
                              help="Minimum frequency of itemset in dataset")
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05,
                                 help="Minimum probability of rule being true")
    with col3:
        min_lift = st.slider("Minimum Lift", 0.5, 5.0, 1.0, 0.1,
                           help="Minimum strength of association (1 = independent)")
    
    if st.button("⛏️ Mine Association Rules", type="primary"):
        with st.spinner("Mining association rules..."):
            try:
                # Create transactions
                transactions = []
                for _, row in df[selected_features].iterrows():
                    transaction = []
                    for feature in selected_features:
                        if pd.notna(row[feature]):
                            transaction.append(f"{feature}={row[feature]}")
                    transactions.append(transaction)
                
                # Convert to one-hot encoded format
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                
                # Generate frequent itemsets
                frequent_itemsets = apriori(df_encoded, 
                                           min_support=min_support,
                                           use_colnames=True,
                                           max_len=3)  # Limit to 3 items for simplicity
                
                if len(frequent_itemsets) == 0:
                    st.warning("⚠️ No frequent itemsets found. Try lowering the minimum support.")
                else:
                    # Generate association rules
                    rules = association_rules(frequent_itemsets,
                                             metric="confidence",
                                             min_threshold=min_confidence)
                    
                    # Filter by lift
                    rules = rules[rules['lift'] >= min_lift]
                    
                    if len(rules) == 0:
                        st.warning("⚠️ No association rules found with current parameters. Try adjusting confidence or lift.")
                    else:
                        st.success(f"🎉 Found {len(rules)} association rules!")
                        
                        # Sort by confidence and lift
                        rules = rules.sort_values(['confidence', 'lift'], ascending=False)
                        
                        # Display rules
                        st.subheader("📊 Discovered Association Rules")
                        
                        # Format rules for better display
                        display_rules = rules.copy()
                        display_rules['antecedents'] = display_rules['antecedents'].apply(
                            lambda x: ', '.join(list(x))
                        )
                        display_rules['consequents'] = display_rules['consequents'].apply(
                            lambda x: ', '.join(list(x))
                        )
                        
                        # Show top 20 rules
                        st.dataframe(
                            display_rules[['antecedents', 'consequents', 
                                         'support', 'confidence', 'lift']].head(20),
                            use_container_width=True
                        )
                        
                        # Rule analysis
                        st.subheader("📈 Rule Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Strongest Rules (by Confidence):**")
                            top_conf = rules.head(3)
                            for idx, rule in top_conf.iterrows():
                                antecedents = list(rule['antecedents'])
                                consequents = list(rule['consequents'])
                                st.write(f"**{idx+1}.** {antecedents[0] if antecedents else ''} → {consequents[0] if consequents else ''}")
                                st.write(f"   Confidence: {rule['confidence']:.2%}, Support: {rule['support']:.2%}")
                        
                        with col2:
                            st.write("**Most Interesting Rules (by Lift):**")
                            top_lift = rules.sort_values('lift', ascending=False).head(3)
                            for idx, rule in top_lift.iterrows():
                                antecedents = list(rule['antecedents'])
                                consequents = list(rule['consequents'])
                                st.write(f"**{idx+1}.** {antecedents[0] if antecedents else ''} → {consequents[0] if consequents else ''}")
                                st.write(f"   Lift: {rule['lift']:.2f}, Confidence: {rule['confidence']:.2%}")
                        
                        # Visualizations
                        if len(rules) > 0:
                            st.subheader("📊 Rule Visualizations")
                            
                            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                            
                            # Support vs Confidence
                            scatter = axes[0].scatter(rules['support'], rules['confidence'], 
                                                     c=rules['lift'], cmap='viridis', 
                                                     alpha=0.6, s=100)
                            axes[0].set_xlabel('Support')
                            axes[0].set_ylabel('Confidence')
                            axes[0].set_title('Support vs Confidence (colored by Lift)')
                            plt.colorbar(scatter, ax=axes[0])
                            
                            # Lift distribution
                            axes[1].hist(rules['lift'], bins=20, edgecolor='black', alpha=0.7)
                            axes[1].axvline(x=1.0, color='red', linestyle='--', label='Independent')
                            axes[1].set_xlabel('Lift')
                            axes[1].set_ylabel('Frequency')
                            axes[1].set_title('Distribution of Lift Values')
                            axes[1].legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Export results
                        st.subheader("📥 Export Rules")
                        
                        csv = display_rules.to_csv(index=False)
                        st.download_button(
                            label="📄 Download Association Rules as CSV",
                            data=csv,
                            file_name="association_rules.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"❌ Error mining association rules: {str(e)}")
                st.info("💡 Try selecting different features or adjusting parameters.")
    
    # Explanation of metrics
    with st.expander("ℹ️ Understanding Association Rule Metrics"):
        st.markdown("""
        ### 📊 Key Metrics:
        
        **Support**: How frequently the itemset appears in the dataset.
        - *Example*: Support of 0.1 means the itemset appears in 10% of transactions.
        
        **Confidence**: Probability that consequent occurs given antecedent.
        - *Example*: Confidence of 0.8 means when antecedent occurs, consequent occurs 80% of the time.
        
        **Lift**: How much more likely consequent is given antecedent vs by chance.
        - *Lift = 1*: Independent (no association)
        - *Lift > 1*: Positive association (more likely together)
        - *Lift < 1*: Negative association (less likely together)
        
        ### 🎯 Example Interpretation:
        - **Rule**: `test preparation course=completed → math_level=High`
        - **Confidence**: 0.75 → 75% of students who completed test prep have high math scores
        - **Lift**: 1.8 → They're 1.8x more likely to have high math scores than average
        """)

# Visualization Page
elif page == "📈 Visualization":
    st.markdown('<h2 class="sub-header">📈 Visual Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.dataset is None:
        st.warning("⚠️ Please load a dataset first from the Data Upload page!")
        st.stop()
    
    df = st.session_state.dataset
    
    # Select visualization type
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Distribution Analysis", "Correlation Analysis", "Score Analysis", "Group Comparison", "Clustering Results"]
    )
    
    if viz_type == "Distribution Analysis":
        st.subheader("📊 Data Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Numerical distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_num_col = st.selectbox("Select numerical feature:", numeric_cols)
                fig = px.histogram(df, x=selected_num_col, 
                                   title=f'{selected_num_col} Distribution',
                                   nbins=30,
                                   color_discrete_sequence=['#3B82F6'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Categorical distribution
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                selected_cat_col = st.selectbox("Select categorical feature:", categorical_cols)
                value_counts = df[selected_cat_col].value_counts().reset_index()
                value_counts.columns = [selected_cat_col, 'count']
                
                fig = px.pie(value_counts, 
                            values='count', 
                            names=selected_cat_col,
                            title=f'{selected_cat_col} Distribution',
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Analysis":
        st.subheader("📈 Feature Correlation Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix,
                           title='Feature Correlation Heatmap',
                           color_continuous_scale='RdBu',
                           aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter matrix
            st.subheader("Scatter Matrix")
            selected_cols = st.multiselect("Select columns for scatter matrix:", 
                                          numeric_df.columns.tolist(),
                                          default=numeric_df.columns.tolist()[:4])
            if len(selected_cols) >= 2:
                fig = px.scatter_matrix(df[selected_cols])
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Score Analysis":
        st.subheader("🎯 Student Score Analysis")
        
        # Check for score columns
        score_cols = [col for col in df.columns if 'score' in col.lower() and df[col].dtype in [np.int64, np.float64]]
        
        if score_cols:
            selected_scores = st.multiselect("Select score columns:", score_cols, default=score_cols[:min(3, len(score_cols))])
            
            if selected_scores:
                # Box plot of scores
                fig = px.box(df, y=selected_scores, 
                            title='Score Distribution Box Plot',
                            points='all')
                st.plotly_chart(fig, use_container_width=True)
                
                # Violin plot
                fig = px.violin(df, y=selected_scores, 
                               title='Score Distribution Violin Plot',
                               box=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Score comparison by category
                if len(selected_scores) >= 2:
                    category_col = st.selectbox("Select category for comparison:", 
                                               df.select_dtypes(include=['object']).columns.tolist())
                    if category_col:
                        fig = px.scatter(df, x=selected_scores[0], y=selected_scores[1],
                                        color=category_col,
                                        title=f'{selected_scores[0]} vs {selected_scores[1]} by {category_col}')
                        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Group Comparison":
        st.subheader("📊 Group Comparison Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            group_col = st.selectbox("Select grouping variable:", 
                                    df.select_dtypes(include=['object']).columns.tolist())
        
        with col2:
            value_col = st.selectbox("Select numerical variable:", 
                                    df.select_dtypes(include=[np.number]).columns.tolist())
        
        if group_col and value_col:
            # Box plot
            fig = px.box(df, x=group_col, y=value_col,
                        title=f'{value_col} Distribution by {group_col}',
                        color=group_col)
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart of means
            means = df.groupby(group_col)[value_col].mean().reset_index()
            fig = px.bar(means, x=group_col, y=value_col,
                        title=f'Average {value_col} by {group_col}',
                        color=group_col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Clustering Results":
        if 'clusters' in st.session_state and st.session_state.clusters is not None:
            st.subheader("🔍 Clustering Visualization")
            
            df_clustered = st.session_state.clusters
            
            # Select features for visualization
            numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'cluster']
            
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis feature:", numeric_cols)
                y_col = st.selectbox("Y-axis feature:", [col for col in numeric_cols if col != x_col])
                
                fig = px.scatter(df_clustered, x=x_col, y=y_col,
                                color='cluster',
                                title=f'Cluster Visualization: {x_col} vs {y_col}',
                                hover_data=df_clustered.columns.tolist())
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No clustering results found. Please perform clustering analysis first.")

# Performance Evaluation Page
elif page == "📋 Performance Evaluation":
    st.markdown('<h2 class="sub-header">📋 Model Performance Evaluation</h2>', unsafe_allow_html=True)
    
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.warning("⚠️ Please train a classification model first in the 'Classification' page.")
        st.stop()
    
    # Get data from session state
    model = st.session_state.trained_model
    X_test = st.session_state.get('X_test', None)
    y_test = st.session_state.get('y_test', None)
    y_pred = st.session_state.get('y_pred', None)
    
    if X_test is None or y_test is None or y_pred is None:
        st.error("❌ Model evaluation data not found. Please train the model again.")
        st.stop()
    
    target_name = st.session_state.get('target_name', 'target')
    selected_features = st.session_state.get('selected_features', [])
    
    st.success(f"✅ Evaluating model for target: **{target_name}**")
    
    # Display basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", "Decision Tree")
    with col2:
        st.metric("Features Used", len(selected_features))
    with col3:
        st.metric("Test Samples", len(X_test))
    
    # Performance Metrics
    st.subheader("📈 Performance Metrics")
    
    # Calculate various metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    # Display metrics in columns
    cols = st.columns(4)
    metric_names = list(metrics.keys())
    for i, (col, metric_name) in enumerate(zip(cols, metric_names)):
        col.metric(metric_name, f"{metrics[metric_name]:.2%}")
    
    # Detailed classification report
    st.subheader("📋 Detailed Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format({
        'precision': '{:.2%}',
        'recall': '{:.2%}',
        'f1-score': '{:.2%}',
        'support': '{:.0f}'
    }))
    
    # Confusion Matrix with better visualization
    st.subheader("🧮 Confusion Matrix Heatmap")
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test) if len(np.unique(y_test)) < 10 else range(len(np.unique(y_test))), 
                yticklabels=np.unique(y_test) if len(np.unique(y_test)) < 10 else range(len(np.unique(y_test))),
                ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)
    
    # Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("⚖️ Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(feature_importance)), 
                      feature_importance['Importance'],
                      color='skyblue')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, feature_importance['Importance'])):
            ax.text(importance + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}',
                   va='center')
        
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['Feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance in Decision Tree')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Error Analysis
    st.subheader("🔍 Error Analysis")
    
    # Create error analysis dataframe
    error_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Is_Correct': y_test == y_pred
    })
    
    # Calculate error rate
    accuracy = metrics['Accuracy']
    error_rate = 1 - accuracy
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Correct Predictions", f"{accuracy:.2%}")
    with col2:
        st.metric("Error Rate", f"{error_rate:.2%}")
    
    # Show misclassified examples
    misclassified = error_df[~error_df['Is_Correct']]
    if len(misclassified) > 0:
        st.write(f"**{len(misclassified)} misclassified samples:**")
        st.dataframe(misclassified.head(10))
    else:
        st.success("🎉 Perfect prediction! No misclassified samples.")
    
    # Export results option
    st.subheader("📥 Export Results")
    
    results_data = {
        'model_type': 'Decision Tree',
        'target_variable': target_name,
        'accuracy': accuracy,
        'precision': metrics['Precision'],
        'recall': metrics['Recall'],
        'f1_score': metrics['F1-Score'],
        'test_samples': len(X_test),
        'features': ', '.join(selected_features)
    }
    
    results_df = pd.DataFrame([results_data])
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📄 Download Results as CSV",
        data=csv,
        file_name="model_performance_results.csv",
        mime="text/csv"
    )

# Run the application
if __name__ == "__main__":
    pass