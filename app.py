import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# TensorFlow/Keras Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Risk Prediction Dashboard", layout="wide")
st.title("🚜 Agricultural Equipment Risk Prediction Dashboard")

# ==========================================
# DATA LOADING & PREPROCESSING (CACHED)
# ==========================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('unclean_datafarm.csv')
    
    # Drop rows where target variable is missing
    df = df.dropna(subset=['Risk_Category']).copy()
    
    # Drop columns with excessive missing values or irrelevant IDs
    cols_to_drop = [
        'Log_ID', 'Equipment_ID', 'Owner_Farm_ID', 'Operator_ID', 
        'Operator_Name', 'Field_ID', 'Field_Name', 
        'Maintenance_Type', 'Part_Replaced', 'Date', 'Purchase_Date'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Fix typographical errors
    typo_map = {'Tractr': 'Tractor', 'Harvstr': 'Harvester', 'Sprayr': 'Sprayer', 'Plntr': 'Planter'}
    df['Equipment_Type'] = df['Equipment_Type'].replace(typo_map)
    
    return df

df = load_and_clean_data()

# Setup Features and Target
X = df.drop(columns=['Risk_Category'])
y = df['Risk_Category']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

numeric_features = ['Experience_Years', 'Area_Acres', 'Hours_Used', 'Fuel_Liters', 'Maintenance_Cost', 'Breakdown_Probability']
categorical_features = ['Equipment_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Phase 1: Machine Learning (RF)", "Phase 2: Deep Learning (MLP)"])

# ==========================================
# PAGE 1: DATA OVERVIEW
# ==========================================
if page == "Data Overview":
    st.header("Dataset Exploration")
    st.write("Preview of the cleaned dataset used for training:")
    st.dataframe(df.head(15))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Category Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Risk_Category', palette='viridis', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Equipment Type Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Equipment_Type', palette='magma', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ==========================================
# PAGE 2: MACHINE LEARNING (ML)
# ==========================================
elif page == "Phase 1: Machine Learning (RF)":
    st.header("Machine Learning: Random Forest Baseline")
    
    with st.spinner("Training Random Forest Classifier..."):
        # Preprocessor
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
        
        # Transform data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Train
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X_train_processed, y_train)
        y_pred = rf_model.predict(X_test_processed)
        
        st.success("Model Trained Successfully!")
        st.metric(label="Test Accuracy", value=f"{accuracy_score(y_test, y_pred):.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Feature Importances")
            cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            all_feature_names = numeric_features + list(cat_feature_names)
            
            importances = rf_model.feature_importances_
            feature_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='viridis', ax=ax)
            st.pyplot(fig)

# ==========================================
# PAGE 3: DEEP LEARNING (DL)
# ==========================================
elif page == "Phase 2: Deep Learning (MLP)":
    st.header("Deep Learning: Multi-Layer Perceptron (MLP)")
    
    # Preprocessor for DL
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    input_dim = X_train_processed.shape[1]
    num_classes = len(np.unique(y_encoded))
    
    epochs_slider = st.sidebar.slider("Number of Epochs", min_value=10, max_value=150, value=50, step=10)
    
    if st.button("Train Neural Network"):
        with st.spinner(f"Training Neural Network for {epochs_slider} epochs... This may take a moment."):
            
            # DL Architecture
            dl_model = Sequential([
                Dense(64, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(num_classes, activation='softmax')
            ])
            
            dl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
            
            # Train
            history = dl_model.fit(
                X_train_processed, y_train,
                validation_split=0.2,
                epochs=epochs_slider,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            test_loss, test_accuracy = dl_model.evaluate(X_test_processed, y_test, verbose=0)
            
            st.success("Neural Network Trained Successfully!")
            st.metric(label="DL Test Accuracy", value=f"{test_accuracy:.4f}")
            
            st.subheader("Training Curves")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                ax.plot(history.history['accuracy'], label='Train Accuracy')
                ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax.set_title('Accuracy')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
            with col2:
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Train Loss')
                ax.plot(history.history['val_loss'], label='Validation Loss')
                ax.set_title('Loss')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
