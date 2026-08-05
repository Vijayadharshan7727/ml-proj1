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
# 1. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Agri-Risk AI", page_icon="🚜", layout="wide")

# Custom CSS for a cleaner, modern look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button { border-radius: 8px; background-color: #4CAF50; color: white; border: none; padding: 10px 24px; transition: 0.3s; }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.05); }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & PREPROCESSING (CACHED)
# ==========================================
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('unclean_datafarm.csv')
    df = df.dropna(subset=['Risk_Category']).copy()
    
    cols_to_drop = ['Log_ID', 'Equipment_ID', 'Owner_Farm_ID', 'Operator_ID', 
                    'Operator_Name', 'Field_ID', 'Field_Name', 'Maintenance_Type', 
                    'Part_Replaced', 'Date', 'Purchase_Date']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
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

# Global Preprocessor
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004122.png", width=100)
    st.title("Agri-Risk AI")
    st.markdown("---")
    page = st.radio("Explore Modules", ["📊 Data Explorer", "🌲 Phase 1: Machine Learning", "🧠 Phase 2: Deep Learning"])
    st.markdown("---")
    st.caption("Developed for Predictive Maintenance & Risk Assessment.")

# ==========================================
# 4. PAGE: DATA EXPLORER
# ==========================================
if page == "📊 Data Explorer":
    st.title("📊 Dataset Overview & Insights")
    st.markdown("Explore the cleaned historical data for agricultural equipment maintenance and risk profiles.")
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Features Used", X.shape[1])
    c3.metric("Equipment Types", df['Equipment_Type'].nunique())
    c4.metric("High Risk Items", len(df[df['Risk_Category'] == 'High']))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interactive Tabs
    tab1, tab2 = st.tabs(["📈 Visualizations", "🗄️ Raw Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Risk Category Distribution")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, x='Risk_Category', palette='rocket', ax=ax, order=['Low', 'Medium', 'High'])
            ax.set_ylabel("Number of Equipments")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Equipment Usage Breakdown")
            fig, ax = plt.subplots(figsize=(6,4))
            df['Equipment_Type'].value_counts().plot.pie(autopct='%1.1f%%', cmap='viridis', ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)
            
    with tab2:
        st.dataframe(df.style.highlight_max(axis=0, color='#ffcccc'), use_container_width=True)

# ==========================================
# 5. PAGE: MACHINE LEARNING (RF)
# ==========================================
elif page == "🌲 Phase 1: Machine Learning":
    st.title("🌲 Phase 1: Random Forest Classifier")
    st.markdown("A robust baseline model using ensemble learning to predict equipment failure risk.")
    
    if st.button("🚀 Train Machine Learning Model"):
        with st.spinner("Training Random Forest Classifier..."):
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf_model.fit(X_train_processed, y_train)
            y_pred = rf_model.predict(X_test_processed)
            
            st.balloons()
            st.success(f"Training Complete! Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5,4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, cbar=False)
                ax.set_ylabel('Actual Risk')
                ax.set_xlabel('Predicted Risk')
                st.pyplot(fig)
                
            with col2:
                st.subheader("Feature Importances")
                cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                all_feature_names = numeric_features + list(cat_feature_names)
                
                importances = rf_model.feature_importances_
                feature_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(6,5))
                sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(8), palette='mako', ax=ax)
                st.pyplot(fig)

# ==========================================
# 6. PAGE: DEEP LEARNING (MLP)
# ==========================================
elif page == "🧠 Phase 2: Deep Learning":
    st.title("🧠 Phase 2: Neural Network (MLP)")
    st.markdown("Advanced Multi-Layer Perceptron architecture utilizing Batch Normalization and Dropout to prevent overfitting.")
    
    # Interactive Hyperparameters
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Model Tuning")
    epochs = st.sidebar.slider("Epochs", min_value=10, max_value=150, value=50, step=10)
    batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
    
    if st.button("🚀 Train Neural Network"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        input_dim = X_train_processed.shape[1]
        num_classes = len(np.unique(y_encoded))
        
        # Build Architecture
        dl_model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        dl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                         loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Custom Callback to update Streamlit Progress Bar
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Training Epoch {epoch+1}/{epochs} | Loss: {logs['loss']:.4f}")

        history = dl_model.fit(
            X_train_processed, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[StreamlitCallback()],
            verbose=0
        )
        
        test_loss, test_accuracy = dl_model.evaluate(X_test_processed, y_test, verbose=0)
        
        status_text.empty()
        progress_bar.empty()
        st.success(f"Neural Network Trained Successfully! Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Plotting Curves
        st.subheader("📈 Learning Curves")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(history.history['accuracy'], label='Train Accuracy', color='#4CAF50', linewidth=2)
            ax.plot(history.history['val_accuracy'], label='Val Accuracy', color='#FF9800', linewidth=2, linestyle='--')
            ax.set_title('Model Accuracy')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(history.history['loss'], label='Train Loss', color='#F44336', linewidth=2)
            ax.plot(history.history['val_loss'], label='Val Loss', color='#2196F3', linewidth=2, linestyle='--')
            ax.set_title('Model Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
