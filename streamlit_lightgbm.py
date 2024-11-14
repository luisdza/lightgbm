# streamlit_lightgbm.py

import streamlit as st
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from flaml.automl.data import load_openml_dataset

# Set Streamlit page configuration
st.set_page_config(
    page_title="LightGBM Model Trainer",
    page_icon="🌟",
    layout="wide")

# Custom CSS for aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        width: 100%;
        height: 50px;
        font-size: 16px;
    }
    .stSidebar {
        background-color: #f7f9fc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("🚀 Microsoft LightGBM with Streamlit")
st.markdown("<hr>", unsafe_allow_html=True)
st.write("Upload your dataset to train a LightGBM model with configurable parameters!")

# Example dataset
st.write("### Example Dataset")
st.write("Download [houses dataset](https://www.openml.org/d/537) from OpenML. The task is to predict the median price of the house in the region based on demographic composition and the state of the housing market in the region.")

# Test dataset loading
st.write("### Load Test Dataset")
if st.button("Load Example Dataset"):
    with st.spinner('Loading dataset...'):
        try:
            dataset = load_openml_dataset(dataset_id=537, data_dir="./")
            if dataset is None:
                raise ValueError(
                    "Failed to load dataset. The dataset returned None. Please check the dataset ID and try again.")
            X_train, X_test, y_train, y_test = dataset
            if X_train is None or y_train is None:
                raise ValueError(
                    "Failed to load dataset. The dataset returned None. Please check the dataset ID and try again.")
            st.success("Dataset loaded successfully!")
            y_train = y_train.reset_index(drop=True)
            data = pd.concat([X_train, y_train], axis=1)
            sample_dataset_loaded = True
            print(
                "Dataset loaded: X_train shape:",
                X_train.shape,
                "y_train shape:",
                y_train.shape)
        except Exception as e:
            st.error(
                f"Error loading dataset: {e}. Please verify your internet connection or try a different dataset.")
            print("Error loading dataset:", e)
            st.stop()
else:
    sample_dataset_loaded = False

# Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None or sample_dataset_loaded:
    # Load dataset if uploaded
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if data.empty:
                raise ValueError(
                    "The uploaded CSV file is empty. Please upload a valid dataset.")
            print("Uploaded dataset shape:", data.shape)
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            print("Error reading the uploaded file:", e)
            st.stop()

    st.write("### Dataset Preview")
    if "show_full_dataset" not in st.session_state:
        st.session_state.show_full_dataset = False

    def update_preview():
        st.session_state.show_full_dataset = not st.session_state.show_full_dataset

    st.checkbox(
        "Show full dataset",
        value=st.session_state.show_full_dataset,
        on_change=update_preview)
    # Debug print statement
    print("Preview option selected:", st.session_state.show_full_dataset)
    if st.session_state.show_full_dataset:
        st.dataframe(data, use_container_width=True)
    else:
        st.dataframe(data.head(), use_container_width=True)

    # Validate that the dataset has sufficient columns
    if len(data.columns) < 2:
        st.error(
            "The dataset must contain at least two columns: one target and at least one feature.")
        st.stop()

    # Select target and features
    st.sidebar.header("Data Configuration")
    target_column = st.sidebar.selectbox(
        "Select the target column", data.columns)
    feature_columns = st.sidebar.multiselect(
        "Select the feature columns",
        data.columns,
        default=data.columns.drop(target_column))
    if len(feature_columns) == 0:
        st.error("Please select at least one feature column.")
        st.stop()
    print("Target column:", target_column)
    print("Feature columns:", feature_columns)

    # Helper functions to determine objective and metric
    def get_objective_and_metric(y):
        if pd.api.types.is_float_dtype(y):
            return 'regression', 'rmse'
        elif y.nunique() > 2:
            return 'multiclass', 'multi_logloss'
        else:
            return 'binary', 'binary_error'

    # Model parameters configuration
    st.sidebar.header("LightGBM Parameters Configuration")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.05)
    num_leaves = st.sidebar.slider("Number of Leaves", 10, 100, 31)
    n_estimators = st.sidebar.slider("Number of Boosting Rounds", 50, 500, 100)
    feature_fraction = st.sidebar.slider("Feature Fraction", 0.1, 1.0, 0.9)
    early_stopping_rounds = st.sidebar.slider(
        "Early Stopping Rounds", 5, 50, 10)
    default_threshold = st.sidebar.slider(
        "Default Classification Threshold", 0.0, 1.0, 0.5)
    importance_type = st.sidebar.selectbox(
        "Select Feature Importance Type", [
            'split', 'gain'], index=0)
    print(
        "Model parameters: learning_rate=",
        learning_rate,
        ", num_leaves=",
        num_leaves,
        ", n_estimators=",
        n_estimators,
        ", feature_fraction=",
        feature_fraction,
        ", early_stopping_rounds=",
        early_stopping_rounds)

    if st.button("Train Model"):
        # Prepare data
        X = data[feature_columns]
        y = data[target_column]
        print("Training data shapes: X=", X.shape, ", y=", y.shape)

        # Split data if not using preloaded dataset
        if not sample_dataset_loaded:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            print(
                "Split data: X_train shape:",
                X_train.shape,
                "X_test shape:",
                X_test.shape,
                "y_train shape:",
                y_train.shape,
                "y_test shape:",
                y_test.shape)

        # Check for NaN or missing values
        if X_train.isnull().values.any() or y_train.isnull().values.any(
        ) or X_test.isnull().values.any() or y_test.isnull().values.any():
            st.error(
                "The dataset contains NaN or missing values. Please clean your data before training.")
            print("Error: Dataset contains NaN or missing values.")
            st.stop()

        # LightGBM dataset preparation with error handling
        try:
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            print("LightGBM datasets prepared.")
        except ValueError as e:
            st.error(f"Error in dataset preparation: {e}")
            print("Error in dataset preparation:", e)
            st.stop()

        # Define parameters
        objective, metric = get_objective_and_metric(y)
        params = {
            'objective': objective,
            'metric': metric,
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction
        }
        print("LightGBM parameters:", params)

        # Train model
        with st.spinner('Training the model...'):
            try:
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=n_estimators,
                    valid_sets=[
                        train_data,
                        test_data],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=10)
                print("Model training completed.")
            except Exception as e:
                st.error(f"Error during model training: {e}")
                print("Error during model training:", e)
                st.stop()

        # Predict and evaluate
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        print("Predictions made.")
        if pd.api.types.is_float_dtype(y):  # regression
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            st.success(f"Model RMSE: {rmse:.4f}")
            print("Model RMSE:", rmse)
        else:  # classification
            y_pred_binary = (y_pred >= default_threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred_binary)
            st.success(f"Model Accuracy: {accuracy:.4f}")
            print("Model Accuracy:", accuracy)

        # Feature importance
        feature_importance = model.feature_importance(
            importance_type=importance_type)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(feature_importance_df, use_container_width=True)
        print("Feature importance calculated.")

        # Plot feature importance
        if not feature_importance_df.empty:
            st.write("### Feature Importance Chart")
            fig, ax = plt.subplots()
            sns.barplot(
                x='Importance',
                y='Feature',
                data=feature_importance_df,
                ax=ax,
                palette='coolwarm')
            ax.set_title('Feature Importance')
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Features')
            st.pyplot(fig)
            print("Feature importance chart plotted.")
        else:
            st.warning("No features were deemed important by the model. Please check your dataset and model parameters. Ensure that the features are correctly selected and the target variable is appropriate for the task.")
            print("Warning: No important features identified.")

    else:
        st.info("Please configure the parameters and click 'Train Model' to proceed.")
        print("Model training not initiated. Awaiting user input.")
