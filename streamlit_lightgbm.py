# streamlit_lightgbm.py

"""
This module provides a Streamlit application for training LightGBM models.
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from flaml.automl.data import load_openml_dataset

print(lgb.__version__)

# Initialize session state variables
if 'show_full_dataset' not in st.session_state:
    st.session_state.show_full_dataset = False
if 'SAMPLE_DATASET_LOADED' not in st.session_state:
    st.session_state.SAMPLE_DATASET_LOADED = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = False

# Set Streamlit page configuration
st.set_page_config(
    page_title="LightGBM Model Trainer",
    page_icon="🌟",
    layout="wide"
)

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
st.write(
    "Upload your dataset to train a LightGBM model with configurable parameters!"
)

# Sidebar configuration
st.sidebar.header("Data Configuration")

def update_preview():
    """Toggle the dataset preview between full and partial."""
    st.session_state.show_full_dataset = not st.session_state.show_full_dataset

# Example dataset
st.write("### Example Dataset")
st.write(
    "Download [houses dataset](https://www.openml.org/d/537) from OpenML. "
    "The task is to predict the median price of the house in the region based on "
    "demographic composition and the state of the housing market in the region."
)

# Test dataset loading
st.write("### Load Test Dataset")
if st.button("Load Example Dataset"):
    with st.spinner('Loading dataset...'):
        try:
            dataset = load_openml_dataset(dataset_id=537, data_dir="./")
            if dataset is None:
                raise ValueError(
                    "Failed to load dataset. The dataset returned None. "
                    "Please check the dataset ID and try again."
                )
            X_train, X_test, y_train, y_test = dataset
            if X_train is None or y_train is None:
                raise ValueError(
                    "Failed to load dataset. The dataset returned None. "
                    "Please check the dataset ID and try again."
                )
            st.success("Dataset loaded successfully!")
            y_train = y_train.reset_index(drop=True)
            st.session_state.data = pd.concat([X_train, y_train], axis=1)
            st.session_state.SAMPLE_DATASET_LOADED = True
            st.write(
                "Dataset loaded: X_train shape:", X_train.shape,
                "y_train shape:", y_train.shape
            )
        except ValueError as e:
            st.error(
                f"Error loading dataset: {e}. Please verify your internet connection "
                "or try a different dataset."
            )
            st.write("Error loading dataset:", e)
            st.stop()

st.checkbox(
    "Show full dataset",
    value=st.session_state.show_full_dataset,
    on_change=update_preview
)

st.write("Preview option selected:", st.session_state.show_full_dataset)

# Upload dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        st.session_state.data = pd.read_csv(uploaded_file)
        if st.session_state.data.empty:
            raise ValueError(
                "The uploaded CSV file is empty. Please upload a valid dataset."
            )
        st.write("Uploaded dataset shape:", st.session_state.data.shape)
    except ValueError as e:
        st.error(f"Error reading the uploaded file: {e}")
        st.write("Error reading the uploaded file:", e)
        st.stop()

if st.session_state.data is not None or st.session_state.SAMPLE_DATASET_LOADED:
    st.write("### Dataset Preview")
    if st.session_state.show_full_dataset:
        st.dataframe(st.session_state.data, use_container_width=True)
    else:
        st.dataframe(st.session_state.data.head(), use_container_width=True)

    if len(st.session_state.data.columns) < 2:
        st.error(
            "The dataset must contain at least two columns: one target and at least one feature."
        )
        st.stop()

    # Add checkbox to clean data after dataset is loaded
    st.sidebar.checkbox(
        "Clean data (remove NaN values) before training",
        value=st.session_state.clean_data,
        key='clean_data_checkbox'
    )

    # Update clean_data state
    st.session_state.clean_data = st.session_state.clean_data_checkbox

    st.session_state.target_column = st.sidebar.selectbox(
        "Select the target column",
        options=st.session_state.data.columns,
        index=st.session_state.data.columns.get_loc(st.session_state.target_column) if st.session_state.target_column in st.session_state.data.columns else 0,
        key='target_column_select'
    )
    st.session_state.feature_columns = st.sidebar.multiselect(
        "Select the feature columns",
        options=st.session_state.data.columns,
        default=list(st.session_state.data.columns.drop(st.session_state.target_column)),
        key='feature_columns_select'
    )
    if len(st.session_state.feature_columns) == 0:
        st.error("Please select at least one feature column.")
        st.stop()
    st.write("Target column:", st.session_state.target_column)
    st.write("Feature columns:", st.session_state.feature_columns)

    def get_objective_and_metric(target_values):
        """Determine the LightGBM objective and metric based on the target variable."""
        if pd.api.types.is_float_dtype(target_values):
            return 'regression', 'rmse'
        elif target_values.nunique() > 2:
            return 'multiclass', 'multi_logloss'
        else:
            return 'binary', 'binary_error'

    st.sidebar.header("LightGBM Parameters Configuration")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.05)
    num_leaves = st.sidebar.slider("Number of Leaves", 10, 100, 31)
    n_estimators = st.sidebar.slider("Number of Boosting Rounds", 50, 500, 100)
    feature_fraction = st.sidebar.slider("Feature Fraction", 0.1, 1.0, 0.9)
    early_stopping_rounds = st.sidebar.slider(
        "Early Stopping Rounds", 5, 50, 10
    )
    default_threshold = st.sidebar.slider(
        "Default Classification Threshold", 0.0, 1.0, 0.5
    )
    importance_type = st.sidebar.selectbox(
        "Select Feature Importance Type", ['split', 'gain'], index=0
    )
    st.write(
        "Model parameters: learning_rate=", learning_rate,
        ", num_leaves=", num_leaves,
        ", n_estimators=", n_estimators,
        ", feature_fraction=", feature_fraction,
        ", early_stopping_rounds=", early_stopping_rounds
    )

    if st.button("Train Model"):
        if st.session_state.data is None:
            st.error("No dataset loaded. Please upload a dataset or load the example dataset.")
            st.stop()

        X = st.session_state.data[st.session_state.feature_columns]
        y = st.session_state.data[st.session_state.target_column]
        st.write("Training data shapes: X=", X.shape, ", y=", y.shape)

        # Clean data if checkbox is selected
        if st.session_state.clean_data:
            st.write("Cleaning data: removing NaN values.")
            data_before = X.shape[0]
            X = X.dropna()
            y = y.loc[X.index]
            data_after = X.shape[0]
            st.write(f"Removed {data_before - data_after} rows with NaN values.")

        if not st.session_state.SAMPLE_DATASET_LOADED:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            st.write(
                "Split data: X_train shape:", X_train.shape,
                "X_test shape:", X_test.shape,
                "y_train shape:", y_train.shape,
                "y_test shape:", y_test.shape
            )
        else:
            X_train = X
            y_train = y
            X_test, y_test = None, None

        if (
            X_train.isnull().values.any() or
            y_train.isnull().values.any()
        ):
            st.error(
                "The dataset contains NaN or missing values after cleaning. "
                "Please check your data."
            )
            st.write("Error: Dataset contains NaN or missing values.")
            st.stop()

        try:
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = None
            if X_test is not None and y_test is not None:
                test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            st.write("LightGBM datasets prepared.")
        except ValueError as e:
            st.error(f"Error in dataset preparation: {e}")
            st.write("Error in dataset preparation:", e)
            st.stop()

        objective, metric = get_objective_and_metric(y)
        params = {
            'objective': objective,
            'metric': metric,
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction
        }
        st.write("LightGBM parameters:", params)

        with st.spinner('Training the model...'):
            try:
                valid_sets = [train_data]
                if test_data is not None:
                    valid_sets.append(test_data)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=n_estimators,
                    valid_sets=valid_sets,
                    callbacks=[
                        lgb.log_evaluation(period=10),
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds)
                    ]
                )
                st.write("Model training completed.")
            except ValueError as e:
                st.error(f"Error during model training: {e}")
                st.write("Error during model training:", e)
                st.stop()

        if X_test is not None:
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)
            st.write("Predictions made.")
            if pd.api.types.is_float_dtype(y):
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                st.success(f"Model RMSE: {rmse:.4f}")
                st.write("Model RMSE:", rmse)
            else:
                y_pred_binary = (y_pred >= default_threshold).astype(int)
                accuracy = accuracy_score(y_test, y_pred_binary)
                st.success(f"Model Accuracy: {accuracy:.4f}")
                st.write("Model Accuracy:", accuracy)
        else:
            st.write("Test data not available for evaluation.")

        feature_importance = model.feature_importance(
            importance_type=importance_type
        )
        feature_importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(feature_importance_df, use_container_width=True)
        st.write("Feature importance calculated.")

        if not feature_importance_df.empty:
            st.write("### Feature Importance Chart")
            fig, ax = plt.subplots()
            sns.barplot(
                x='Importance', y='Feature',
                data=feature_importance_df,
                ax=ax,
                hue='Feature',
                palette='coolwarm',
                legend=False
            )
            ax.set_title('Feature Importance')
            ax.set_xlabel('Feature Importance')
            ax.set_ylabel('Features')
            st.pyplot(fig)
            st.write("Feature importance chart plotted.")
        else:
            st.warning(
                "No features were deemed important by the model. "
                "Please check your dataset and model parameters."
            )
            st.write("Warning: No important features identified.")
    else:
        st.info(
            "Please configure the parameters and click 'Train Model' to proceed."
        )
        st.write("Model training not initiated. Awaiting user input.")