import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Iris K-Means Clustering", page_icon="🌸", layout="wide"
)

# -------------------------------------------------
# Custom CSS
# -------------------------------------------------
st.markdown(
    """
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #ddd;
    }

    .result-text {
        font-size: 28px;
        font-weight: 700;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Load Pickle Files
# -------------------------------------------------
MODEL_FILE = "kmeans_iris_model.pkl"
SCALER_FILE = "iris_scaler.pkl"
FEATURES_FILE = "iris_features.pkl"
MAPPING_FILE = "cluster_to_species.pkl"

required_files = [MODEL_FILE, SCALER_FILE, FEATURES_FILE, MAPPING_FILE]

missing_files = [f for f in required_files if not os.path.exists(f)]

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    '<div class="main-title">🌸 Iris K-Means Clustering</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Unsupervised Machine Learning using K-Means Clustering</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Check Pickle Files
# -------------------------------------------------
if missing_files:
    st.error("⚠️ Required pickle files are missing!")

    st.write("Please keep these files in the same folder as app.py:")

    for file in required_files:
        if file in missing_files:
            st.error(f"❌ {file}")
        else:
            st.success(f"✅ {file}")

    st.stop()

# -------------------------------------------------
# Load Models
# -------------------------------------------------
try:
    with open(MODEL_FILE, "rb") as file:
        kmeans = pickle.load(file)

    with open(SCALER_FILE, "rb") as file:
        scaler = pickle.load(file)

    with open(FEATURES_FILE, "rb") as file:
        features = pickle.load(file)

    with open(MAPPING_FILE, "rb") as file:
        cluster_to_species = pickle.load(file)

except Exception as e:
    st.error(f"Error loading pickle files: {e}")
    st.stop()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("⚙️ Model Information")

st.sidebar.write("**Algorithm:** K-Means Clustering")
st.sidebar.write(f"**Number of Clusters:** {kmeans.n_clusters}")
st.sidebar.write("**Initialization:** K-Means++")
st.sidebar.write("**Random State:** 42")

st.sidebar.markdown("---")
st.sidebar.write("### Features")

for feature in features:
    st.sidebar.write(f"• {feature}")

# -------------------------------------------------
# Main Layout
# -------------------------------------------------
col1, col2 = st.columns([1, 1])

# -------------------------------------------------
# Input Section
# -------------------------------------------------
with col1:
    st.subheader("🌿 Enter Flower Measurements")

    st.write("Enter the four Iris flower measurements:")

    sepal_length = st.number_input(
        "Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.8, step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.3, step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.3, step=0.1
    )

    predict_button = st.button("🔍 Predict Cluster", use_container_width=True)

# -------------------------------------------------
# Input Preview
# -------------------------------------------------
with col2:
    st.subheader("📊 Input Preview")

    input_data = pd.DataFrame(
        {
            "Feature": [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width",
            ],
            "Value (cm)": [
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
            ],
        }
    )

    st.dataframe(input_data, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="info-box">
        <b>How it works:</b><br>
        1. Input values are collected.<br>
        2. The same StandardScaler used during training is applied.<br>
        3. The trained K-Means model predicts the cluster.<br>
        4. The cluster is mapped to its corresponding Iris species.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if predict_button:

    try:
        # ✅ FIX: Creating DataFrame with exact feature names seen at fit time
        # ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        sample_df = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=features,
        )

        # Scale input using the saved scaler with matching feature names
        input_scaled = scaler.transform(sample_df)

        # Predict cluster
        predicted_cluster = int(kmeans.predict(input_scaled)[0])

        # Map cluster to species
        predicted_species = cluster_to_species.get(
            predicted_cluster, "Unknown"
        )

        # Distance from cluster centers
        distances = kmeans.transform(input_scaled)[0]
        distance = float(distances[predicted_cluster])

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric("Predicted Cluster", f"Cluster {predicted_cluster}")

        with result_col2:
            st.metric("Mapped Species", predicted_species)

        with result_col3:
            st.metric("Cluster Distance", f"{distance:.4f}")

        st.markdown(
            f"""
            <div class="result-box">
                <div>🌸 Predicted Iris Species</div>
                <div class="result-text">{predicted_species}</div>
                <div>Cluster {predicted_cluster}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show scaled values
        with st.expander("🔬 View Standardized Input Values"):
            scaled_df = pd.DataFrame(input_scaled, columns=features)

            st.dataframe(
                scaled_df.round(4), use_container_width=True, hide_index=True
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------------------------------
# About Project
# -------------------------------------------------
st.markdown("---")

st.subheader("📌 About This Project")

about_col1, about_col2 = st.columns(2)

with about_col1:
    st.write("""
    **Iris K-Means Clustering** is an unsupervised machine learning
    project that groups Iris flowers into clusters based on their
    numerical measurements.
    """)

with about_col2:
    st.write("""
    The final model uses **K = 3**, with the four measurements:
    Sepal Length, Sepal Width, Petal Length and Petal Width.
    """)

st.caption("Built with Python • Scikit-learn • Pandas • NumPy • Streamlit")