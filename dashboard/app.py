
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# -----------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------
# 2. LOAD PROJECT FILES
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_sales.csv"
MODEL_PATH = BASE_DIR / "models" / "best_sales_model.pkl"
RESULTS_PATH = BASE_DIR / "outputs" / "model_results.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
results = load_results()
model = load_model()

# -----------------------------------------
# 3. SIDEBAR
# -----------------------------------------

st.sidebar.title("📊 Sales Analytics")
st.sidebar.write("Sales Prediction Using Python")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Advertising Analysis",
        "Sales Prediction",
        "Model Performance",
        "Business Insights"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Built using Python, Streamlit and Machine Learning")

# -----------------------------------------
# 4. DASHBOARD OVERVIEW
# -----------------------------------------

if page == "Dashboard Overview":

    st.title("📊 Sales Prediction Dashboard")
    st.write("Advertising Performance and Sales Analysis")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Average Sales", round(df["Sales"].mean(), 2))
    col3.metric("Highest Sales", round(df["Sales"].max(), 2))
    col4.metric("Average TV Advertising", round(df["TV"].mean(), 2))

    st.divider()

    st.subheader("Advertising Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="Sales",
            nbins=20,
            title="Sales Distribution",
            color_discrete_sequence=["#3498db"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df,
            x="TV",
            y="Sales",
            title="TV Advertising vs Sales",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# -----------------------------------------
# 5. ADVERTISING ANALYSIS
# -----------------------------------------

elif page == "Advertising Analysis":

    st.title("📈 Advertising Analysis")

    st.write("Explore the relationship between advertising channels and sales.")

    selected_channel = st.selectbox(
        "Select Advertising Channel",
        ["TV", "Radio", "Newspaper"]
    )

    fig = px.scatter(
        df,
        x=selected_channel,
        y="Sales",
        trendline="ols",
        title=f"{selected_channel} Advertising vs Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")

    correlation = df.corr(numeric_only=True)

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Advertising and Sales Correlation"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Advertising Channel Statistics")

    summary = df[["TV", "Radio", "Newspaper"]].describe().T
    st.dataframe(summary, use_container_width=True)

# -----------------------------------------
# 6. SALES PREDICTION
# -----------------------------------------

elif page == "Sales Prediction":

    st.title("🔮 Sales Prediction")
    st.write("Enter advertising budgets to predict expected sales.")

    st.info(
        "Enter values using the same scale as the original dataset. "
        "Predictions are most reliable within the ranges of the training data."
    )

    col1, col2 = st.columns(2)

    with col1:
        tv = st.number_input(
            "TV Advertising Budget",
            min_value=0.7,
            max_value=296.4,
            value=200.0,
            step=1.0
        )

        radio = st.number_input(
            "Radio Advertising Budget",
            min_value=0.0,
            max_value=49.6,
            value=40.0,
            step=1.0
        )

        newspaper = st.number_input(
            "Newspaper Advertising Budget",
            min_value=0.3,
            max_value=114.0,
            value=30.0,
            step=1.0
        )

        predict_button = st.button(
            "Predict Sales",
            type="primary",
            use_container_width=True
        )

    with col2:
        st.subheader("Advertising Inputs")

        input_df = pd.DataFrame({
            "Advertising Channel": ["TV", "Radio", "Newspaper"],
            "Budget": [tv, radio, newspaper]
        })

        fig = px.bar(
            input_df,
            x="Advertising Channel",
            y="Budget",
            title="Your Advertising Budget"
        )

        st.plotly_chart(fig, use_container_width=True)

    if predict_button:

        input_data = pd.DataFrame({
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        })

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Sales",
            value=f"{prediction:.2f}"
        )

        st.success(
            f"The model predicts approximately {prediction:.2f} sales units."
        )

# -----------------------------------------
# 7. MODEL PERFORMANCE
# -----------------------------------------

elif page == "Model Performance":

    st.title("🤖 Machine Learning Model Performance")

    st.write("Comparison of the three trained regression models.")

    st.dataframe(
        results.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True
    )

    fig = px.bar(
        results,
        x="Model",
        y="R2 Score",
        color="Model",
        title="Model Comparison Based on R² Score",
        text_auto=".3f"
    )

    st.plotly_chart(fig, use_container_width=True)

    best_model = results.sort_values(
        by=["R2 Score", "MAE"],
        ascending=[False, True]
    ).iloc[0]

    st.success(
        f"Best Model: {best_model['Model']} "
        f"(R² Score: {best_model['R2 Score']:.4f})"
    )

    st.caption(
        "Scores are from one 80/20 train-test split. "
        "They are not a guarantee of future prediction performance."
    )

# -----------------------------------------
# 8. BUSINESS INSIGHTS
# -----------------------------------------

elif page == "Business Insights":

    st.title("💡 Business Insights")

    st.write("Key observations from the advertising dataset.")

    correlation = df.corr(numeric_only=True)["Sales"].drop("Sales")

    strongest_channel = correlation.abs().idxmax()
    strongest_value = correlation[strongest_channel]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Average Sales",
            f"{df['Sales'].mean():.2f}"
        )

    with col2:
        st.metric(
            "Strongest Correlated Channel",
            strongest_channel
        )

    st.subheader("Advertising and Sales Relationships")

    for channel, value in correlation.items():

        st.write(
            f"**{channel}:** Correlation with Sales = {value:.3f}"
        )

    st.subheader("Key Observations")

    st.write(
        f"1. {strongest_channel} has the strongest absolute "
        "correlation with Sales among the three advertising channels."
    )

    st.write(
        f"2. Average sales across the dataset are "
        f"{df['Sales'].mean():.2f} units."
    )

    st.write(
        "3. The Random Forest model achieved the highest R² "
        "score in our test-set comparison."
    )

    st.warning(
        "Correlation does not prove causation. "
        "This dataset does not include advertising costs, profit, "
        "dates or customer segments, so ROI and time-based trends "
        "cannot be calculated reliably."
    )

# -----------------------------------------
# 9. FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "Sales Prediction Using Python | "
    "Python • Pandas • Scikit-learn • Streamlit • Plotly"
)
