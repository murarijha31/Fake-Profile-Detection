import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fake Profile Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    pipeline = joblib.load(
        BASE_DIR / "models" / "fake_profile_pipeline.pkl"
    )

    features = joblib.load(
        BASE_DIR / "models" / "features.pkl"
    )

    return pipeline, features


pipeline, features = load_model()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    real_df = pd.read_csv(
        BASE_DIR / "data" / "users.csv"
    )

    fake_df = pd.read_csv(
        BASE_DIR / "data" / "fusers.csv"
    )

    real_df["label"] = 0
    fake_df["label"] = 1

    combined_df = pd.concat(
        [real_df, fake_df],
        ignore_index=True
    )

    return real_df, fake_df, combined_df


real_df, fake_df, df = load_data()


# =========================================================
# MODEL
# =========================================================

random_forest = pipeline.named_steps["model"]


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background-color: #0e1117;
    }


    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }


    .subtitle {
        font-size: 17px;
        color: #aab2c0;
        margin-bottom: 25px;
    }


    /* Metric cards */

    .metric-card {
        background-color: #171b24;
        border: 1px solid #2b3240;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.25);
    }


    .metric-title {
        color: #aab2c0;
        font-size: 14px;
        margin-bottom: 8px;
    }


    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }


    /* Section headings */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }


    /* Result cards */

    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 20px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #707887;
        margin-top: 50px;
        padding: 20px;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔐 Fake Profile")

st.sidebar.markdown(
    "### Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "🔎 Analyze Profile",
        "📊 Analytics",
        "🧠 Model Insights",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning based system "
    "for detecting potentially fake "
    "social-media profiles."
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🔐 Fake Profile Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered Social Media Risk Analysis'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="section-title">'
        '📊 System Overview'
        '</div>',
        unsafe_allow_html=True
    )

    total_profiles = len(df)

    genuine_profiles = len(
        df[df["label"] == 0]
    )

    fake_profiles = len(
        df[df["label"] == 1]
    )

    fake_percentage = (
        fake_profiles / total_profiles
    ) * 100


    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    👥 Total Profiles
                </div>

                <div class="metric-value">
                    {total_profiles:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    🟢 Genuine Profiles
                </div>

                <div class="metric-value">
                    {genuine_profiles:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    🔴 Fake Profiles
                </div>

                <div class="metric-value">
                    {fake_profiles:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    ⚠️ Fake Profile Rate
                </div>

                <div class="metric-value">
                    {fake_percentage:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # =====================================================
    # PROFILE DISTRIBUTION
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Profile Distribution'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    # Pie chart

    with col1:

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        counts = [
            genuine_profiles,
            fake_profiles
        ]

        labels = [
            "Genuine",
            "Fake"
        ]

        ax.pie(
            counts,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            "Genuine vs Fake Profiles"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # Bar chart

    with col2:

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.bar(
            labels,
            counts
        )

        ax.set_title(
            "Profile Count"
        )

        ax.set_ylabel(
            "Number of Profiles"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # =====================================================
    # PROJECT OBJECTIVE
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '🎯 Project Objective'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        The objective of this project is to develop a
        machine-learning based system capable of identifying
        potentially fake social-media profiles using
        profile characteristics and behavioral features.
        """
    )


# =========================================================
# ANALYZE PROFILE
# =========================================================

elif page == "🔎 Analyze Profile":

    st.markdown(
        '<div class="section-title">'
        '🔎 Profile Analyzer'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the profile information below to analyze the profile."
    )


    # =====================================================
    # INPUT SECTION
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        followers = st.number_input(
            "Followers",
            min_value=0,
            value=100,
            step=1
        )

        friends = st.number_input(
            "Following",
            min_value=0,
            value=200,
            step=1
        )

        statuses = st.number_input(
            "Posts",
            min_value=0,
            value=100,
            step=1
        )

        favourites = st.number_input(
            "Favourites",
            min_value=0,
            value=50,
            step=1
        )


    with col2:

        listed_count = st.number_input(
            "Listed Count",
            min_value=0,
            value=0,
            step=1
        )

        account_age = st.number_input(
            "Account Age (days)",
            min_value=1,
            value=1000,
            step=1
        )

        has_description = st.selectbox(
            "Has Description?",
            ["Yes", "No"]
        )

        has_location = st.selectbox(
            "Has Location?",
            ["Yes", "No"]
        )

        has_url = st.selectbox(
            "Has URL?",
            ["Yes", "No"]
        )


    st.write("")


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    analyze = st.button(
        "🔍 Analyze Profile",
        type="primary",
        use_container_width=True
    )


    if analyze:

        # =================================================
        # FEATURE ENGINEERING
        # =================================================

        followers_friends_ratio = (
            followers / (friends + 1)
        )

        statuses_per_day = (
            statuses / (account_age + 1)
        )

        favourites_per_day = (
            favourites / (account_age + 1)
        )


        # =================================================
        # CREATE INPUT DATAFRAME
        # =================================================

        input_data = pd.DataFrame([
            {
                "followers_count":
                    followers,

                "friends_count":
                    friends,

                "statuses_count":
                    statuses,

                "favourites_count":
                    favourites,

                "listed_count":
                    listed_count,

                "followers_friends_ratio":
                    followers_friends_ratio,

                "account_age_days":
                    account_age,

                "statuses_per_day":
                    statuses_per_day,

                "favourites_per_day":
                    favourites_per_day,

                "has_description":
                    1 if has_description == "Yes" else 0,

                "has_location":
                    1 if has_location == "Yes" else 0,

                "has_url":
                    1 if has_url == "Yes" else 0
            }
        ])


        # =================================================
        # FEATURE ORDER
        # =================================================

        input_data = input_data[features]


        # =================================================
        # PREDICTION
        # =================================================

        prediction = pipeline.predict(
            input_data
        )[0]


        # Find probability of class 1 = Fake

        class_list = list(
            random_forest.classes_
        )

        fake_index = class_list.index(1)


        probability = pipeline.predict_proba(
            input_data
        )[0][fake_index]


        st.divider()


        # =================================================
        # PREDICTION RESULT
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🎯 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )


        if prediction == 1:

            st.error(
                "⚠️ POTENTIALLY SUSPICIOUS PROFILE"
            )

        else:

            st.success(
                "✅ LIKELY GENUINE PROFILE"
            )


        # =================================================
        # PROBABILITY
        # =================================================

        st.metric(
            "Predicted Fake-Class Probability",
            f"{probability * 100:.2f}%"
        )


        st.caption(
            "This is a model-based estimate and should not "
            "be treated as definitive proof that a profile is fake."
        )


        # =================================================
        # RISK LEVEL
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '⚠️ Risk Assessment'
            '</div>',
            unsafe_allow_html=True
        )


        if probability >= 0.75:

            st.error(
                "🔴 HIGH RISK"
            )

        elif probability >= 0.40:

            st.warning(
                "🟡 MEDIUM RISK"
            )

        else:

            st.success(
                "🟢 LOW RISK"
            )


        # =================================================
        # PROFILE INDICATORS
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📌 Profile Indicators'
            '</div>',
            unsafe_allow_html=True
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Follower / Following",
                f"{followers_friends_ratio:.2f}"
            )


        with c2:

            st.metric(
                "Posts / Day",
                f"{statuses_per_day:.2f}"
            )


        with c3:

            st.metric(
                "Favourites / Day",
                f"{favourites_per_day:.2f}"
            )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-title">'
        '📊 Profile Analytics'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # FOLLOWERS
    # =====================================================

    st.subheader(
        "Followers Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        real_df["followers_count"].dropna(),
        bins=40,
        alpha=0.6,
        label="Genuine"
    )

    ax.hist(
        fake_df["followers_count"].dropna(),
        bins=40,
        alpha=0.6,
        label="Fake"
    )

    ax.set_xlabel(
        "Followers"
    )

    ax.set_ylabel(
        "Number of Profiles"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # =====================================================
    # FOLLOWING
    # =====================================================

    st.subheader(
        "Following Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        real_df["friends_count"].dropna(),
        bins=40,
        alpha=0.6,
        label="Genuine"
    )

    ax.hist(
        fake_df["friends_count"].dropna(),
        bins=40,
        alpha=0.6,
        label="Fake"
    )

    ax.set_xlabel(
        "Following"
    )

    ax.set_ylabel(
        "Number of Profiles"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # =====================================================
    # POSTS
    # =====================================================

    st.subheader(
        "Posts Distribution"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        real_df["statuses_count"].dropna(),
        bins=40,
        alpha=0.6,
        label="Genuine"
    )

    ax.hist(
        fake_df["statuses_count"].dropna(),
        bins=40,
        alpha=0.6,
        label="Fake"
    )

    ax.set_xlabel(
        "Posts"
    )

    ax.set_ylabel(
        "Number of Profiles"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# =========================================================
# MODEL INSIGHTS
# =========================================================

elif page == "🧠 Model Insights":

    st.markdown(
        '<div class="section-title">'
        '🧠 Model Insights'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    importance = pd.DataFrame(
        {
            "Feature": features,

            "Importance":
                random_forest.feature_importances_
        }
    )


    importance = importance.sort_values(
        by="Importance",
        ascending=True
    )


    # =====================================================
    # FEATURE IMPORTANCE CHART
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )


    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # =====================================================
    # FEATURE TABLE
    # =====================================================

    st.subheader(
        "Feature Importance Ranking"
    )


    importance_display = importance.sort_values(
        by="Importance",
        ascending=False
    ).copy()


    importance_display["Importance"] = (
        importance_display["Importance"]
        .round(4)
    )


    st.dataframe(
        importance_display,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-title">'
        'ℹ️ About the Project'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        ### 🔐 Fake Profile Detection & Social Media Risk Analysis

        This project uses machine learning to identify
        potentially fake social-media profiles based on
        profile characteristics and behavioral features.

        ### 🤖 Machine Learning Model

        The system uses a Random Forest classification model.

        The final application uses a Scikit-learn Pipeline
        containing preprocessing and the machine-learning model.

        ### 📊 Features

        The system uses features such as:

        - Followers count
        - Following count
        - Posts count
        - Favourites count
        - Listed count
        - Followers / Following ratio
        - Account age
        - Posts per day
        - Favourites per day
        - Description availability
        - Location availability
        - URL availability

        ### 🌐 Application

        The Streamlit dashboard allows users to:

        - Analyze profile characteristics
        - Generate a model prediction
        - View predicted fake-class probability
        - View profile indicators
        - Explore dataset analytics
        - Explore model feature importance

        ### ⚠️ Limitation

        The prediction is a machine-learning estimate based
        on the dataset used for training. It should not be
        treated as definitive proof that a real social-media
        account is fake.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    🔐 Fake Profile Detection System

    <br><br>

    Machine Learning • Python • Scikit-learn • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)