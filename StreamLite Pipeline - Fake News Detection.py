# %% [markdown]
# ### ALL - IN - ONE (PYTHON WEB APP) FRAMEWORK USING STREAMLIT

# %%
# Load libraries for the process
import gc
import time
import traceback
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from scipy.sparse import hstack
from transformers import pipeline

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM DARK THEME CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Veritas AI | Misinformation Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Glassmorphism & High-Contrast CSS
st.markdown(
    """
<style>
    /* Dark Theme Core Base */
    .stApp {
        background-color: #0d0f17 !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Text & Label High-Contrast Adjustments */
    .stApp p, .stApp span, .stApp label, .stApp div, .stMarkdown {
        color: #f8fafc !important;
    }

    /* Sidebar Text Styling */
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Gradient Hero Text */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8 !important;
        text-align: center;
        margin-bottom: 2.5rem;
    }

    /* Glassmorphism Card Styling */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }

    .glass-card h3, .glass-card h2, .glass-card h1 {
        color: #ffffff !important;
    }

    /* Custom Metric Badges */
    .metric-badge {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Form Controls Contrast & Styling */
    .stTextInput label, .stTextArea label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    .stTextArea textarea, .stTextInput input {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-size: 0.95rem;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.4) !important;
    }

    /* Metric Visual Customization */
    div[data-testid="stMetricValue"] {
        color: #818cf8 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    /* Primary AI Action Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(168, 85, 247, 0.6) !important;
    }

    /* Sidebar Background Styling */
    section[data-testid="stSidebar"] {
        background-color: #07090e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session State Initialization
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------
# 2. CACHED MODEL & TRANSFORMER PIPELINE LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_ml_components():
    fake_news_model = joblib.load("fake_news_detection_model")
    tfidf_weights = joblib.load("tfidf_vectorizer")
    onehot_weights = joblib.load("one_hot_encoder")
    scaler_weights = joblib.load("scaler.pkl")
    return fake_news_model, tfidf_weights, onehot_weights, scaler_weights


@st.cache_resource
def load_nlp_pipelines():
    # Lightweight Sentiment Analysis (~250MB RAM)
    sentiment_pipe = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",
        truncation=True,
        max_length=512,
    )
    
    # Lightweight Emotion Detection (~260MB RAM)
    emotion_pipe = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        framework="pt",
        truncation=True,
        max_length=512,
    )

    # Force Python Garbage Collection
    gc.collect()

    return sentiment_pipe, emotion_pipe


# Initialization Handler with Explicit Diagnostic Errors
weights_loaded = True
init_error_msg = ""

try:
    fake_news_model, tfidf_weights, onehot_weights, scaler_weights = (
        load_ml_components()
    )
except Exception as e:
    weights_loaded = False
    init_error_msg += f"**ML Components Error:** {e}\n\n"

try:
    sentiment_pipe, emotion_pipe = load_nlp_pipelines()
except Exception as e:
    weights_loaded = False
    init_error_msg += f"**NLP Pipelines Error:** {e}\n\n"

if not weights_loaded:
    st.error(
        "⚠️ System Assets Initialization Warning\n\n"
        f"{init_error_msg}"
        "Please verify that `fake_news_detection_model`, `tfidf_vectorizer`, `one_hot_encoder`, "
        "and `scaler.pkl` exist in your working directory."
    )

# ---------------------------------------------------------
# 3. EXTERNAL FACT-CHECKING API FUNCTION
# ---------------------------------------------------------
def check_google_factcheck(user_title, api_key):
    """Passes concise title queries to Google Fact Check Tools API."""
    if (
        not user_title
        or not api_key
        or api_key == "YOUR_GOOGLE_API_KEY_HERE"
    ):
        return 0, None

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": user_title.strip()[:100],
        "languageCode": "en",
        "key": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=4)
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            if claims:
                rating = (
                    claims[0]
                    .get("claimReview", [{}])[0]
                    .get("textualRating", "")
                )
                return 1, rating
    except Exception:
        pass

    return 0, None


# ---------------------------------------------------------
# 4. INFERENCE ENGINE WITH OVERRIDE LOGIC
# ---------------------------------------------------------
def predict_article(user_title, user_text, api_key):
    fully_combined_article = f"{user_title} {user_text}".strip()

    # Step A: Dynamic NLP Inference with Capitalized Formatting
    try:
        sentiment_raw = sentiment_pipe(fully_combined_article[:512])[0]["label"]
        sentiment_res = str(sentiment_raw).capitalize()
    except Exception:
        sentiment_res = "Neutral"

    try:
        emotion_raw = emotion_pipe(fully_combined_article[:512])[0]["label"]
        emotion_res = str(emotion_raw).capitalize()
    except Exception:
        emotion_res = "Neutral"

    # Step B: Live API Lookup
    fact_flag, api_rating = check_google_factcheck(user_title, user_text, api_key)

    # Step C: Metadata Calculations
    char_count = len(fully_combined_article)
    word_count = len(fully_combined_article.split())
    avg_word_len = char_count / word_count if word_count > 0 else 0.0

    # Step D: Feature Transformations
    # 1. TF-IDF
    X_tfidf = tfidf_weights.transform([fully_combined_article])

    # 2. Numerical Features
    num_raw = [[char_count, word_count, avg_word_len]]
    if hasattr(scaler_weights, "feature_names_in_"):
        num_df = pd.DataFrame(num_raw, columns=scaler_weights.feature_names_in_)
        X_num = scaler_weights.transform(num_df)
    else:
        X_num = scaler_weights.transform(np.array(num_raw))

    # 3. Categorical Features (One-Hot Encoded)
    cat_raw = [[sentiment_res, emotion_res]]
    if hasattr(onehot_weights, "feature_names_in_"):
        cat_df = pd.DataFrame(cat_raw, columns=onehot_weights.feature_names_in_)
        X_cat = onehot_weights.transform(cat_df)
    else:
        X_cat = onehot_weights.transform(np.array(cat_raw))

    # 4. Fact Flag
    X_fact = np.array([[fact_flag]])

    # ---------------------------------------------------------
    # MATCHING TRAINING SEQUENCE:
    # hstack([TFIDF, NUMERICAL, CATEGORICAL, FACT_FLAG])
    # ---------------------------------------------------------
    X_meta = np.hstack([X_num, X_cat, X_fact])
    X_final = hstack([X_tfidf, X_meta])

    # Step E: Model Statistical Prediction
    raw_pred = fake_news_model.predict(X_final)[0]
    raw_prob = fake_news_model.predict_proba(X_final)[0][1]

    # Step F: Fact-Check Database Rule Override
    final_prediction = raw_pred
    final_prob = raw_prob
    override_applied = False

    if fact_flag == 1 and api_rating:
        rating_lower = api_rating.lower()
        if any(
            term in rating_lower
            for term in [
                "false",
                "fake",
                "incorrect",
                "misleading",
                "pants on fire",
            ]
        ):
            final_prediction = 0  # Misinformation / Fake
            final_prob = 0.05
            override_applied = True
        elif any(
            term in rating_lower
            for term in ["true", "accurate", "correct", "verified"]
        ):
            final_prediction = 1  # Verified / Real
            final_prob = 0.95
            override_applied = True

    return {
        "prediction": final_prediction,
        "probability": final_prob,
        "sentiment": sentiment_res,
        "emotion": emotion_res,
        "fact_flag": fact_flag,
        "api_rating": api_rating,
        "override_applied": override_applied,
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_len": avg_word_len,
    }
# ---------------------------------------------------------
# 5. SIDEBAR ARCHITECTURE & BENCHMARKS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<span class='metric-badge'>v2.4 Production Engine</span>",
        unsafe_allow_html=True,
    )
    st.title("🛡️ Engine Specs")

    st.markdown("""
    **Model Architecture:**
    - Regularized Logistic Regression ($L_1$ Lasso)
    - TF-IDF Vectorizer (10,000 N-gram terms)
    - Sentiment (*DistilBERT*) & Emotion (*DistilRoBERTa*)
    - Standard Scaled Metadata Features
    
    ---
    **Model Benchmarks:**
    - **Accuracy:** 98.97%
    - **Recall (Fake News):** 99.33%
    - **Precision:** 98.67%
    - **ROC-AUC:** 0.9982
    """)
    st.divider()
    st.caption("Powered by Scikit-Learn, PyTorch & Streamlit")

# ---------------------------------------------------------
# 6. HERO SECTION
# ---------------------------------------------------------
st.markdown(
    "<h1 class='hero-title'>Veritas AI Detector</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='hero-subtitle'>Enterprise Misinformation Detection & Real-Time Content Verification Engine</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 7. MAIN INTERFACE LAYOUT
# ---------------------------------------------------------
col_input, col_output = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📄 Input Article Analysis")

    user_title = st.text_input(
        "Article Title (Optional):",
        placeholder="e.g., Breaking News: Major Event Announced...",
    )
    user_text = st.text_area(
        "Article Body Text:",
        height=240,
        placeholder="Paste article body text here...",
    )

    analyze_btn = st.button("⚡ Run Verification Engine")
    st.markdown("</div>", unsafe_allow_html=True)

with col_output:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📊 Verification Report")

    if analyze_btn:
        if not user_text.strip() and not user_title.strip():
            st.warning(
                "Please provide an article title or body text to evaluate."
            )
        elif not weights_loaded:
            st.error(
                "Cannot run prediction: Essential model weights or components failed to load on startup. "
                "Check the top banner error for details."
            )
        else:
            with st.spinner(
                "Running NLP models (DistilBERT & Emotion) and querying external databases..."
            ):
                api_key = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", None)
                try:
                    st.session_state.results = predict_article(
                        user_title=user_title,
                        user_text=user_text,
                        api_key=api_key,
                    )
                except Exception as eval_err:
                    st.error(f"Inference Engine Error: {eval_err}")

    if st.session_state.results is not None:
        res = st.session_state.results

        # Metric Visualization Output
        if res["prediction"] == 1:
            st.success("### ✅ VERIFIED: Likely Authentic News")
            st.progress(float(res["probability"]))
            st.write(
                f"**Credibility Score:** `{res['probability'] * 100:.2f}%` Credibility Rating"
            )
        else:
            st.error("### ⚠️ FLAG: Likely Misinformation")
            st.progress(float(1 - res["probability"]))
            st.write(
                f"**Risk Score:** `{(1 - res['probability']) * 100:.2f}%` Fake Probability"
            )

        if res["override_applied"]:
            st.info(
                "ℹ️ Output overridden by explicit match in external Fact-Check database."
            )

        st.divider()

        # Detected NLP Metrics
        st.markdown("**Automated Transformer Feature Extraction:**")
        ncol1, ncol2 = st.columns(2)
        ncol1.info(f"**Sentiment:** {res['sentiment'].capitalize()}")
        ncol2.info(f"**Dominant Emotion:** {res['emotion'].capitalize()}")

        # Verification Status
        st.markdown("**Fact-Check Registry Status:**")
        if res["fact_flag"] == 1:
            st.success(
                f"Fact Match Found! Official Rating: '{res['api_rating']}'"
            )
        else:
            st.warning("No direct match in Google Fact Check registry.")

        # Structural Metadata
        st.markdown("**Structural Article Metadata:**")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Word Count", f"{res['word_count']}")
        mcol2.metric("Char Count", f"{res['char_count']}")
        mcol3.metric("Avg Word Length", f"{res['avg_word_len']:.2f}")

    else:
        st.info(
            "Paste an article on the left and click **Run Verification Engine** to generate an assessment."
        )

    st.markdown("</div>", unsafe_allow_html=True)
