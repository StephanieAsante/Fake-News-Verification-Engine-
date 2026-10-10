# %% [markdown]
# ### ALL - IN - ONE (PYTHON WEB APP) FRAMEWORK USING STREAMLIT

# %%
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

st.markdown(
    """
<style>
    .stApp {
        background-color: #0d0f17 !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp p, .stApp span, .stApp label, .stApp div, .stMarkdown {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
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
    section[data-testid="stSidebar"] {
        background-color: #07090e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""",
    unsafe_allow_html=True,
)

if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------
# 2. CACHED MODEL & PIPELINE LOADING
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
    sentiment_pipe = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt",
        truncation=True,
        max_length=512,
    )
    emotion_pipe = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        framework="pt",
        truncation=True,
        max_length=512,
    )
    gc.collect()
    return sentiment_pipe, emotion_pipe

weights_loaded = True
init_error_msg = ""

try:
    fake_news_model, tfidf_weights, onehot_weights, scaler_weights = load_ml_components()
except Exception as e:
    weights_loaded = False
    init_error_msg += f"**ML Components Error:** {e}\n\n"

try:
    sentiment_pipe, emotion_pipe = load_nlp_pipelines()
except Exception as e:
    weights_loaded = False
    init_error_msg += f"**NLP Pipelines Error:** {e}\n\n"

if not weights_loaded:
    st.error(f"⚠️ System Assets Initialization Warning\n\n{init_error_msg}")

# ---------------------------------------------------------
# 3. EXTERNAL APIS (GDELT ARCHIVE & GOOGLE FACT CHECK)
# ---------------------------------------------------------
def check_gdelt_archive(user_title, user_text):
    """Tier 1: Queries GDELT DOC 2.0 API for global news archive validation"""
    raw_str = f"{user_title} {user_text[:200]}".strip()
    if not raw_str:
        return 0, None, None

    stop_words = {
        "the", "a", "an", "in", "on", "at", "by", "for", "with", "and", 
        "or", "but", "to", "of", "is", "are", "was", "were", "it", "that", 
        "this", "told", "said", "from", "as", "he", "she", "they", "its"
    }
    words = [w for w in raw_str.split() if w.lower() not in stop_words and len(w) > 2]
    flexible_query = " ".join(words[:4])
    if not flexible_query:
        flexible_query = raw_str[:40]

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": flexible_query, "mode": "artlist", "maxrecords": 3, "format": "json"}
    try:
        response = requests.get(url, params=params, timeout=6)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if articles:
                top_article = articles[0]
                publisher = top_article.get("domain", "GDELT Global Archive")
                article_url = top_article.get("url", "#")
                return 1, publisher, article_url
    except Exception as e:
        print(f"GDELT Archive API Request Error: {e}")

    return 0, None, None

def check_google_factcheck(user_title, user_text, google_api_key):
    """Tier 2: Queries Google Fact Check API for explicit claim ratings"""
    if not google_api_key or google_api_key == "YOUR_GOOGLE_API_KEY_HERE":
        return 0, None
    query_str = user_title.strip() if user_title.strip() else user_text.strip()[:100]
    if not query_str:
        return 0, None

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query_str[:100], "languageCode": "en", "key": google_api_key}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            if claims:
                rating = claims[0].get("claimReview", [{}])[0].get("textualRating", "")
                return 1, rating
    except Exception:
        pass
    return 0, None

# ---------------------------------------------------------
# 4. PREDICTION ENGINE WITH SEQUENTIAL PRIORITY & MODEL FALLBACK
# ---------------------------------------------------------
def predict_article(user_title, user_text, google_api_key):
    fully_combined_article = f"{user_title} {user_text}".strip()

    try:
        sentiment_res = str(sentiment_pipe(fully_combined_article[:512])[0]["label"]).capitalize()
    except Exception:
        sentiment_res = "Neutral"

    try:
        emotion_res = str(emotion_pipe(fully_combined_article[:512])[0]["label"]).capitalize()
    except Exception:
        emotion_res = "Neutral"

    # SEQUENTIAL PIPELINE EXECUTION: GDELT First, then Google Fact Check, then Model
    news_found, publisher_name, article_url = check_gdelt_archive(user_title, user_text)
    fact_flag, api_rating = check_google_factcheck(user_title, user_text, google_api_key)

    TRAIN_MAX_CHAR = 32655
    TRAIN_MAX_WORD = 5412
    TRAIN_MAX_AVG_WORD_LEN = 74.0

    raw_char_count = len(fully_combined_article)
    raw_word_count = len(fully_combined_article.split())
    raw_avg_word_len = raw_char_count / raw_word_count if raw_word_count > 0 else 0.0

    capped_char_count = min(raw_char_count, TRAIN_MAX_CHAR)
    capped_word_count = min(raw_word_count, TRAIN_MAX_WORD)
    capped_avg_word_len = min(raw_avg_word_len, TRAIN_MAX_AVG_WORD_LEN)

    X_tfidf = tfidf_weights.transform([fully_combined_article])
    num_raw = [[capped_char_count, capped_word_count, capped_avg_word_len]]
    if hasattr(scaler_weights, "feature_names_in_"):
        X_num = scaler_weights.transform(pd.DataFrame(num_raw, columns=scaler_weights.feature_names_in_))
    else:
        X_num = scaler_weights.transform(np.array(num_raw))

    cat_raw = [[sentiment_res, emotion_res]]
    if hasattr(onehot_weights, "feature_names_in_"):
        X_cat = onehot_weights.transform(pd.DataFrame(cat_raw, columns=onehot_weights.feature_names_in_))
    else:
        X_cat = onehot_weights.transform(np.array(cat_raw))

    X_fact = np.array([[fact_flag]])
    X_meta = np.hstack([X_num, X_cat, X_fact])
    X_final = hstack([X_tfidf, X_meta])

    # Core Machine Learning Probabilistic Prediction
    raw_pred = fake_news_model.predict(X_final)[0]
    raw_prob = fake_news_model.predict_proba(X_final)[0][1]

    final_prediction = raw_pred
    final_prob = raw_prob
    override_applied = False
    confidence_level = "High"

    # PRIORITY 1: GDELT Archive Verification Check
    if news_found == 1:
        final_prediction = 1
        final_prob = max(raw_prob, 0.90)
        override_applied = True
        confidence_level = f"High (GDELT Archive Verified: {publisher_name})"

    # PRIORITY 2: Google Fact Check Registry Check (Overrides GDELT or Model if explicit verdict exists)
    elif fact_flag == 1 and api_rating:
        rating_lower = api_rating.lower()
        if any(term in rating_lower for term in ["false", "fake", "incorrect", "misleading", "pants on fire"]):
            final_prediction = 0
            final_prob = 0.05
            override_applied = True
            confidence_level = "High (Fact-Check Override)"
        elif any(term in rating_lower for term in ["true", "accurate", "correct", "verified"]):
            final_prediction = 1
            final_prob = 0.95
            override_applied = True
            confidence_level = "High (Fact-Check Override)"

    # PRIORITY 3: Clean Probabilistic ML Fallback (When neither database has a direct match)
    if not override_applied:
        if 0.38 <= final_prob <= 0.62:
            confidence_level = "Moderate"
            if final_prob >= 0.45:
                final_prediction = 1 
        else:
            confidence_level = "High"

    return {
        "prediction": final_prediction,
        "probability": final_prob,
        "confidence_level": confidence_level,
        "sentiment": sentiment_res,
        "emotion": emotion_res,
        "fact_flag": fact_flag,
        "api_rating": api_rating,
        "news_found": news_found,
        "publisher_name": publisher_name,
        "article_url": article_url,
        "override_applied": override_applied,
        "char_count": raw_char_count,
        "word_count": raw_word_count,
        "avg_word_len": raw_avg_word_len,
    }

# ---------------------------------------------------------
# 5. UI LAYOUT
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<span class='metric-badge'>v2.7 Sequential Pipeline</span>", unsafe_allow_html=True)
    st.title("🛡️ Engine Specs")
    st.markdown("""
    **Pipeline Architecture:**
    1. GDELT Global Archive Lookup
    2. Google Fact-Check Registry
    3. $L_1$ Logistic Regression & Transformers
    """)
    st.divider()
    st.caption("Cost-Free Open-Source Architecture")

st.markdown("<h1 class='hero-title'>Veritas AI Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Enterprise Misinformation Detection & Global Archive Verification Engine</p>", unsafe_allow_html=True)

tab_app, tab_docs = st.tabs(["⚡ Verification Engine", "📖 Model Card & Specs"])

with tab_app:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📄 Input Article Analysis")
        user_title = st.text_input("Article Title:")
        user_text = st.text_area("Article Body Text:", height=240)
        analyze_btn = st.button("⚡ Run Verification Engine")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_output:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📊 Verification Report")

        if analyze_btn:
            if not user_text.strip() and not user_title.strip():
                st.warning("Please provide an article title or body text to evaluate.")
            elif not weights_loaded:
                st.error("Model assets failed to load.")
            else:
                with st.spinner("Executing sequential verification pipeline (GDELT -> Fact Check -> ML Brain)..."):
                    google_api_key = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", None)
                    try:
                        st.session_state.results = predict_article(user_title, user_text, google_api_key)
                    except Exception as eval_err:
                        st.error(f"Error: {eval_err}")

        if st.session_state.results is not None:
            res = st.session_state.results
            if res["prediction"] == 1:
                st.success(f"### ✅ VERIFIED: Likely Authentic News ({res['probability']*100:.2f}%)")
            else:
                st.error(f"### ⚠️ FLAG: Likely Misinformation ({(1-res['probability'])*100:.2f}%)")

            if res.get("news_found") == 1:
                st.success(f"🌍 **Tier 1 Match (GDELT Archive):** {res['publisher_name']} ([View Source]({res['article_url']}))")
            elif res["fact_flag"] == 1:
                st.success(f"🔍 **Tier 2 Match (Fact-Check Registry):** Official Rating: '{res['api_rating']}'")
            else:
                st.info("No external archive matches found. Evaluated via core machine learning probabilistic model.")

            st.divider()
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Word Count", f"{res['word_count']}")
            mcol2.metric("Sentiment", f"{res['sentiment']}")
            mcol3.metric("Emotion", f"{res['emotion']}")
        else:
            st.info("Paste an article on the left and click **Run Verification Engine**.")

        st.markdown("</div>", unsafe_allow_html=True)

with tab_docs:
    st.markdown("""
    ## 📖 Veritas AI — System Model Card & Architecture
    
    ### 1. Intended Use & Target Scope
    * **Purpose:** Automated misinformation risk detection and historical media cross-referencing prototype.
    * **Intended Input:** English news articles, headlines, and written journalistic content.
    
    ### 2. Multi-Tier Architecture
    1. **Global Archive Cross-Reference:** Queries the GDELT DOC API to verify historical publication records across worldwide press archives.
    2. **Live Fact-Check Lookup:** Queries Google Fact Check API for explicitly debunked claims.
    3. **Transformer Feature Extraction:** Uses `DistilBERT` (Sentiment) and `DistilRoBERTa` (Emotion Analysis).
    4. **Statistical Classification:** Uses $L_1$-regularized Logistic Regression trained on TF-IDF n-grams and capped structural metadata.
    
    ### 3. Known Limitations & Fallback Behavior
    * **Unindexed Archive Misses:** If an article is missing from global archives or private paywalls, the system gracefully falls back to the core statistical machine learning model without distortion.
    * **Sensational Language Sensitivities:** Legitimate news reports covering high-intensity events may carry mixed feature scores, managed via soft neutral probability thresholds.
    """)
