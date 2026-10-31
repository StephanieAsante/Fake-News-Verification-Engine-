
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
@st.cache_resource #The @st.cache_resource decorator in Streamlit is a performance optimization tool that loads heavy objects
# such as machine learning models, scalers, encoders, and transformer pipeline weights—into your server's memory only once when the app initializes, rather than reloading them every single time a user interacts with the application or clicks a button
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
        "text-classification", # Tells Hugging Face what NLP task the pipeline should perform (classifying input text into labels like Positive or Negative)
        model="distilbert-base-uncased-finetuned-sst-2-english", # Specifies the exact pre-trained model checkpoint to download and use from the Hugging Face Hub.
        framework="pt", # Forces the underlying framework to use PyTorch (pt), ensuring compatibility with GPU acceleration and tensor operations
        truncation=True, # Enables cutting the text short
        max_length=512, # Automatically cuts off any input text that exceeds 512 tokens, which is the native architectural sequence limit for DistilBERT models.
    )
    
    # Lightweight Emotion Detection (~260MB RAM)
    emotion_pipe = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        framework="pt",
        truncation=True,
        max_length=512, # A token is the fundamental unit of text that a transformer model reads, which can be an entire word, a part of a word (subword), a single prefix/suffix, or even a punctuation mark.
    )

    # Force Python Garbage Collection
    gc.collect() #calls the built-in garbage collector to immediately sweep through memory, 
    # It finds objects, data structures, or in our case, massive temporary data structures, tokenizers, and network weights from our DistilBert that are no longer referenced or needed, and deallocate them to free up RAM.

    return sentiment_pipe, emotion_pipe

# Initialization Handler with Explicit Diagnostic Errors
# This entire block of code is to ensure that if the loading of weights or transformers are unable to load successfully,
# an error message (called init_error_msg) will be printed out telling us to deal with the particular error so execution can proceed
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
# 3. EXTERNAL APIS (TIER 1: GDELT & TIER 2: GOOGLE FACT-CHECK)
# ---------------------------------------------------------
def check_gdelt_archive(user_title, user_text):
    # The above holds the name of the function('check_gdelt_archive') and what information the function will need to execute
    # which in our case is the title of the article and the text (Note: caveats to what inputs the text will contain will be highlighted in the arguments below)
    """
    Tier 1: Queries the GDELT DOC 2.0 API (Free / Open Source Historical Global Archive)
    using flexible keyword extractions to check if historical articles exist.
    """
    raw_str = f"{user_title} {user_text[:200]}".strip() # he function combines the article title with the first 200 characters of the article text to form a searchable string. 
    if not raw_str:
        return 0, None, None # If both are empty, it immediately exits and returns 0, None, None
        
    # Stop-word removal: It creates a set of common English filler words (like "the", "and", "is", "to", etc.) that carry no unique meaning for a search query.    
    stop_words = {
        "the", "a", "an", "in", "on", "at", "by", "for", "with", "and", 
        "or", "but", "to", "of", "is", "are", "was", "were", "it", "that", 
        "this", "told", "said", "from", "as", "he", "she", "they", "its"
    }
    words = [w for w in raw_str.split() if w.lower() not in stop_words and len(w) > 2] # It splits the combined text into individual words, converts them to lowercase, strips out the stop words, and ignores any word shorter than 3 characters (len(w) > 3 or > 2)

    # t takes the top 4 remaining core keywords, joins them to build a clean, focused search query optimized for GDELT's search syntax. 
    # e.g (Global Summit Climate Agreement Signed). If no valid words remain, it defaults to using the first 40 characters of the raw string.
    flexible_query = " ".join(words[:4])
    if not flexible_query:
        flexible_query = raw_str[:40]

    url = "https://api.gdeltproject.org/api/v2/doc/doc" # This targets the brain source we are about to tap inot
    params = {
        "query": flexible_query, # using the extraction parameter set in our variable 'flexible_query'
        "mode": "artlist", # Tells GDELT to return a list of matching articles.
        "maxrecords": 3, # Requests up to 3 matching records.
        "format": "json" # Formats the response as a JSON dictionary for easy parsing in Python.
    }

    try: # Tells Python to attempt executing the network request block. If any error (like a connection drop, timeout, or bad response format) occurs inside this block, Python immediately jumps to the except block instead of crashing the web app.
        response = requests.get(url, params=params, timeout=6) # Sends an HTTP GET request to the GDELT DOC 2.0 API endpoint (url) passing along your query parameters (params)
        if response.status_code == 200: # Checks the HTTP status code returned by the server. A 200 OK status means the request was completely successful and data was returned
            data = response.json() # Takes the raw response text from GDELT, converting the JSON payload into a structured Python dictionary so you can easily navigate it like a map.
            articles = data.get("articles", []) # Extracts the list of matching articles from the JSON dictionary using .get(). If the "articles" key doesn't exist in the response, it safely defaults to an empty list ([]) to prevent KeyError crashes.
            if articles: # if articles were actually found
                top_article = articles[0] # Selects the very first article (the top/most relevant match) from the returned list of results.
                publisher = top_article.get("domain", "GDELT Verified Global Archive") # Extracts the domain name (the publisher) of the article. If the domain property is missing, it assigns a clean default fallback label ("GDELT Verified Global Archive")
                article_url = top_article.get("url", "#") # Extracts the direct hyperlink to the source article so it can be displayed in your Streamlit app interface. If no URL is found, it defaults to a placeholder link ("#").
                return 1, publisher, article_url # Returns a success signal (1) along with the publisher's name and article link, confirming that GDELT successfully corroborated the article.
    except Exception as e: # if the process throws in an error message, 
        print(f"GDELT Archive API Request Error: {e}") # print this message

    return 0, None, None # If the try block fails or no matching articles are found, the function gracefully returns 0, None, None, allowing your application to smoothly fall back to your machine learning classifier


def check_google_factcheck(user_title, user_text, google_api_key):
    """Tier 2: Queries Google Fact Check API for explicit claim ratings."""
    if not google_api_key or google_api_key == "YOUR_GOOGLE_API_KEY_HERE": # if Google Fact Checker API key is not provided, the process should fall to None
        return 0, None

    query_str = user_title.strip() if user_title.strip() else user_text.strip()[:100] # first check user title, but if user title is not provided, use the first 100 words from the text of the article
    if not query_str:
        return 0, None # if both title and text are not provided, return 0, None

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search" # Connecting to the fact checker API Access point
    params = {
        "query": query_str[:100],# first 100 words will be used to check in the fact checker database
        "languageCode": "en",# The English language will be used
        "key": google_api_key, # The database will be accessed through the provided API key
    }

    try:
        response = requests.get(url, params=params, timeout=5) # In the query's interaction of the database via the url and parameters provided, the factcheker has 5 secs to respond to the request made
        if response.status_code == 200: # if the response is 200, it means the request has been found
            data = response.json() # It returns the results as a in a JSON format 
            claims = data.get("claims", []) # Extract the list of matching claims
            if claims:
                rating = (
                    claims[0] # select the first claim matching our request 
                    .get("claimReview", [{}])[0] # Particularly, get the claim review, which is the key identifier which can hold the exact verbatim claim, who made the claim and the url to it 
                    .get("textualRating", "") # and the textual rating (which the results of comparing the new unverified text with a repository of verified news to know where the new article stands)
                )
                return 1, rating # After doing this, return as 1, meaning the article you want to check for has been factchecked
    except Exception:
        pass # if it can't be found in the fact-checker database, pass it  

    return 0, None # and return as 0

# ---------------------------------------------------------
# 4. INFERENCE ENGINE WITH REORDERED PRIORITY SEQUENCE
# ---------------------------------------------------------
def predict_article(user_title, user_text, google_api_key):
    fully_combined_article = f"{user_title} {user_text}".strip()

    # Step A: Dynamic NLP Inference
    try:
        sentiment_raw = sentiment_pipe(fully_combined_article[:512])[0]["label"] # Takes the sentiment pipe with the Hugging Face transformer and applies it to our combined title and text row, truncating to 512 tokens for the analysis
        sentiment_res = str(sentiment_raw).capitalize() # capitalizes the begining of each sentiment prediction
    except Exception:
        sentiment_res = "Neutral" # If an error occurs or the transformer cannot successfully classify the text, the code instantly intercepts the failure and assigns a default, safe fallback value: "Neutral" 

    try:
        emotion_raw = emotion_pipe(fully_combined_article[:512])[0]["label"] 
        emotion_res = str(emotion_raw).capitalize() # capitalizes the begining of each sentiment prediction
    except Exception:
        emotion_res = "Neutral" # If an error occurs or the transformer cannot successfully classify the text, the code instantly intercepts the failure and assigns a default, safe fallback value: "Neutral" 

    # Step B: SEQUENTIAL EXTERNAL LOOKUPS (GDELT First, then Google Fact Check)
    news_found, publisher_name, article_url = check_gdelt_archive(
        user_title, user_text
    ) # Gives us the text of the news article found, the name of the publisher, and the article URL when the check_gdelt_archive function runs successfully from the user inputting the article title and text
    fact_flag, api_rating = check_google_factcheck(
        user_title, user_text, google_api_key
    ) # This gives us the fact flag of 1 if it's real and the api_rating if the function runs successfully from the user inputting the article title, text, and the google_api_key is provided

    # Step C: Metadata Calculations & Capping Bounds
    # This provides a cap for our scale calculations so articles with over 32655 characters, 5412 words and an average over 74.0 do not confused our scaler_weights loaded 
    # but suggest that counts above them should be giving the full score, so the capped figures became the threshold 
    TRAIN_MAX_CHAR = 31384
    TRAIN_MAX_WORD = 5219
    TRAIN_MAX_AVG_WORD_LEN = 16.131

    raw_char_count = len(fully_combined_article)
    raw_word_count = len(fully_combined_article.split())
    raw_avg_word_len = (
        raw_char_count / raw_word_count if raw_word_count > 0 else 0.0
    )

    capped_char_count = min(raw_char_count, TRAIN_MAX_CHAR)
    capped_word_count = min(raw_word_count, TRAIN_MAX_WORD)
    capped_avg_word_len = min(raw_avg_word_len, TRAIN_MAX_AVG_WORD_LEN)

    # Step D: Feature Transformations
    # Note that the feature transformation should happen how they were done during the model development and how the weights were loaded
    X_tfidf = tfidf_weights.transform([fully_combined_article])

    num_raw = [[capped_char_count, capped_word_count, capped_avg_word_len]]
    if hasattr(scaler_weights, "feature_names_in_"):
        num_df = pd.DataFrame(
            num_raw, columns=scaler_weights.feature_names_in_
        )
        X_num = scaler_weights.transform(num_df)
    else:
        X_num = scaler_weights.transform(np.array(num_raw))

    cat_raw = [[sentiment_res, emotion_res]]
    if hasattr(onehot_weights, "feature_names_in_"):
        cat_df = pd.DataFrame(
            cat_raw, columns=onehot_weights.feature_names_in_
        )
        X_cat = onehot_weights.transform(cat_df)
    else:
        X_cat = onehot_weights.transform(np.array(cat_raw))

    X_fact = np.array([[fact_flag]])# this checks whether fact-checker resulted in 1 or 0

    X_meta = np.hstack([X_num, X_cat, X_fact])
    X_final = hstack([X_tfidf, X_meta])

    # Step E: Model Statistical Prediction (The Core ML Brain)
    raw_pred = fake_news_model.predict(X_final)[0]
    raw_prob = fake_news_model.predict_proba(X_final)[0][1]  # Prob of Class 1 (Real) or 0 (Fake)

    # Step F: Reordered Priority Sequence & Clean Probabilistic Fallback
    final_prediction = raw_pred
    final_prob = raw_prob
    override_applied = False
    confidence_level = "High"

    # 1. PRIORITY 1: GDELT Global Archive Match (Checked FIRST)
    # This says that if the first tier, which is our GDELT, is ==1, our final prediction automatically becomes 1 (Real news)
    # our final prob becomes the maximum, which is 90%, and an override is applied to flag that article as 'Likely Verified News'
    if news_found == 1:
        final_prediction = 1
        final_prob = max(raw_prob, 0.90)
        override_applied = True
        confidence_level = f"High (GDELT Archive Verified: {publisher_name})" # after that, you print the statement with the publisher's name

    # 2. PRIORITY 2: Google Fact Check Database Match (Checked SECOND)
    elif fact_flag == 1 and api_rating: # if fact_flag ==1 (meaning that particular article has been fact-checked) and the API rating given is 
        rating_lower = api_rating.lower() # is lower
        if any(
            term in rating_lower # print any of the below that matches
            for term in [
                "false",
                "fake",
                "incorrect",
                "misleading",
                "pants on fire",
            ]
        ):
            final_prediction = 0 # final prediction becomes 0, which is fake news
            final_prob = 0.05 # with a final prob of 0.05
            override_applied = True # and this will override any statistical prediction as the fact-checker is already telling us that the publication is fack
            confidence_level = "High (Fact-Check Override)"
        elif any(
            term in rating_lower # else if 'term' in rating_lower indicates that the article is true
            for term in ["true", "accurate", "correct", "verified"]
        ):
            final_prediction = 1 # final prediction becomes 1, which is real news
            final_prob = 0.95 # it's assigned a probability of 0.95 
            override_applied = True # and it permits a successful overide of any prediction made by the predictive mdoel
            confidence_level = "High (Fact-Check Override)"

    # 3. PRIORITY 3: Natural Probabilistic Machine Learning Fallback (No Overrides)
    if not override_applied:
        if 0.38 <= final_prob <= 0.62:
            confidence_level = "Moderate" # This ensures that a moderate confidence is presented for articles that have between 38% to 62% probalitiy of treal news becoming fake news
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
    } # This bracket houses all the expected outcomes the model will need to produce its displaying results

# ---------------------------------------------------------
# 5. SIDEBAR ARCHITECTURE & BENCHMARKS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<span class='metric-badge'>v2.8 Reordered Pipeline</span>",
        unsafe_allow_html=True, # tells Streamlit to permit the rendering of raw HTML code (such as <h1>, <p class='hero-subtitle'>, or custom <div> and styling tags) directly within the app interface.
    )
    st.title("🛡️ Engine Specs")

    st.markdown("""
    **Model Architecture:**
    1. GDELT Global Archive Lookup
    2. Google Fact-Check Registry
    3. L_2 Logistic Regression & Transformers
    
    ---
    **Model Benchmarks:**
    - **Accuracy:** 97.07%
    - **Recall (Fake News):** 97.2%
    - **Precision:** 97.1%
    - **ROC-AUC:** 0.9707
    """)
    st.divider()
    st.caption("Powered by GDELT Archive, Scikit-Learn & Streamlit")

# ---------------------------------------------------------
# 6. HERO SECTION
# ---------------------------------------------------------
st.markdown(
    "<h1 class='hero-title'>Veritas AI Detector</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='hero-subtitle'>Enterprise Misinformation Detection & Global Archive Verification Engine</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 7. MAIN INTERFACE LAYOUT
# ---------------------------------------------------------
tab_app, tab_docs = st.tabs(["⚡ Verification Engine", "📖 Model Card & Specs"])

with tab_app:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📄 Input Article Analysis")

        user_title = st.text_input(
            "Article Title (Highly Recommended):",
            placeholder="e.g., Breaking News: Major Event Announced...",
            help="Providing an explicit article headline significantly enhances feature vector density and model prediction accuracy.",
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
                    "Running sequential pipeline (GDELT -> Fact Check -> ML Brain)..."
                ):
                    google_api_key = st.secrets.get(
                        "GOOGLE_FACTCHECK_API_KEY", None
                    )

                    try:
                        st.session_state.results = predict_article(
                            user_title=user_title,
                            user_text=user_text,
                            google_api_key=google_api_key,
                        )
                    except Exception as eval_err:
                        st.error(f"Inference Engine Error: {eval_err}")

        if st.session_state.results is not None:
            res = st.session_state.results

            # Metric Visualization Output
            if res.get("confidence_level") == "Moderate":
                st.warning("⚠️ **Moderate Confidence Signal (Borderline / Mixed Case)**")
                st.progress(float(res["probability"]))
                st.info(
                    f"The engine scored this article with a probability of **{res['probability'] * 100:.2f}% Real**. "
                    "Because this falls in the neutral zone, stylistic elements resemble mixed linguistic signals."
                )
            elif res["prediction"] == 1:
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

            if res.get("override_applied"):
                st.info(
                    "ℹ️ Output verified via external archive or Fact-Check database match."
                )

            st.divider()

            # GDELT Archive Cross-Reference Status
            st.markdown("**Global Archive Cross-Reference (GDELT):**")
            if res.get("news_found") == 1:
                st.success(
                    f"🌍 **Tier 1 Match (GDELT Archive):** {res['publisher_name']} "
                    f"([View Source Article]({res['article_url']}))"
                )
            else:
                st.info(
                    "No direct match in GDELT historical archive."
                    "(Note: The GDELT archive connection verifies published news and media coverage dating back up to 90 days)."
                    "Evaluation has been smoothly handed over to the core probabilistic machine learning model."
                )

            # Fact-Check Registry Status
            st.markdown("**Fact-Check Registry Status:**")
            if res["fact_flag"] == 1:
                st.success(
                    f"🔍 **Tier 2 Match (Fact-Check Registry):** Official Rating: '{res['api_rating']}'"
                )
            else:
                st.info(
                    "No direct match in Google Fact Check registry."
                    "(Note: The Fact Check API indexes claims that have been explicitly investigated, rated, or debunked by independent verification organizations)."
                    "Evaluation proceeds using natural text analysis."
                )

            st.divider()

            # Detected NLP Metrics
            st.markdown("**Automated Transformer Feature Extraction:**")
            ncol1, ncol2 = st.columns(2)
            ncol1.info(f"**Sentiment:** {res['sentiment'].capitalize()}")
            ncol2.info(f"**Dominant Emotion:** {res['emotion'].capitalize()}")

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


with tab_docs:
    st.markdown("""
    ## 📖 Veritas AI — System Model Card & Architecture
    
    ### 1. Intended Use & Target Scope
    * **Purpose:** Automated misinformation risk detection and historical media cross-referencing prototype.
    * **Intended Input:** English news articles, headlines, and written journalistic content.
    
    ### 2. Multi-Tier Sequential Architecture
    1. **GDELT Global Archive Lookup (Tier 1):** Checks worldwide news archives to verify publication.
    2. **Google Fact-Check Registry (Tier 2):** Queries explicitly debunked or verified claims.
    3. **Transformer Feature Extraction (Tier 3):** Uses `DistilBERT` (Sentiment) and `DistilRoBERTa` (Emotion Analysis).
    4. **Statistical Classification:** Uses $L_1$-regularized Logistic Regression trained on TF-IDF n-grams and capped structural metadata.
    
    ### 3. Known Limitations & Fallback Behavior
    * **Unindexed Archive Misses:** If an article is missing from global archives, the system gracefully falls back to the core statistical machine learning model without distortion.
    * **Sensational Language Sensitivities:** Legitimate news articles covering emotionally intense events (e.g., crime, court cases) may trigger higher statistical risk scores due to stylistic vocabulary overlap with sensational fake news.
    """)

# %%


# %%



