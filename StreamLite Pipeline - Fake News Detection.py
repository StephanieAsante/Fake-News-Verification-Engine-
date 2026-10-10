# Veritas AI: misinformation detection (Streamlit)
# Run with:  streamlit run app.py
import difflib
import gc
import os
import re
import time
from datetime import date, timedelta
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.sparse import hstack, issparse
from transformers import pipeline

# ---------------------------------------------------------
# 0. CONSTANTS
# ---------------------------------------------------------
GDELT_WINDOW_DAYS = 90            # DOC 2.0 API only covers ~3 months
GDELT_START_DATE = date(2015, 2, 19)   # GDELT 2.0 begins here
MIN_TITLE_SIMILARITY = 0.55       # to count as a candidate match
OVERRIDE_SIMILARITY = 0.70        # to allow the "trusted match" override
BQ_MAX_BYTES = 50 * 1024**3       # refuse any BigQuery query billing more than ~50 GB

TRAIN_MAX_CHAR = 32655
TRAIN_MAX_WORD = 5412
TRAIN_MAX_AVG_WORD_LEN = 74.0

TRUSTED_DOMAINS = {
    "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com", "apnews.com",
    "nytimes.com", "theguardian.com", "washingtonpost.com", "npr.org",
    "aljazeera.com", "bloomberg.com", "ft.com", "wsj.com", "cbsnews.com",
    "nbcnews.com", "abcnews.go.com", "usatoday.com", "france24.com",
}

STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "by", "for", "with", "and", "or", "but",
    "to", "of", "is", "are", "was", "were", "it", "that", "this", "told", "said",
    "from", "as", "he", "she", "they", "its", "has", "have", "had", "will",
    "would", "after", "over", "about", "into", "more", "than", "new", "not",
    "who", "what", "when", "how", "why",
}

AMBIGUOUS_TERMS = ("half", "mixture", "partly", "mostly true", "unproven",
                   "unverified", "context", "outdated")
FALSE_TERMS = ("false", "fake", "incorrect", "misleading", "pants on fire",
               "not true", "fabricated", "satire")
TRUE_TERMS = ("true", "accurate", "correct", "verified")

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
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
    .stApp p, .stApp span, .stApp label { color: #f8fafc !important; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] {
        background-color: #07090e !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-title {
        font-size: 3rem; font-weight: 800; text-align: center; margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.1rem; color: #94a3b8 !important;
        text-align: center; margin-bottom: 2.5rem;
    }
    /* st.container(border=True) is the card */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }
    .metric-badge {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc !important;
        padding: 6px 14px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600; display: inline-block;
    }
    .stTextInput label, .stTextArea label, .stDateInput label {
        color: #cbd5e1 !important; font-weight: 600 !important; font-size: 0.95rem !important;
    }
    .stTextArea textarea, .stTextInput input, .stDateInput input {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        color: #ffffff !important; border-radius: 12px !important;
    }
    div[data-testid="stMetricValue"] { color: #818cf8 !important; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: #ffffff !important; border: none !important;
        padding: 14px 28px !important; border-radius: 12px !important;
        font-weight: 700 !important; font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(168, 85, 247, 0.6) !important;
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

    # Prevents a crash when sentiment/emotion returns a label the encoder never saw
    # (e.g. the "Neutral" fallback). Unknown categories become all-zero columns.
    try:
        onehot_weights.handle_unknown = "ignore"
    except Exception:
        pass
    return fake_news_model, tfidf_weights, onehot_weights, scaler_weights


@st.cache_resource
def load_nlp_pipelines():
    sentiment_pipe = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt", truncation=True, max_length=512,
    )
    emotion_pipe = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        framework="pt", truncation=True, max_length=512,
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
    st.error(
        "⚠️ System Assets Initialization Warning\n\n"
        f"{init_error_msg}"
        "Verify that `fake_news_detection_model`, `tfidf_vectorizer`, `one_hot_encoder` "
        "and `scaler.pkl` are in the working directory."
    )

# ---------------------------------------------------------
# 3. HELPERS
# ---------------------------------------------------------
def get_secret(name, default=None):
    """st.secrets raises if no secrets.toml exists; fall back to env vars."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


def root_domain(domain: str) -> str:
    d = (domain or "").lower().removeprefix("www.")
    parts = d.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "com"}:   # bbc.co.uk
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def content_words(s: str):
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", s or "")
    return [t.lower() for t in toks if t.lower() not in STOP_WORDS and len(t) > 2]


def extract_headline(title: str, text: str) -> str:
    title = (title or "").strip()
    if title:
        return re.split(r"\s[\|\-–—]\s", title)[0].strip()   # drop " | BBC News" suffixes
    return (text or "").strip()[:150]


def infer_date_from_url(url: str):
    m = re.search(r"/(20\d{2})/(\d{1,2})/(\d{1,2})/", url or "")
    if not m:
        return None
    try:
        return date(*map(int, m.groups()))
    except ValueError:
        return None


def safe_url(url):
    return url if url and urlparse(url).scheme in ("http", "https") else None


def title_similarity(a: str, b: str) -> float:
    """Compare headlines ignoring ' - BBC News' style suffixes and punctuation."""
    def norm(s):
        s = re.split(r"\s[\|\-–—]\s", (s or "").strip())[0].lower()
        return re.sub(r"[^a-z0-9\s]", " ", s)
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    wa, wb = set(content_words(na)), set(content_words(nb))
    jac = len(wa & wb) / len(wa | wb) if wa and wb else 0.0
    return max(ratio, jac)


def score_articles(headline: str, articles):
    """Pick the article whose title best matches the headline."""
    best = None
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title:
            continue
        sim = title_similarity(headline, title)
        if sim >= MIN_TITLE_SIMILARITY and (best is None or sim > best["similarity"]):
            domain = a.get("domain", "") or ""
            best = {
                "status": "matched",
                "similarity": sim,
                "publisher": domain or "unknown",
                "trusted": root_domain(domain) in TRUSTED_DOMAINS,
                "url": safe_url(a.get("url")),
                "matched_title": title,
            }
    return best


def to_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)

# ---------------------------------------------------------
# 4. EXTERNAL LOOKUPS
# ---------------------------------------------------------
def build_gdelt_queries(headline: str):
    """Most specific -> least specific."""
    queries = []
    words = headline.split()
    if 3 <= len(words) <= 15:
        queries.append('"' + re.sub(r'["\']', "", headline) + '"')
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", headline)
    content = [t for t in toks if t.lower() not in STOP_WORDS and len(t) > 2]
    distinctive = [t for t in content if t[0].isupper() or t.isdigit()]
    ranked = list(dict.fromkeys(distinctive + content))
    if len(ranked) >= 3:
        queries.append(" ".join(ranked[:3]))
    return queries


def check_gdelt_doc(headline: str):
    """GDELT DOC 2.0: fast, but ONLY covers roughly the last 3 months."""
    queries = build_gdelt_queries(headline)
    if not queries:
        return {"status": "no_query"}

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    debug, last_error, any_ok = [], None, False

    for i, q in enumerate(queries):
        if i > 0:
            time.sleep(5.5)                      # GDELT allows ~1 request / 5 s per IP
        params = {
            "query": f"{q} sourcelang:english",
            "mode": "artlist", "format": "json", "maxrecords": 50,
            "sort": "hybridrel", "timespan": "3months",
        }
        entry = {"query": params["query"], "http_status": None,
                 "articles_returned": 0, "error": None, "top_results": []}
        debug.append(entry)

        data = None
        for _ in range(2):                       # one retry on rate limit
            try:
                r = requests.get(url, params=params, timeout=20,
                                 headers={"User-Agent": "VeritasAI/2.8"})
            except requests.RequestException as e:
                entry["error"] = f"{type(e).__name__}: {e}"[:200]
                last_error = entry["error"]
                break
            entry["http_status"] = r.status_code
            if r.status_code == 429:
                entry["error"] = "429 rate limited"
                last_error = entry["error"]
                time.sleep(5.5)
                continue
            try:
                data = r.json()
                entry["error"] = None
            except ValueError:
                entry["error"] = r.text[:200]    # GDELT plain-text error with HTTP 200
                last_error = entry["error"]
            break
        if not data:
            continue

        any_ok = True
        arts = data.get("articles", [])
        entry["articles_returned"] = len(arts)
        entry["top_results"] = [
            {"title": (a.get("title") or "")[:90], "domain": a.get("domain"),
             "similarity": round(title_similarity(headline, a.get("title") or ""), 2)}
            for a in arts[:5]
        ]
        best = score_articles(headline, arts)
        if best:
            best["debug"] = debug
            return best

    status = "no_match" if any_ok else "error"
    return {"status": status, "detail": last_error, "debug": debug}


def bq_configured() -> bool:
    try:
        import google.cloud.bigquery  # noqa: F401
        return "gcp_service_account" in st.secrets
    except Exception:
        return False


@st.cache_resource
def get_bq_client():
    from google.cloud import bigquery
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"])
    )
    return bigquery.Client(credentials=creds, project=creds.project_id)


@st.cache_resource
def bq_result_cache():
    return {}     # only successful lookups are stored, so errors are never cached


def check_gdelt_bigquery(headline: str, pub_date: date, window_days: int = 3):
    """Full GDELT 2.0 GKG archive (2015+) via BigQuery. Always date-bounded to control cost."""
    cache_key = (headline.lower(), pub_date)
    cache = bq_result_cache()
    if cache_key in cache:
        return cache[cache_key]

    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", headline)
    content = [t.lower() for t in toks if t.lower() not in STOP_WORDS and len(t) > 3]
    distinctive = [t.lower() for t in toks
                   if (t[0].isupper() or t.isdigit()) and t.lower() in content]
    kws = list(dict.fromkeys(distinctive + content))[:3]
    if len(kws) < 2:
        return {"status": "no_query"}

    try:
        from google.cloud import bigquery
        start = pub_date - timedelta(days=window_days)
        end = pub_date + timedelta(days=window_days + 1)
        kw_clause = " AND ".join(
            f"LOWER(title) LIKE CONCAT('%', @kw{i}, '%')" for i in range(len(kws))
        )
        # NOTE: confirm the <PAGE_TITLE> tag on a sample row before relying on this.
        sql = f"""
        WITH t AS (
          SELECT DocumentIdentifier AS url, SourceCommonName AS domain,
                 REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title
          FROM `gdelt-bq.gdeltv2.gkg_partitioned`
          WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start) AND TIMESTAMP(@end)
            AND SourceCommonName IN UNNEST(@domains)
            AND Extras LIKE '%<PAGE_TITLE>%'
        )
        SELECT * FROM t WHERE title IS NOT NULL AND {kw_clause} LIMIT 100
        """
        params = [
            bigquery.ScalarQueryParameter("start", "DATE", start),
            bigquery.ScalarQueryParameter("end", "DATE", end),
            bigquery.ArrayQueryParameter("domains", "STRING", sorted(TRUSTED_DOMAINS)),
            *[bigquery.ScalarQueryParameter(f"kw{i}", "STRING", k) for i, k in enumerate(kws)],
        ]
        cfg = bigquery.QueryJobConfig(query_parameters=params, maximum_bytes_billed=BQ_MAX_BYTES)
        rows = list(get_bq_client().query(sql, job_config=cfg).result())
    except Exception as e:
        return {"status": "error", "detail": str(e)[:300]}

    articles = [{"title": r.title, "domain": r.domain, "url": r.url} for r in rows]
    result = score_articles(headline, articles) or {"status": "no_match"}
    cache[cache_key] = result
    return result


def check_gdelt(user_title: str, user_text: str, pub_date):
    headline = extract_headline(user_title, user_text)
    if not headline:
        return {"status": "no_query"}
    age = (date.today() - pub_date).days if pub_date else None
    if age is not None and age > GDELT_WINDOW_DAYS:
        if not bq_configured():
            return {"status": "historical_unavailable"}
        return check_gdelt_bigquery(headline, pub_date)
    return check_gdelt_doc(headline)


def check_google_factcheck(user_title: str, user_text: str, api_key):
    """Returns dict. Only claims that overlap the query are accepted."""
    base = {"status": "skipped", "rating": None, "claim": None}
    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        return base
    query = (user_title.strip() or user_text.strip())[:100]
    if not query:
        return base

    try:
        r = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"query": query, "languageCode": "en", "key": api_key},
            timeout=8,
        )
        r.raise_for_status()
        claims = r.json().get("claims", [])
    except (requests.RequestException, ValueError) as e:
        return {**base, "status": "error", "detail": str(e)[:200]}

    q_words = set(content_words(query))
    best, best_score = None, 0.0
    for c in claims:
        c_words = set(content_words(c.get("text", "")))
        reviews = c.get("claimReview") or []
        if not q_words or not c_words or not reviews or not reviews[0].get("textualRating"):
            continue
        score = len(q_words & c_words) / len(q_words | c_words)
        if score > best_score:
            best, best_score = c, score

    if best is None or best_score < 0.25:
        return {**base, "status": "no_match"}
    return {
        "status": "matched",
        "rating": best["claimReview"][0]["textualRating"],
        "claim": best.get("text", ""),
    }


def classify_rating(rating: str):
    """1 = true, 0 = false, None = ambiguous/unknown (no override)."""
    r = (rating or "").lower()
    if any(t in r for t in AMBIGUOUS_TERMS):
        return None
    if any(t in r for t in FALSE_TERMS):     # checked before TRUE so "not true" isn't "true"
        return 0
    if any(t in r for t in TRUE_TERMS):
        return 1
    return None

# ---------------------------------------------------------
# 5. INFERENCE ENGINE
# ---------------------------------------------------------
def predict_article(user_title, user_text, user_url, pub_date, google_api_key):
    article = f"{user_title} {user_text}".strip()

    if pub_date is None:
        pub_date = infer_date_from_url(user_url)

    # NLP features
    try:
        sentiment_res = str(sentiment_pipe(article[:512])[0]["label"]).capitalize()
    except Exception:
        sentiment_res = "Neutral"
    try:
        emotion_res = str(emotion_pipe(article[:512])[0]["label"]).capitalize()
    except Exception:
        emotion_res = "Neutral"

    # External lookups
    factcheck = check_google_factcheck(user_title, user_text, google_api_key)
    gdelt = check_gdelt(user_title, user_text, pub_date)
    fact_flag = 1 if factcheck["status"] == "matched" else 0

    # Metadata features (capped to training ranges)
    n_char = len(article)
    n_word = len(article.split())
    avg_len = n_char / n_word if n_word > 0 else 0.0
    num_raw = [[min(n_char, TRAIN_MAX_CHAR), min(n_word, TRAIN_MAX_WORD),
                min(avg_len, TRAIN_MAX_AVG_WORD_LEN)]]

    X_tfidf = tfidf_weights.transform([article])

    if hasattr(scaler_weights, "feature_names_in_"):
        X_num = scaler_weights.transform(pd.DataFrame(num_raw, columns=scaler_weights.feature_names_in_))
    else:
        X_num = scaler_weights.transform(np.array(num_raw))

    cat_raw = [[sentiment_res, emotion_res]]
    if hasattr(onehot_weights, "feature_names_in_"):
        X_cat = onehot_weights.transform(pd.DataFrame(cat_raw, columns=onehot_weights.feature_names_in_))
    else:
        X_cat = onehot_weights.transform(np.array(cat_raw))

    # OneHotEncoder returns a sparse matrix by default; np.hstack would fail on it
    X_meta = np.hstack([to_dense(X_num), to_dense(X_cat), np.array([[fact_flag]])])
    X_final = hstack([X_tfidf, X_meta]).tocsr()

    raw_pred = int(fake_news_model.predict(X_final)[0])
    classes = list(getattr(fake_news_model, "classes_", [0, 1]))
    real_idx = classes.index(1) if 1 in classes else 1
    raw_prob = float(fake_news_model.predict_proba(X_final)[0][real_idx])   # P(Real)

    final_prediction, final_prob = raw_pred, raw_prob
    override_applied = False
    confidence_level = "High"

    # Priority 1: explicit fact-check verdict
    verdict = classify_rating(factcheck["rating"]) if factcheck["status"] == "matched" else None
    if verdict is not None:
        final_prediction = verdict
        final_prob = 0.95 if verdict == 1 else 0.05
        override_applied = True
        confidence_level = "High (Fact-Check Override)"

    # Priority 2: trusted-source headline match (corroboration, not proof of truth)
    elif (gdelt["status"] == "matched" and gdelt["trusted"]
          and gdelt["similarity"] >= OVERRIDE_SIMILARITY):
        final_prediction = 1
        final_prob = max(raw_prob, 0.90)
        override_applied = True
        confidence_level = f"High (Trusted-source headline match: {gdelt['publisher']})"

    # Priority 3: ML model alone
    if not override_applied:
        if 0.38 <= final_prob <= 0.62:
            confidence_level = "Moderate"
            final_prediction = 1 if final_prob >= 0.45 else 0

    return {
        "prediction": int(final_prediction),
        "probability": float(final_prob),
        "raw_probability": raw_prob,
        "confidence_level": confidence_level,
        "sentiment": sentiment_res,
        "emotion": emotion_res,
        "factcheck": factcheck,
        "gdelt": gdelt,
        "pub_date": pub_date,
        "override_applied": override_applied,
        "char_count": n_char,
        "word_count": n_word,
        "avg_word_len": avg_len,
    }

# ---------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<span class='metric-badge'>v2.8 Hardened Pipeline</span>", unsafe_allow_html=True)
    st.title("🛡️ Engine Specs")
    st.markdown("""
    **Pipeline:**
    1. Google Fact-Check Registry (validated matches only)
    2. GDELT corroboration (trusted outlets only)
    3. $L_1$ Logistic Regression + Transformer features

    **GDELT coverage:**
    - Last ~90 days: DOC 2.0 API
    - Older: GKG archive via BigQuery (needs credentials and a publication date)
    """)
    st.divider()
    st.caption("Powered by GDELT, Scikit-Learn & Streamlit")

# ---------------------------------------------------------
# 7. HERO
# ---------------------------------------------------------
st.markdown("<h1 class='hero-title'>Veritas AI Detector</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='hero-subtitle'>Misinformation Detection & Trusted-Source Corroboration Engine</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 8. MAIN INTERFACE
# ---------------------------------------------------------
def render_gdelt(g, pub_date):
    st.markdown("**Trusted-Source Corroboration (GDELT):**")
    status = g["status"]
    if status == "matched":
        link = f" ([View article]({g['url']}))" if g.get("url") else ""
        label = "Trusted outlet" if g["trusted"] else "Untrusted outlet (informational only)"
        box = st.success if g["trusted"] and g["similarity"] >= OVERRIDE_SIMILARITY else st.info
        box(
            f"🌍 Headline matches an article from **{g['publisher']}**{link}\n\n"
            f"{label} · title similarity {g['similarity']:.0%}\n\n"
            "_A match shows the headline was published; it does not verify the article body._"
        )
    elif status == "historical_unavailable":
        st.info("Article is older than GDELT's ~90-day API window and BigQuery isn't configured, "
                "so the archive check was skipped. This is not evidence either way.")
    elif status == "error":
        st.warning(f"GDELT lookup failed (rate limit or service error); result unaffected. "
                   f"{g.get('detail') or ''}")
    elif status == "no_query":
        st.info("Not enough distinctive words to build an archive query.")
    else:
        st.info("No corroborating coverage found. "
                + ("" if pub_date else "Add a publication date to search older archives. ")
                + "Absence of a match does not mean the article is fake.")

    dbg = g.get("debug")
    if dbg:
        with st.expander("GDELT diagnostics (what was sent and what came back)"):
            st.json(dbg)


def render_factcheck(f):
    st.markdown("**Fact-Check Registry:**")
    if f["status"] == "matched":
        st.success(f"Matched claim: _{f['claim'][:160]}_\n\nOfficial rating: **{f['rating']}**")
    elif f["status"] == "error":
        st.warning("Fact-check lookup failed; result unaffected.")
    elif f["status"] == "skipped":
        st.info("Fact-check lookup skipped (no API key configured).")
    else:
        st.info("No relevant claim found in the Google Fact Check registry.")


tab_app, tab_docs = st.tabs(["⚡ Verification Engine", "📖 Model Card & Specs"])

with tab_app:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        with st.container(border=True):
            st.subheader("📄 Input Article Analysis")
            user_title = st.text_input(
                "Article Title (highly recommended):",
                placeholder="e.g., Major Event Announced...",
            )
            user_url = st.text_input("Article URL (optional, used to infer the date):")
            pub_date_in = st.date_input(
                "Publication date (optional, needed for pre-2026 archive search):",
                value=None, min_value=GDELT_START_DATE, max_value=date.today(),
            )
            user_text = st.text_area("Article Body Text:", height=220,
                                     placeholder="Paste article body text here...")
            analyze_btn = st.button("⚡ Run Verification Engine")

    with col_output:
        with st.container(border=True):
            st.subheader("📊 Verification Report")

            if analyze_btn:
                if not user_text.strip() and not user_title.strip():
                    st.warning("Please provide an article title or body text to evaluate.")
                elif not weights_loaded:
                    st.error("Cannot run prediction: model assets failed to load (see banner above).")
                else:
                    st.session_state.results = None      # never show a stale report
                    with st.spinner("Running NLP models and querying external sources..."):
                        try:
                            st.session_state.results = predict_article(
                                user_title, user_text, user_url, pub_date_in,
                                get_secret("GOOGLE_FACTCHECK_API_KEY"),
                            )
                        except Exception as eval_err:
                            st.error(f"Inference Engine Error: {eval_err}")

            res = st.session_state.results
            if res is not None:
                prob = min(max(res["probability"], 0.0), 1.0)

                if res["confidence_level"] == "Moderate":
                    st.warning("⚠️ **Moderate confidence (borderline / mixed signals)**")
                    st.progress(prob)
                    st.info(f"Model score: **{prob * 100:.2f}% Real**. This sits in the neutral zone; "
                            "treat the result as inconclusive.")
                elif res["prediction"] == 1:
                    st.success("### ✅ Likely Authentic")
                    st.progress(prob)
                    st.write(f"**Credibility score:** `{prob * 100:.2f}%`")
                else:
                    st.error("### ⚠️ Likely Misinformation")
                    st.progress(1 - prob)
                    st.write(f"**Risk score:** `{(1 - prob) * 100:.2f}%`")

                st.caption(f"Confidence: {res['confidence_level']}"
                           + (" · external source applied" if res["override_applied"] else " · ML model only"))

                st.divider()
                render_gdelt(res["gdelt"], res["pub_date"])
                render_factcheck(res["factcheck"])

                st.divider()
                st.markdown("**Transformer Features & Metadata:**")
                c1, c2 = st.columns(2)
                c1.info(f"**Sentiment:** {res['sentiment']}")
                c2.info(f"**Dominant emotion:** {res['emotion']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Word Count", f"{res['word_count']}")
                m2.metric("Char Count", f"{res['char_count']}")
                m3.metric("Avg Word Length", f"{res['avg_word_len']:.2f}")
            else:
                st.info("Paste an article on the left and click **Run Verification Engine**.")

with tab_docs:
    st.markdown("""
    ## 📖 Veritas AI: Model Card & Architecture

    ### Pipeline
    1. **Fact-check lookup:** Google Fact Check API. Only claims that overlap the input are used,
       and ambiguous ratings ("half true", "mostly true", ...) never trigger an override.
    2. **Trusted-source corroboration:** GDELT DOC 2.0 for articles from the last ~90 days; the GDELT GKG
       archive on BigQuery for older ones. A match counts only if the outlet is on the trusted list and the
       headline is a close match.
    3. **Statistical classifier:** $L_1$ Logistic Regression over TF-IDF n-grams, capped metadata, and
       DistilBERT sentiment / emotion features.

    ### Known limitations
    * A headline match shows that a trusted outlet published *something* with that headline on that date.
      It does not verify the pasted body text.
    * Archive misses are not evidence of fabrication (indexing gaps, paywalls, older years).
    * The classifier may not generalise to recent or unseen outlets; validate on recent, source-disjoint data.
    """)
