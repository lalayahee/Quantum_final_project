import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# --- BACKEND LINKAGE ---
try:
    from quantum_ml.quantum_preprocess import preprocess_data
    from quantum_ml.vqc_circuit import qnode
    PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "quantum_ml", "trained_params.npy")
    PARAMS = np.load(PARAMS_PATH, allow_pickle=True)
except Exception:
    PARAMS = None

# --- CONFIGURATION & PROTOTYPE STYLING ---
st.set_page_config(layout="wide", page_title="Price compare: ML vs Quantum")

# Custom CSS to fix overlapping labels and match prototype
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #ffffff;
    }

    /* Left and Right Container Boxes */
    div[data-testid="column"] {
        border: 2px solid #0e7490;
        border-radius: 15px;
        padding: 25px !important;
        margin: 10px;
    }

    /* Result Box at the top left */
    .result-container {
        border: 2px solid #0e7490;
        border-radius: 10px;
        height: 120px;
        display: flex;
        align-items: center;
        padding-left: 20px;
        margin-bottom: 25px;
        background-color: #f0f9ff;
    }

    .result-container h3 {
        text-transform: uppercase;
        font-size: 1.2rem;
        color: #0e7490;
        letter-spacing: 0.6px;
        margin: 0;
    }

    /* FIXED: Input Field Labels moved up */
    .field-label {
        color: #0e7490;
        font-weight: 700;
        margin-bottom: 8px !important;
        margin-top: 15px !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: block;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #1a748c;
        color: white;
        border-radius: 15px;
        height: 3.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        margin-top: 30px;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #145a6d;
        color: white;
        border: 1px solid #0e7490;
    }

    /* Hide default streamlit header/footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "classical_ml", "data", "house_price_top10.csv"))

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame(columns=["square", "price_per_sqm", "communityaverage", "totalprice", "lat", "lng"])

df = load_data()
# Handle empty dataframe case
if not df.empty:
    medians = df[["square", "price_per_sqm", "communityaverage", "totalprice"]].median()
else:
    medians = pd.Series({"square": 100.0, "price_per_sqm": 5000.0, "communityaverage": 5000.0, "totalprice": 500000.0})

@st.cache_resource
def get_scaler():
    """Load raw dataset and fit MinMaxScaler(feature_range=(0, pi)) on the raw features.
    The previous implementation incorrectly fitted the scaler on already-scaled training data
    which caused transforms of raw user inputs to produce invalid values."""
    try:
        csv_path = os.path.normpath(os.path.join(BASE_DIR, "..", "classical_ml", "data", "house_price_top10.csv"))
        if not os.path.exists(csv_path):
            return None
        df_local = pd.read_csv(csv_path)
        FEATURES = ["square", "price_per_sqm", "communityaverage", "totalprice"]
        X_raw = df_local[FEATURES].dropna().values
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        scaler.fit(X_raw)
        return scaler
    except Exception as e:
        # store error for debugging in session state
        try:
            st.session_state['scaler_load_error'] = str(e)
        except Exception:
            pass
        return None

scaler = get_scaler()

@st.cache_resource
def load_ml_model():
    """Try to load one of several candidate ML model files and return (model, path, error, calibrated_flag).
    If loading fails due to pickling/sklearn-version issues the error text is returned for UI debugging.

    If a test set (`classical_ml/data/X_test.npy`, `y_test.npy`) is available, attempt a quick sigmoid calibration
    using `CalibratedClassifierCV(cv='prefit', method='sigmoid')`. Calibration is optional and only attempted
    when the loaded model supports `predict_proba` or `decision_function`.
    """
    from sklearn.calibration import CalibratedClassifierCV

    model_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "classical_ml", "models"))
    candidates = ["rf_production_1766088474.pkl", "rf_best.pkl", "best.pkl"]
    last_error = None

    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                calibrated = False
                # Attempt to calibrate using test set if available
                try:
                    test_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "classical_ml", "data"))
                    X_test_path = os.path.join(test_dir, "X_test.npy")
                    y_test_path = os.path.join(test_dir, "y_test.npy")
                    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
                        X_test = np.load(X_test_path)
                        y_test = np.load(y_test_path)
                        # Align features if model exposes them
                        try:
                            if hasattr(model, "feature_names_in_"):
                                # rebuild dataframe then align
                                import pandas as _pd
                                df_test = _pd.DataFrame(X_test, columns=list(model.feature_names_in_))
                                X_test_aligned = align_input_to_model(df_test, model, medians).values
                            else:
                                X_test_aligned = X_test
                        except Exception:
                            X_test_aligned = X_test

                        # Only calibrate if model supports probability or decision function
                        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
                            try:
                                calibrator = CalibratedClassifierCV(base_estimator=model, cv='prefit', method='sigmoid')
                                calibrator.fit(X_test_aligned, y_test)
                                model = calibrator
                                calibrated = True
                            except Exception:
                                # If calibration fails, ignore and return original model
                                calibrated = False
                except Exception:
                    calibrated = False

                return model, path, None, calibrated
            except Exception as e:
                last_error = f"{name}: {e}"
    return None, None, last_error, False

ml_model, ml_model_path, ml_model_error, ml_model_calibrated = load_ml_model()

def align_input_to_model(X_df, model, medians):
    try:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            fill_vals = {c: (float(medians[c]) if c in medians.index else 0.0) for c in cols}
            return X_df.reindex(columns=cols).fillna(value=fill_vals)
    except: pass
    return X_df


def compute_ml_fallback_prob(square, psqm, community, total, df_local, logistic_k=4.0):
    """Heuristic fallback to produce an ML-like probability when pickled models fail to load.
    Uses min/max normalization of the dataset features and a weighted logistic mapping to [0,1].

    The sharpness of the logistic can be adjusted via `logistic_k`. We also record the intermediate
    score in `st.session_state['ml_fallback_score']` for debugging when possible.
    """
    try:
        FEATURES = ["square", "price_per_sqm", "communityaverage", "totalprice"]
        if df_local is None or df_local.empty:
            # sensible defaults when data not available
            mins = {"square": 10.0, "price_per_sqm": 1.0, "communityaverage": 1.0, "totalprice": 10.0}
            maxs = {"square": 500.0, "price_per_sqm": 12.0, "communityaverage": 200000.0, "totalprice": 5000.0}
        else:
            mins = df_local[FEATURES].min().to_dict()
            maxs = df_local[FEATURES].max().to_dict()

        def norm(v, c):
            mi = mins.get(c, 0.0)
            ma = maxs.get(c, 1.0)
            if ma <= mi:
                return 0.5
            return float((v - mi) / (ma - mi))

        n_square = np.clip(norm(square, "square"), 0.0, 1.0)
        n_psqm = np.clip(norm(psqm, "price_per_sqm"), 0.0, 1.0)
        n_comm = np.clip(norm(community, "communityaverage"), 0.0, 1.0)
        n_total = np.clip(norm(total, "totalprice"), 0.0, 1.0)

        # Weighted combination (emphasize price_per_sqm and community average)
        w = np.array([0.1, 0.5, 0.2, 0.2])
        vec = np.array([n_square, n_psqm, n_comm, n_total])
        score = float(np.dot(w, vec))

        # Store score for debugging
        try:
            st.session_state['ml_fallback_score'] = score
        except Exception:
            pass

        # Softer logistic than before to avoid forcing extremes (default k=4.0)
        prob = 1.0 / (1.0 + float(np.exp(-logistic_k * (score - 0.5))))
        return float(np.clip(prob, 0.0, 1.0))
    except Exception:
        return 0.5

# --- NAVIGATION ---
page = st.sidebar.selectbox("Navigate", ["Home", "Project Overview", "Comparative Analysis", "About Us", "Service", "Contact"])
st.sidebar.markdown("---")
st.sidebar.write("**Group6 — Quantum ML for Real Estate**")

# Probability calibration / smoothing controls
st.sidebar.markdown("---")
calibration_enabled = st.sidebar.checkbox("Enable ML probability calibration (use test set if available)", value=True)
smoothing_enabled = st.sidebar.checkbox("Enable probability shrinkage (pull extremes toward 0.5)", value=True)
shrink_factor = st.sidebar.slider("Shrink factor (1 = no shrink, 0 = full shrink to 0.5)", min_value=0.5, max_value=1.0, value=0.8, step=0.05)

# Render non-Home pages here and short-circuit the script to skip the prediction UI
if page != "Home":
    if page == "Project Overview":
        st.title("Project Overview")
        st.markdown("""
        **Research Focus:** Exploring quantum machine learning for real estate price prediction

        **Approach:** Hybrid classical-quantum computing pipeline

        **Innovation:** Comparing traditional ML with quantum algorithms (VQC)

        **Impact:** Understanding quantum ML potential for practical applications
        """)
        st.subheader("Key Components")
        st.markdown("""
        - Classical ML baseline models (Random Forest, SVM, Logistic Regression)
        - Quantum Variational Classifier (VQC) implementation
        - Comprehensive performance comparison
        - Educational analysis of workflow differences between classical and quantum approaches
        """)
        st.markdown("<div style='border:1px dashed #ccc; padding:20px; text-align:center;'>Figure 1: Interface image (placeholder)</div>", unsafe_allow_html=True)
    elif page == "Comparative Analysis":
        st.title("Comparative Analysis")
        st.markdown("""
        **Focus areas:**
        - Multiple ML algorithms comparison
        - Feature engineering and selection
        - Performance benchmarking (accuracy, runtime, robustness)
        - Implement classical baseline and develop quantum ML pipeline
        - VQC architecture design, quantum data preprocessing, hybrid optimization strategy
        - Accuracy comparison, computational complexity analysis, workflow efficiency evaluation
        """)
    elif page == "About Us":
        st.title("About Us")
        st.markdown("Group6 — A student project investigating the practical potential of quantum machine learning applied to real estate price prediction.")
    elif page == "Service":
        st.title("Service")
        st.markdown("We provide a reproducible comparative pipeline that implements both classical baselines and a VQC quantum model, along with benchmarking and educational materials.")
    elif page == "Contact":
        st.title("Contact")
        st.markdown("For questions or collaboration: **Group6** — Email: `group6@example.com` — GitHub: `https://github.com/lalayahee/Quantum_final_project`")
    st.stop()

# --- UI LAYOUT ---
left_col, right_col = st.columns(2)

with left_col:
    # Result Display
    result_placeholder = st.empty()
    result_placeholder.markdown('<div class="result-container"><h3>Result: — (Submit to calculate)</h3></div>', unsafe_allow_html=True)

    # Input Grid
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<label class="field-label">Home size (sqm)</label>', unsafe_allow_html=True)
        square = st.number_input("size", label_visibility="collapsed", value=float(medians["square"]), key="in_sq")
        
        st.markdown('<label class="field-label">Community Avg Price</label>', unsafe_allow_html=True)
        community = st.number_input("comm", label_visibility="collapsed", value=float(medians["communityaverage"]), key="in_comm")
    
    with c2:
        st.markdown('<label class="field-label">Price per sqm</label>', unsafe_allow_html=True)
        psqm = st.number_input("psqm", label_visibility="collapsed", value=float(medians["price_per_sqm"]), key="in_psqm")
        
        st.markdown('<label class="field-label">Total Price</label>', unsafe_allow_html=True)
        total = st.number_input("total", label_visibility="collapsed", value=float(medians["totalprice"]), key="in_total")

    submitted = st.button("SUBMIT PREDICTION")

    # --- Debug / Status (shows model/params/scaler load state) ---
    try:
        if ml_model is None:
            if ml_model_error:
                st.warning(f"ML model not loaded: {ml_model_error}")
            else:
                st.info("No ML model found")
        else:
            st.success(f"ML model loaded: {os.path.basename(ml_model_path)} | predict_proba: {hasattr(ml_model, 'predict_proba')}")

        if PARAMS is None:
            st.warning("Quantum params not loaded (PARAMS is None).")
        else:
            st.success("Quantum params loaded")

        if scaler is None:
            s_err = st.session_state.get('scaler_load_error') if 'scaler_load_error' in st.session_state else None
            if s_err:
                st.warning(f"Scaler not available: {s_err}")
            else:
                st.info("Scaler not available")
        else:
            st.success("Scaler loaded")
    except Exception:
        pass

with right_col:
    map_container = st.container()
    st.markdown("---")
    chart_c1, chart_c2 = st.columns(2)

# --- LOGIC ---
if submitted:
    quant_prob = 0.5
    ml_prob = 0.5
    
    # 1. ML Logic
    if ml_model is not None:
        X_user_ml = pd.DataFrame([{"square": square, "price_per_sqm": psqm, "communityaverage": community, "totalprice": total}])
        X_user_ml = align_input_to_model(X_user_ml, ml_model, medians)
        try:
            if hasattr(ml_model, "predict_proba"):
                probs = ml_model.predict_proba(X_user_ml)[0]
                classes = list(getattr(ml_model, "classes_", []))
                # Try to find the positive class index (prefer integer 1, then string '1'), otherwise fallback to last column
                pos_idx = None
                if 1 in classes:
                    pos_idx = classes.index(1)
                elif '1' in classes:
                    pos_idx = classes.index('1')
                else:
                    pos_idx = -1
                try:
                    ml_prob = float(probs[pos_idx])
                except Exception:
                    ml_prob = float(np.clip(probs[-1], 0.0, 1.0))
                # store debug info
                try:
                    st.session_state['ml_debug'] = {'probs': probs.tolist(), 'classes': classes, 'pos_idx': pos_idx}
                except Exception:
                    pass
            else:
                pred = ml_model.predict(X_user_ml)[0]
                ml_prob = float(max(0.0, min(1.0, float(pred))))
                try:
                    st.session_state['ml_debug'] = {'pred': float(pred)}
                except Exception:
                    pass
        except Exception as e:
            st.error(f"ML Error: {e}")
    else:
        # Fallback heuristic when ML pickles can't be loaded (avoids 50/50 default)
        try:
            ml_prob = compute_ml_fallback_prob(square, psqm, community, total, df)
            st.info("ML model unavailable — using heuristic fallback")
            st.session_state['ml_debug'] = st.session_state.get('ml_debug', {})
            st.session_state['ml_debug']['used_fallback'] = True
        except Exception as e:
            st.error(f"ML fallback error: {e}")

    # If calibration is enabled and we attempted to calibrate at load time, surface that info
    try:
        st.session_state['ml_calibrated'] = bool(ml_model_calibrated and calibration_enabled)
    except Exception:
        st.session_state['ml_calibrated'] = False

    # Apply smoothing (shrink extremes toward 0.5) if enabled
    try:
        if smoothing_enabled and ml_prob is not None:
            raw_ml_prob = float(ml_prob)
            shrunk = 0.5 + (raw_ml_prob - 0.5) * float(shrink_factor)
            ml_prob = float(np.clip(shrunk, 1e-6, 1-1e-6))
            st.session_state['ml_debug'] = st.session_state.get('ml_debug', {})
            st.session_state['ml_debug'].update({'raw_ml_prob': raw_ml_prob, 'ml_prob_shrunk': ml_prob, 'shrink_factor': float(shrink_factor)})
        else:
            st.session_state['ml_debug'] = st.session_state.get('ml_debug', {})
            st.session_state['ml_debug'].setdefault('raw_ml_prob', float(ml_prob if ml_prob is not None else 0.5))
    except Exception:
        pass

    # 2. Quantum Logic
    if PARAMS is not None and scaler is not None:
        try:
            X_user = np.array([[square, psqm, community, total]])
            X_scaled = scaler.transform(X_user)[0]
            q_out = qnode(PARAMS, X_scaled)
            quant_prob = float((q_out + 1) / 2)
            try:
                st.session_state['quant_debug'] = {'q_out': float(q_out), 'X_scaled': X_scaled.tolist()}
            except Exception:
                pass
        except: pass

    # 3. Update Result
    sentiment = "High Value" if quant_prob > 0.5 else "Standard Value"
    result_placeholder.markdown(f'''
        <div class="result-container">
            <h3>Result: {sentiment} <br> 
            <span style="font-size:0.9rem; color:#666;">Quantum: {quant_prob*100:.1f}% | ML: {ml_prob*100:.1f}%</span>
            </h3>
        </div>
    ''', unsafe_allow_html=True)

    # 4. Map
    if not df.empty:
        data_features = df[["square", "price_per_sqm", "communityaverage", "totalprice"]].fillna(0).values
        idx = np.argmin(np.linalg.norm(data_features - [square, psqm, community, total], axis=1))
        row = df.iloc[idx]
        with map_container:
            st.map(pd.DataFrame({'lat': [row['lat']], 'lon': [row['lng']]}), zoom=12)

    # 5. Charts
    def create_bar(prob, title):
        fig = go.Figure(go.Bar(
            x=['Low', 'High'], y=[1-prob, prob],
            text=[f"{(1-prob)*100:.0f}%", f"{prob*100:.0f}%"], textposition='auto',
            marker_color=['#4ebce3', '#105a70'], width=0.4
        ))
        fig.update_layout(title=title, height=300, template="plotly_white", margin=dict(t=40, b=0, l=0, r=0), yaxis=dict(range=[0,1]))
        return fig

    chart_c1.plotly_chart(create_bar(ml_prob, "ML ANALYSIS"), use_container_width=True)
    # Inline ML class/probability details
    ml_dbg = st.session_state.get('ml_debug', {})
    if ml_dbg:
        try:
            ml_classes = ml_dbg.get('classes')
            ml_probs = ml_dbg.get('probs')
            pos_idx = ml_dbg.get('pos_idx')
            if ml_classes and ml_probs is not None and pos_idx is not None:
                class_high = ml_classes[pos_idx]
                st.caption(f"ML: 'High' class = {class_high} | probs = {ml_probs}")
            else:
                st.caption(f"ML debug: {ml_dbg}")
        except Exception:
            st.caption(f"ML debug: {ml_dbg}")

    chart_c2.plotly_chart(create_bar(quant_prob, "QUANTUM ANALYSIS"), use_container_width=True)
    # Inline quantum details
    q_dbg = st.session_state.get('quant_debug')
    if q_dbg:
        try:
            st.caption(f"Quantum: q_out = {q_dbg.get('q_out')} | scaled input = {q_dbg.get('X_scaled')}")
        except Exception:
            st.caption(f"Quantum debug: {q_dbg}")

    # Debugging information for prediction internals
    with st.expander("Debug info (prediction internals)"):
        st.write({
            'ml_model_loaded': ml_model is not None,
            'ml_model_path': os.path.basename(ml_model_path) if ml_model_path else None,
            'ml_calibrated': st.session_state.get('ml_calibrated'),
            'ml_debug': st.session_state.get('ml_debug'),
            'ml_fallback_score': st.session_state.get('ml_fallback_score'),
            'quant_debug': st.session_state.get('quant_debug')
        })