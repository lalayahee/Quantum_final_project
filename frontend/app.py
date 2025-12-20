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

# Custom CSS to match the image prototype
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
        padding: 20px !important;
        margin: 10px;
    }

    /* Result Box at the top left */
    .result-container {
        border: 2px solid #0e7490;
        border-radius: 10px;
        height: 150px;
        display: flex;
        align-items: center;
        padding-left: 20px;
        margin-bottom: 30px;
    }

    /* Input Field Labels */
    .field-label {
        color: #0e7490;
        font-weight: 500;
        margin-bottom: -15px;
        font-size: 0.9rem;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #1a748c;
        color: white;
        border-radius: 15px;
        height: 3.5rem;
        font-size: 1.1rem;
        border: none;
        margin-top: 20px;
    }
    
    .stButton > button:hover {
        background-color: #145a6d;
        color: white;
    }

    /* Hide default streamlit header/footer for cleaner look */
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
medians = df[["square", "price_per_sqm", "communityaverage", "totalprice"]].median()

@st.cache_resource
def get_scaler():
    try:
        from quantum_ml.quantum_preprocess import preprocess_data
        X_train, _, _, _ = preprocess_data()
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        scaler.fit(X_train)
        return scaler
    except: return None

scaler = get_scaler()

@st.cache_resource
def load_ml_model():
    """Attempt to load a classical ML model from classical_ml/models directory."""
    model_dir = os.path.normpath(os.path.join(BASE_DIR, "..", "classical_ml", "models"))
    candidates = [
        "rf_production_1766088474.pkl",
        "rf_production_1766081555.pkl",
        "rf_final_1766088474.pkl",
        "rf_final_1766081555.pkl",
        "rf_retrained.pkl",
        "rf_best.pkl",
        "svc_best.pkl",
        "mlp_best.pkl",
        "logreg_best.pkl",
        "best.pkl",
    ]
    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            try:
                return joblib.load(path), path
            except Exception:
                continue
    # fallback: try to load any .pkl in the folder
    try:
        for f in os.listdir(model_dir):
            if f.endswith(".pkl"):
                try:
                    return joblib.load(os.path.join(model_dir, f)), os.path.join(model_dir, f)
                except Exception:
                    continue
    except Exception:
        pass
    return None, None

ml_model, ml_model_path = load_ml_model()

# --- New helper: align input to pipeline / model expected features
def align_input_to_model(X_df, model, medians):
    """Return X_df reindexed to model expected columns (fill missing with medians)."""
    try:
        # Prefer explicit feature_names_in_ if available
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
            fill_vals = {c: (float(medians[c]) if c in medians.index else 0.0) for c in cols}
            return X_df.reindex(columns=cols).fillna(value=fill_vals)
        # Try ColumnTransformer 'pre' if present
        pre = model.named_steps.get("pre") if hasattr(model, "named_steps") else None
        if pre is not None:
            cols = []
            for t in pre.transformers_:
                part = t[2]
                if isinstance(part, (list, tuple, pd.Index, np.ndarray)):
                    cols += list(part)
            cols = [c for c in dict.fromkeys(cols)]
            if cols:
                fill_vals = {c: (float(medians[c]) if c in medians.index else 0.0) for c in cols}
                return X_df.reindex(columns=cols).fillna(value=fill_vals)
    except Exception:
        pass
    return X_df

# --- UI LAYOUT ---
left_col, right_col = st.columns(2)

with left_col:
    # Top Result Display
    result_text = st.empty()
    result_text.markdown('<div class="result-container"><h3>Result: —</h3></div>', unsafe_allow_html=True)

    # Input Grid
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="field-label">Home size</p>', unsafe_allow_html=True)
        square = st.number_input("size", label_visibility="collapsed", value=float(medians["square"]))
        
        st.write("") # Spacer
        st.markdown('<p class="field-label">Community average price</p>', unsafe_allow_html=True)
        community = st.number_input("comm", label_visibility="collapsed", value=float(medians["communityaverage"]))
    
    with c2:
        st.markdown('<p class="field-label">Price per sqm</p>', unsafe_allow_html=True)
        psqm = st.number_input("psqm", label_visibility="collapsed", value=float(medians["price_per_sqm"]))
        
        st.write("") # Spacer
        st.markdown('<p class="field-label">Total price</p>', unsafe_allow_html=True)
        total = st.number_input("total", label_visibility="collapsed", value=float(medians["totalprice"]))

    submitted = st.button("Submit")

with right_col:
    # Map section (matching prototype)
    map_subcol = st.container()
    
    # Charts section
    st.markdown("---")
    chart_c1, chart_c2 = st.columns(2)

# --- LOGIC ---
if submitted:
    # 1. Quantum Logic
    quant_prob = 0.5

    # 2. Machine Learning Logic (load and predict)
    ml_prob = 0.5
    if ml_model is not None:
        # Build a DataFrame with named fields so the pipeline's ColumnTransformer works as expected
        X_user_ml = pd.DataFrame([{
            "square": square,
            "price_per_sqm": psqm,
            "communityaverage": community,
            "totalprice": total
        }])

        # Use the new helper to align inputs
        try:
            X_user_ml = align_input_to_model(X_user_ml, ml_model, medians)
        except Exception:
            pass

        st.caption(f"ML model: {os.path.basename(ml_model_path) if ml_model_path else 'Unknown'}")
        st.write("ML input:", X_user_ml.to_dict(orient='records')[0])

        try:
            if hasattr(ml_model, "predict_proba"):
                probs = ml_model.predict_proba(X_user_ml)
                ml_prob = float(probs[0][1])
            else:
                pred = ml_model.predict(X_user_ml)[0]
                ml_prob = float(max(0.0, min(1.0, float(pred)))) if isinstance(pred, (int, float)) else float(bool(pred))

            # Sanity check: perturb a numeric feature and ensure probability changes
            alt_prob = None
            try:
                X_alt = X_user_ml.copy()
                for c in X_alt.columns:
                    if pd.api.types.is_numeric_dtype(X_alt[c]):
                        X_alt.at[0, c] = X_alt.at[0, c] * 1.5
                        break
                if hasattr(ml_model, "predict_proba"):
                    alt_prob = float(ml_model.predict_proba(X_alt)[0][1])
                else:
                    alt_pred = ml_model.predict(X_alt)[0]
                    alt_prob = float(max(0.0, min(1.0, float(alt_pred)))) if isinstance(alt_pred, (int, float)) else float(bool(alt_pred))
                if abs(alt_prob - ml_prob) < 1e-6:
                    st.warning("Model produced nearly identical probability after a small input change — model may be constant or insensitive to these features.")
            except Exception:
                pass

        except Exception as e:
            st.warning(f"ML model prediction failed: {e}")
    else:
        st.info("No ML model found — using default ML mock value.")

    # Quantum debug caption
    st.caption(f"Quantum params: {'loaded' if PARAMS is not None else 'missing'}, scaler: {'loaded' if scaler is not None else 'missing'}")

    # Quantum prediction (existing)
    if PARAMS is not None and scaler is not None:
        X_user = np.array([[square, psqm, community, total]])
        X_scaled = scaler.transform(X_user)[0]
        q_out = qnode(PARAMS, X_scaled)
        quant_prob = float((q_out + 1) / 2)

    # 2. Update Result Header
    sentiment = "High" if quant_prob > 0.5 else "Low"
    result_text.markdown(f'<div class="result-container"><h3>Result: {sentiment} — Quantum {quant_prob*100:.1f}% | ML {ml_prob*100:.1f}%</h3></div>', unsafe_allow_html=True)

    # 3. Map (Finding nearest lat/long)
    data_features = df[["square", "price_per_sqm", "communityaverage", "totalprice"]].fillna(0).values
    idx = np.argmin(np.linalg.norm(data_features - [square, psqm, community, total], axis=1))
    row = df.iloc[idx]
    
    with map_subcol:
        map_data = pd.DataFrame({'lat': [row['lat']], 'lon': [row['lng']]})
        st.map(map_data, zoom=13)

    # 4. Charts (Matching prototype colors)
    def create_proto_chart(prob, title):
        values = [1-prob, prob]
        fig = go.Figure(go.Bar(
            x=['Low', 'High'],
            y=values,
            text=[f"{v*100:.1f}%" for v in values],
            textposition='auto',
            marker_color=['#4ebce3', '#105a70'], # Light teal and Dark teal
            width=0.5
        ))
        fig.update_layout(
            title=title, height=350, showlegend=False,
            template="plotly_white", margin=dict(t=30, b=10, l=10, r=10),
            yaxis=dict(range=[0,1])
        )
        return fig

    with chart_c1:
        st.write("**Machine Learning**")
        st.plotly_chart(create_proto_chart(ml_prob, "ML Prediction"), use_container_width=True) # Mock ML value
    with chart_c2:
        st.write("**Quantum**")
        st.plotly_chart(create_proto_chart(quant_prob, "Quantum Prediction"), use_container_width=True)