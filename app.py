import streamlit as st
import joblib, numpy as np, pandas as pd
import xgboost as xgb
import plotly.graph_objects as go

st.set_page_config(page_title="KisanFin", page_icon="🌾", layout="centered")

model, cols = joblib.load("model.pkl")

# ---------- Language ----------
T = {
    "English": {
        "title": "🌾 KisanFin",
        "sub": "Smart credit scoring for farmers and gig workers",
        "type": "I am a", "farmer": "Farmer", "gig": "Gig worker",
        "income": "Monthly income (₹)",
        "vol": "Income irregularity (0 = steady, 1 = very irregular)",
        "upi": "UPI transactions per month",
        "rech": "Mobile recharge regularity (0 to 1)",
        "bill": "Bills paid on time (0 to 1)",
        "age": "Age", "dep": "Dependents",
        "land": "Land owned (acres)",
        "days": "Active work days per month",
        "tenure": "Months on the platform",
        "btn": "Check my credit score",
        "score": "Your KisanFin Score", "loan": "Suggested loan limit",
        "risk": "Default risk", "why": "Why this score?",
        "up": "✅ Helping your score", "down": "⚠️ Hurting your score",
        "tips": "How to improve your score",
        "low": "Low risk, likely to be approved",
        "mid": "Medium risk, small loan suggested",
        "high": "High risk, build your record first",
    },
    "తెలుగు": {
        "title": "🌾 కిసాన్‌ఫిన్",
        "sub": "రైతులు మరియు గిగ్ వర్కర్ల కోసం స్మార్ట్ క్రెడిట్ స్కోర్",
        "type": "నేను", "farmer": "రైతు", "gig": "గిగ్ వర్కర్",
        "income": "నెలవారీ ఆదాయం (₹)",
        "vol": "ఆదాయ అస్థిరత (0 = స్థిరం, 1 = చాలా అస్థిరం)",
        "upi": "నెలకు UPI లావాదేవీలు",
        "rech": "మొబైల్ రీఛార్జ్ క్రమబద్ధత (0 నుండి 1)",
        "bill": "సమయానికి చెల్లించిన బిల్లులు (0 నుండి 1)",
        "age": "వయసు", "dep": "ఆధారపడినవారు",
        "land": "భూమి (ఎకరాలు)",
        "days": "నెలలో పని చేసిన రోజులు",
        "tenure": "ప్లాట్‌ఫారమ్‌లో నెలలు",
        "btn": "నా క్రెడిట్ స్కోర్ చూడండి",
        "score": "మీ కిసాన్‌ఫిన్ స్కోర్", "loan": "సూచించిన రుణ పరిమితి",
        "risk": "డిఫాల్ట్ ప్రమాదం", "why": "ఈ స్కోర్ ఎందుకు?",
        "up": "✅ స్కోర్‌ను పెంచేవి", "down": "⚠️ స్కోర్‌ను తగ్గించేవి",
        "tips": "స్కోర్ మెరుగుపరచుకునే మార్గాలు",
        "low": "తక్కువ ప్రమాదం, ఆమోదం అవకాశం ఎక్కువ",
        "mid": "మధ్యస్థ ప్రమాదం, చిన్న రుణం సూచించబడింది",
        "high": "ఎక్కువ ప్రమాదం, ముందుగా రికార్డు నిర్మించండి",
    },
}

# Plain-language reason for each feature: (good text, bad text)
REASONS = {
    "English": {
        "monthly_income": ("Good monthly income", "Low monthly income"),
        "income_volatility": ("Steady income", "Irregular income"),
        "upi_txns_per_month": ("Active digital payments", "Few digital payments"),
        "recharge_regularity": ("Regular mobile recharges", "Irregular mobile recharges"),
        "bill_payment_ratio": ("Bills paid on time", "Late or missed bills"),
        "age": ("Age profile is favourable", "Age profile adds risk"),
        "dependents": ("Fewer dependents", "More dependents"),
        "land_acres": ("Owns farmland", "Little or no land"),
        "active_days": ("Works many days a month", "Few working days"),
        "tenure_months": ("Long time on platform", "Short time on platform"),
        "worker_type": ("Work profile is favourable", "Work profile adds risk"),
    },
    "తెలుగు": {
        "monthly_income": ("మంచి నెలవారీ ఆదాయం", "తక్కువ నెలవారీ ఆదాయం"),
        "income_volatility": ("స్థిరమైన ఆదాయం", "అస్థిరమైన ఆదాయం"),
        "upi_txns_per_month": ("డిజిటల్ చెల్లింపులు ఎక్కువ", "డిజిటల్ చెల్లింపులు తక్కువ"),
        "recharge_regularity": ("క్రమం తప్పని రీఛార్జ్", "క్రమం లేని రీఛార్జ్"),
        "bill_payment_ratio": ("బిల్లులు సమయానికి చెల్లింపు", "బిల్లులు ఆలస్యం/చెల్లించలేదు"),
        "age": ("వయసు అనుకూలం", "వయసు వల్ల ప్రమాదం"),
        "dependents": ("ఆధారపడినవారు తక్కువ", "ఆధారపడినవారు ఎక్కువ"),
        "land_acres": ("సొంత వ్యవసాయ భూమి ఉంది", "భూమి తక్కువ/లేదు"),
        "active_days": ("నెలలో ఎక్కువ రోజులు పని", "పని రోజులు తక్కువ"),
        "tenure_months": ("ప్లాట్‌ఫారమ్‌లో ఎక్కువ కాలం", "ప్లాట్‌ఫారమ్‌లో తక్కువ కాలం"),
        "worker_type": ("పని ప్రొఫైల్ అనుకూలం", "పని ప్రొఫైల్ వల్ల ప్రమాదం"),
    },
}

TIPS = {
    "English": {
        "bill_payment_ratio": "Pay electricity, water and phone bills on time every month.",
        "recharge_regularity": "Recharge your mobile on a regular schedule.",
        "upi_txns_per_month": "Use UPI more often for daily payments to build a digital record.",
        "income_volatility": "Try to keep a steady income, for example by saving during good months.",
        "dependents": "Plan expenses carefully and keep an emergency fund.",
        "monthly_income": "Add a second income source if possible.",
    },
    "తెలుగు": {
        "bill_payment_ratio": "కరెంటు, నీరు, ఫోన్ బిల్లులు ప్రతి నెల సమయానికి చెల్లించండి.",
        "recharge_regularity": "మొబైల్ రీఛార్జ్ క్రమం తప్పకుండా చేయండి.",
        "upi_txns_per_month": "రోజువారీ చెల్లింపులకు UPI ఎక్కువగా వాడండి.",
        "income_volatility": "మంచి నెలల్లో పొదుపు చేసి ఆదాయాన్ని స్థిరంగా ఉంచండి.",
        "dependents": "ఖర్చులను ప్లాన్ చేసి అత్యవసర నిధి ఉంచండి.",
        "monthly_income": "వీలైతే రెండో ఆదాయ మార్గం చూడుకోండి.",
    },
}

lang = st.sidebar.radio("Language / భాష", list(T.keys()))
t = T[lang]

st.title(t["title"])
st.caption(t["sub"])

# ---------- Inputs ----------
wt = st.radio(t["type"], [t["farmer"], t["gig"]], horizontal=True)
is_gig = 1 if wt == t["gig"] else 0

c1, c2 = st.columns(2)
income = c1.number_input(t["income"], 3000, 100000, 15000, step=500)
vol = c2.slider(t["vol"], 0.0, 1.0, 0.3)
upi = c1.number_input(t["upi"], 0, 300, 30)
rech = c2.slider(t["rech"], 0.0, 1.0, 0.8)
bill = c1.slider(t["bill"], 0.0, 1.0, 0.8)
age = c2.number_input(t["age"], 18, 80, 35)
dep = c1.number_input(t["dep"], 0, 10, 2)

if is_gig:
    days = c2.number_input(t["days"], 0, 30, 22)
    tenure = c1.number_input(t["tenure"], 0, 120, 12)
    land = 0.0
else:
    land = c2.number_input(t["land"], 0.0, 50.0, 2.0, step=0.5)
    days, tenure = 20, 24   # same defaults used in training for farmers

# ---------- Predict ----------
if st.button(t["btn"], type="primary"):
    row = pd.DataFrame([{
        "worker_type": is_gig, "monthly_income": income,
        "income_volatility": vol, "upi_txns_per_month": upi,
        "recharge_regularity": rech, "bill_payment_ratio": bill,
        "age": age, "dependents": dep, "land_acres": land,
        "active_days": days, "tenure_months": tenure,
    }])[cols]

    p_default = float(model.predict_proba(row)[0, 1])
    score = int(round(np.clip((1 - p_default) * 100, 0, 100)))

    # Loan limit: up to 4x monthly income, scaled by score
    loan = int(round(income * 4 * (score / 100) ** 1.5 / 500) * 500)

    if score >= 60:
        msg, color = t["low"], "green"
    elif score >= 40:
        msg, color = t["mid"], "orange"
    else:
        msg, color = t["high"], "red"

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": t["score"]},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 40], "color": "#fde2e2"},
                         {"range": [40, 60], "color": "#fff3d6"},
                         {"range": [60, 100], "color": "#dff5e1"}]}))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    st.plotly_chart(fig)

    m1, m2 = st.columns(2)
    m1.metric(t["loan"], f"₹{loan:,}")
    m2.metric(t["risk"], f"{p_default*100:.0f}%")
    st.info(msg)

    # SHAP-style explanation using XGBoost's built-in TreeSHAP (no numba needed)
    sv = model.get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0][:-1]
    contrib = sorted(zip(cols, sv), key=lambda x: abs(x[1]), reverse=True)[:6]
    R = REASONS[lang]

    st.subheader(t["why"])
    good, bad = [], []
    for f, v in contrib:
        if f == "worker_type":
            continue
        # Positive value = pushes toward default (bad), negative = helps
        (bad if v > 0 else good).append(f)

    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**{t['up']}**")
        for f in good:
            st.write("• " + R[f][0])
    with cb:
        st.markdown(f"**{t['down']}**")
        for f in bad:
            st.write("• " + R[f][1])

    tips = [TIPS[lang][f] for f in bad if f in TIPS[lang]]
    if tips:
        st.subheader(t["tips"])
        for tip in tips:
            st.write("💡 " + tip)

st.divider()
st.caption("Demo uses synthetic data. Real deployment would use consented data via Account Aggregator.")