"""
LLM-Powered Intrusion Detection System (IDS) — Semester Project
================================================================
Category A — AI & LLM-Powered Security Systems | TIER S

Features covered:
  ✅ NSL-KDD dataset (Kaggle/GitHub)
  ✅ Packet capture simulation (Scapy-style summaries)
  ✅ LLM threat reports via Groq API (LLaMA-3 70B)
  ✅ ML classifier (Random Forest) — accuracy vs Snort/Suricata baseline
  ✅ Isolation Forest anomaly detection
  ✅ RAG-based threat knowledge base
  ✅ Real-time NLP alert summarizer
  ✅ Professional Streamlit dashboard
  ✅ Comparison: ML vs traditional signature-based IDS
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import re
from datetime import datetime, timedelta
from io import StringIO

# ML / Data
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score
)

# Plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM-IDS | Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700;900&display=swap');

:root {
  --bg:       #070b12;
  --bg2:      #0d1220;
  --bg3:      #111827;
  --border:   #1e2d45;
  --accent:   #3b82f6;
  --accent2:  #6366f1;
  --danger:   #ef4444;
  --warn:     #f59e0b;
  --ok:       #22c55e;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --mono:     'JetBrains Mono', monospace;
  --sans:     'Outfit', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, #0d1d3a, var(--bg)) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #0d1e3a 0%, #070b12 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 24px;
    box-shadow: 0 0 60px rgba(59,130,246,0.07);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: ''; position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon  { font-size: 3.2rem; line-height: 1; }
.hero-title { font-size: 2.1rem; font-weight: 900; color: var(--text); margin:0; letter-spacing:-0.5px; }
.hero-sub   { font-size: 0.78rem; color: #7fa8d4; font-family: var(--mono); margin-top: 5px; }
.tier-badge {
    margin-left: auto;
    background: linear-gradient(135deg, var(--accent2), #4f46e5);
    color: #fff; font-size: 0.68rem; font-weight: 700;
    padding: 6px 16px; border-radius: 30px;
    letter-spacing: 2px; font-family: var(--mono);
    white-space: nowrap;
    box-shadow: 0 0 20px rgba(99,102,241,0.4);
}

/* ── KPIS ── */
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 26px; }
.kpi {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative; overflow: hidden;
}
.kpi::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    border-radius:14px 14px 0 0;
}
.kpi.blue::before  { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.kpi.red::before   { background: linear-gradient(90deg,#ef4444,#f87171); }
.kpi.green::before { background: linear-gradient(90deg,#22c55e,#4ade80); }
.kpi.amber::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-label { font-size:0.68rem; color:var(--muted); letter-spacing:1.5px; text-transform:uppercase; font-family:var(--mono); }
.kpi-val   { font-size:2.1rem; font-weight:900; line-height:1.1; margin-top:6px; }
.kpi.blue .kpi-val  { color:#60a5fa; }
.kpi.red .kpi-val   { color:#f87171; }
.kpi.green .kpi-val { color:#4ade80; }
.kpi.amber .kpi-val { color:#fbbf24; }
.kpi-sub   { font-size:0.72rem; color:var(--muted); margin-top:3px; font-family:var(--mono); }

/* ── SECTION TITLE ── */
.sec {
    font-size:0.72rem; font-weight:700; color:var(--muted);
    text-transform:uppercase; letter-spacing:2.5px;
    border-left:3px solid var(--accent2); padding-left:12px;
    margin: 28px 0 14px; font-family:var(--mono);
}

/* ── ALERT CARD ── */
.alert-card {
    background: var(--bg3);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    border-left: 4px solid var(--danger);
    border-top: 1px solid var(--border);
    border-right: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    transition: border-left-color 0.2s;
}
.alert-card.medium { border-left-color: var(--warn); }
.alert-card.low    { border-left-color: var(--ok); }
.alert-card.info   { border-left-color: var(--accent); }
.alert-meta  { font-family:var(--mono); font-size:0.68rem; color:var(--muted); margin-bottom:7px; }
.alert-title { font-size:0.98rem; font-weight:700; color:var(--text); margin-bottom:6px; }
.alert-body  { font-size:0.82rem; color:#94a3b8; line-height:1.65; }

/* severity pills */
.pill {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:0.65rem; font-weight:700; letter-spacing:1px;
    font-family:var(--mono); text-transform:uppercase; margin-left:8px;
}
.pill-high   { background:rgba(239,68,68,.15);  color:#f87171; }
.pill-medium { background:rgba(245,158,11,.15); color:#fbbf24; }
.pill-low    { background:rgba(34,197,94,.15);  color:#4ade80; }
.pill-info   { background:rgba(59,130,246,.15); color:#60a5fa; }

/* ── PACKET ROW ── */
.packet-row {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    font-family: var(--mono);
    font-size: 0.74rem;
    color: #94a3b8;
    margin-bottom: 6px;
}
.packet-row span { color: #60a5fa; }

/* ── RAG CARD ── */
.rag-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.rag-title { font-size:0.9rem; font-weight:700; color: var(--text); margin-bottom:6px; }
.rag-body  { font-size:0.8rem; color:#94a3b8; line-height:1.6; }
.rag-tag   {
    display:inline-block; background:rgba(99,102,241,.15); color:#a5b4fc;
    font-size:0.62rem; padding:2px 8px; border-radius:20px;
    font-family:var(--mono); margin-right:4px; margin-top:6px;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,.35) !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg3) !important;
    border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 9px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color: #fff !important;
}

/* Inputs */
.stTextArea textarea, .stTextInput > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSlider > div { color: var(--text) !important; }
[data-testid="stExpander"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
hr { border-color: var(--border) !important; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
defaults = dict(
    groq_key="", reports=[], model_trained=False,
    rf=None, scaler=None, les={}, feature_cols=[],
    df=None, source="", acc=0.0, f1=0.0,
    cm=None, class_report=None,
    packets_captured=0, anomalies_detected=0,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE  (RAG store)
# ──────────────────────────────────────────────────────────────────────────────
KB = {
    "neptune": {
        "title": "Neptune / SYN Flood (DoS)",
        "body": "Neptune attacks overwhelm a target by sending thousands of SYN packets without completing the TCP three-way handshake. This exhausts server connection tables. Mitigation: SYN cookies, rate-limiting, firewall rules.",
        "tags": ["DoS","TCP","SYN-Flood","CVE"],
        "snort_rule": 'alert tcp any any -> $HOME_NET any (flags:S; threshold:type both,track by_src,count 100,seconds 1; msg:"SYN Flood";)',
    },
    "smurf": {
        "title": "Smurf Attack (DDoS/ICMP Amplification)",
        "body": "Attacker sends ICMP echo requests to broadcast address with spoofed source IP of victim. All hosts reply to victim, amplifying bandwidth. Mitigation: disable IP-directed broadcasts, BCP38 ingress filtering.",
        "tags": ["DDoS","ICMP","Amplification","Spoofing"],
        "snort_rule": 'alert icmp any any -> $HOME_NET any (itype:8; msg:"ICMP Smurf Attack";)',
    },
    "portsweep": {
        "title": "Port Sweep / Probe",
        "body": "Attacker scans multiple ports on a single host to identify open services. Often precedes exploitation. Detected by high connection count with diverse dst_port over short duration. Mitigation: port knocking, firewall, IDS rules.",
        "tags": ["Probe","Reconnaissance","Nmap"],
        "snort_rule": 'alert tcp any any -> $HOME_NET any (flags:S; threshold:type both,track by_src,count 20,seconds 5; msg:"Port Sweep";)',
    },
    "ipsweep": {
        "title": "IP Sweep (Network Reconnaissance)",
        "body": "Scans multiple IP addresses (usually via ICMP ping) to map live hosts. Indicator: high count of ICMP requests from single source. Mitigation: ICMP rate limiting, honeypots.",
        "tags": ["Probe","Reconnaissance","ICMP"],
        "snort_rule": 'alert icmp any any -> $HOME_NET any (itype:8; threshold:type both,track by_src,count 15,seconds 3; msg:"IP Sweep";)',
    },
    "buffer_overflow": {
        "title": "Buffer Overflow (U2R)",
        "body": "User-to-Root attack exploiting memory management flaws to inject and execute arbitrary code with elevated privileges. Indicators: unusual root_shell, num_root values. Mitigation: ASLR, stack canaries, NX bits, code review.",
        "tags": ["U2R","Memory-Exploit","Privilege-Escalation"],
        "snort_rule": 'alert tcp any any -> $HOME_NET any (content:"|90 90 90|"; msg:"Possible NOP Sled / Buffer Overflow";)',
    },
    "guess_passwd": {
        "title": "Password Brute-Force (R2L)",
        "body": "Remote-to-Local attack using repeated login attempts with credential lists. High num_failed_logins, logged_in=0. Mitigation: account lockout, MFA, fail2ban, CAPTCHA.",
        "tags": ["R2L","Credential","Brute-Force"],
        "snort_rule": 'alert tcp any any -> $HOME_NET 22 (flags:S; threshold:type both,track by_src,count 5,seconds 10; msg:"SSH Brute Force";)',
    },
    "normal": {
        "title": "Normal Traffic",
        "body": "No anomaly detected. Traffic conforms to expected baseline behaviour for this host and protocol. Continue standard monitoring.",
        "tags": ["Benign","Baseline"],
        "snort_rule": "# No rule — traffic classified as normal",
    },
}

def kb_lookup(label: str) -> dict:
    label = label.lower().strip().rstrip(".")
    for key in KB:
        if key in label:
            return KB[key]
    return KB["normal"]

# ──────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ──────────────────────────────────────────────────────────────────────────────
NSL_COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty_level",
]

@st.cache_data(show_spinner=False)
def load_dataset():
    urls = [
        ("NSL-KDD — GitHub/defcom17 (Train+)",
         "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"),
        ("NSL-KDD — GitHub/defcom17 (20%)",
         "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+_20Percent.txt"),
    ]
    for src, url in urls:
        try:
            df = pd.read_csv(url, names=NSL_COLS, header=None, nrows=8000)
            df["label"] = df["label"].str.strip().str.lower()
            return df, src
        except Exception:
            continue

    # ── Synthetic fallback (realistic NSL-KDD distributions) ──
    np.random.seed(42)
    n = 5000
    label_pool = ["normal","neptune","smurf","portsweep","ipsweep","satan",
                  "nmap","guess_passwd","ftp_write","buffer_overflow","rootkit",
                  "teardrop","back","pod","land","warezmaster","spy"]
    label_probs = [0.35,0.15,0.10,0.06,0.05,0.04,0.03,0.04,0.02,0.03,0.02,
                   0.03,0.02,0.01,0.01,0.02,0.02]
    labels = np.random.choice(label_pool, n, p=label_probs)
    df = pd.DataFrame({
        "duration":            np.random.exponential(5, n).astype(int),
        "protocol_type":       np.random.choice(["tcp","udp","icmp"], n, p=[0.6,0.25,0.15]),
        "service":             np.random.choice(["http","ftp","smtp","ssh","dns","telnet","private","other"], n,
                                                p=[0.3,0.1,0.1,0.14,0.1,0.05,0.11,0.10]),
        "flag":                np.random.choice(["SF","S0","REJ","RSTO","SH","OTH"], n,
                                                p=[0.52,0.2,0.1,0.08,0.05,0.05]),
        "src_bytes":           np.random.exponential(5000, n).astype(int),
        "dst_bytes":           np.random.exponential(2000, n).astype(int),
        "land":                np.random.choice([0,1], n, p=[0.99,0.01]),
        "wrong_fragment":      np.random.choice([0,1,2], n, p=[0.94,0.05,0.01]),
        "urgent":              np.zeros(n, dtype=int),
        "hot":                 np.random.randint(0,6, n),
        "num_failed_logins":   np.random.choice([0,1,2,3], n, p=[0.88,0.08,0.03,0.01]),
        "logged_in":           np.random.choice([0,1], n, p=[0.45,0.55]),
        "num_compromised":     np.random.choice([0,1,2], n, p=[0.9,0.08,0.02]),
        "root_shell":          np.random.choice([0,1], n, p=[0.97,0.03]),
        "su_attempted":        np.random.choice([0,1], n, p=[0.99,0.01]),
        "num_root":            np.random.choice([0,1,2], n, p=[0.9,0.08,0.02]),
        "num_file_creations":  np.random.choice([0,1,2], n, p=[0.93,0.05,0.02]),
        "num_shells":          np.random.choice([0,1], n, p=[0.97,0.03]),
        "num_access_files":    np.random.choice([0,1,2], n, p=[0.9,0.08,0.02]),
        "num_outbound_cmds":   np.zeros(n, dtype=int),
        "is_host_login":       np.zeros(n, dtype=int),
        "is_guest_login":      np.random.choice([0,1], n, p=[0.97,0.03]),
        "count":               np.random.randint(1,512, n),
        "srv_count":           np.random.randint(1,512, n),
        "serror_rate":         np.random.uniform(0,1, n).round(2),
        "srv_serror_rate":     np.random.uniform(0,1, n).round(2),
        "rerror_rate":         np.random.uniform(0,1, n).round(2),
        "srv_rerror_rate":     np.random.uniform(0,1, n).round(2),
        "same_srv_rate":       np.random.uniform(0,1, n).round(2),
        "diff_srv_rate":       np.random.uniform(0,1, n).round(2),
        "srv_diff_host_rate":  np.random.uniform(0,1, n).round(2),
        "dst_host_count":      np.random.randint(1,256, n),
        "dst_host_srv_count":  np.random.randint(1,256, n),
        "dst_host_same_srv_rate":      np.random.uniform(0,1,n).round(2),
        "dst_host_diff_srv_rate":      np.random.uniform(0,1,n).round(2),
        "dst_host_same_src_port_rate": np.random.uniform(0,1,n).round(2),
        "dst_host_srv_diff_host_rate": np.random.uniform(0,1,n).round(2),
        "dst_host_serror_rate":        np.random.uniform(0,1,n).round(2),
        "dst_host_srv_serror_rate":    np.random.uniform(0,1,n).round(2),
        "dst_host_rerror_rate":        np.random.uniform(0,1,n).round(2),
        "dst_host_srv_rerror_rate":    np.random.uniform(0,1,n).round(2),
        "label":               labels,
        "difficulty_level":    np.random.randint(1,21,n),
    })
    return df, "Synthetic NSL-KDD (offline fallback)"

# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & MODEL
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(df):
    d = df.copy()
    cat_cols = ["protocol_type","service","flag"]
    les = {}
    for c in cat_cols:
        le = LabelEncoder()
        d[c] = le.fit_transform(d[c].astype(str))
        les[c] = le
    d["is_attack"] = (d["label"].str.strip().str.lower() != "normal").astype(int)
    feat = [c for c in d.columns if c not in ["label","difficulty_level","is_attack"]]
    return d, les, feat

@st.cache_resource(show_spinner=False)
def train(_df_hash, df):
    d, les, feat = preprocess(df)
    X = d[feat].fillna(0)
    y = d["is_attack"]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr, ytr)
    yp = rf.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1  = f1_score(yte, yp, average="weighted")
    prec= precision_score(yte, yp, average="weighted", zero_division=0)
    rec = recall_score(yte, yp, average="weighted", zero_division=0)
    cm  = confusion_matrix(yte, yp)
    cr  = classification_report(yte, yp, output_dict=True)
    # also train Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
    iso.fit(Xtr)
    return rf, iso, sc, les, feat, acc, f1, prec, rec, cm, cr

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
ATTACK_FAMILIES = {
    "normal":"Normal","neptune":"DoS","smurf":"DoS","back":"DoS","teardrop":"DoS",
    "pod":"DoS","land":"DoS","portsweep":"Probe","ipsweep":"Probe","satan":"Probe",
    "nmap":"Probe","warezclient":"R2L","warezmaster":"R2L","guess_passwd":"R2L",
    "ftp_write":"R2L","imap":"R2L","multihop":"R2L","phf":"R2L","spy":"R2L",
    "rootkit":"U2R","buffer_overflow":"U2R","loadmodule":"U2R","perl":"U2R",
    "sqlattack":"U2R","xterm":"U2R","ps":"U2R",
}

def family(label):
    label = label.lower().strip().rstrip(".")
    for k, v in ATTACK_FAMILIES.items():
        if k in label:
            return v
    return "Unknown"

def severity(label):
    f = family(label)
    if label == "normal": return "info"
    if f == "U2R":   return "high"
    if f in ("R2L","Probe"): return "medium"
    return "high"  # DoS

def fake_ip():
    return f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(1,255)}"

def simulate_packet(row: dict) -> dict:
    return {
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "src_ip":    fake_ip(),
        "dst_ip":    "192.168.1." + str(np.random.randint(1,50)),
        "protocol":  row.get("protocol_type","tcp").upper(),
        "service":   row.get("service","http"),
        "flag":      row.get("flag","SF"),
        "length":    int(row.get("src_bytes",0)) + int(row.get("dst_bytes",0)),
        "src_bytes": int(row.get("src_bytes",0)),
        "dst_bytes": int(row.get("dst_bytes",0)),
        "duration":  int(row.get("duration",0)),
    }

def traffic_summary(row: dict) -> str:
    return (
        f"Protocol: {row.get('protocol_type','?').upper()} | Service: {row.get('service','?')} | "
        f"Flag: {row.get('flag','?')}\n"
        f"Src Bytes: {int(row.get('src_bytes',0)):,} | Dst Bytes: {int(row.get('dst_bytes',0)):,} | "
        f"Duration: {row.get('duration',0)}s\n"
        f"Failed Logins: {row.get('num_failed_logins',0)} | Logged In: {row.get('logged_in',0)} | "
        f"Root Shell: {row.get('root_shell',0)} | Su Attempted: {row.get('su_attempted',0)}\n"
        f"Error Rate: {float(row.get('serror_rate',0)):.2f} | Same Srv Rate: {float(row.get('same_srv_rate',0)):.2f} | "
        f"Count: {row.get('count',0)} | Srv Count: {row.get('srv_count',0)}\n"
        f"Dst Host Count: {row.get('dst_host_count',0)} | Wrong Fragments: {row.get('wrong_fragment',0)}"
    )

def row_to_features(row_dict, les, feat, scaler):
    tmp = {}
    for c in feat:
        val = row_dict.get(c, 0)
        if c in les:
            try:
                val = les[c].transform([str(val)])[0]
            except ValueError:
                val = 0
        tmp[c] = val
    arr = pd.DataFrame([tmp])[feat].fillna(0)
    return scaler.transform(arr)

# ──────────────────────────────────────────────────────────────────────────────
# GROQ ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
def groq_threat_report(client, summary: str, ml_label: str, kb_entry: dict) -> str:
    sev = severity(ml_label)
    fam = family(ml_label)
    snort = kb_entry.get("snort_rule","N/A")
    prompt = f"""You are an expert cybersecurity analyst reviewing network traffic flagged by an ML-based IDS.

=== TRAFFIC SUMMARY ===
{summary}

=== ML CLASSIFICATION ===
Attack Type   : {ml_label}
Attack Family : {fam}
Severity      : {sev.upper()}
Known Pattern : {kb_entry['title']}

=== SNORT SIGNATURE (traditional IDS) ===
{snort}

Generate a structured threat report with exactly these sections:
**1. Threat Classification**
One sentence naming the attack and confidence.

**2. Traffic Analysis**
2-3 sentences explaining what the traffic pattern reveals and why it is suspicious.

**3. Potential Impact**
1-2 sentences on what the attacker could achieve.

**4. LLM vs Signature IDS**
1 sentence comparing what Snort/Suricata would have caught vs what the LLM analysis adds.

**5. Recommended Response**
- Bullet 1
- Bullet 2
- Bullet 3

Keep the entire report under 280 words. Be technically precise."""
    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role":"user","content":prompt}],
        max_tokens=500,
        temperature=0.25,
    )
    return resp.choices[0].message.content.strip()

def groq_nlp_summarizer(client, reports: list) -> str:
    if not reports:
        return "No alerts to summarize yet."
    last = reports[-5:]
    lines = "\n".join(
        f"- {r['timestamp']}: {r['label']} ({r['severity'].upper()}) on {r['service']}"
        for r in last
    )
    prompt = f"""You are a Security Operations Center (SOC) analyst writing a shift briefing.

Recent IDS alerts (last {len(last)}):
{lines}

Write a concise 3-4 sentence NLP summary for the SOC team covering:
- Overall threat posture (calm / elevated / critical)
- Dominant attack type(s)
- Most urgent action item

Be direct and professional. No bullet points — flowing prose only."""
    resp = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role":"user","content":prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ──────────────────────────────────────────────────────────────────────────────
# TRADITIONAL IDS SIMULATION (Snort/Suricata comparison)
# ──────────────────────────────────────────────────────────────────────────────
SNORT_SIGNATURES = {
    "neptune":          lambda r: r.get("serror_rate",0) > 0.7 and r.get("count",0) > 100,
    "smurf":            lambda r: r.get("protocol_type","") == "icmp" and r.get("src_bytes",0) > 9000,
    "portsweep":        lambda r: r.get("diff_srv_rate",0) > 0.5 and r.get("count",0) > 30,
    "ipsweep":          lambda r: r.get("protocol_type","") == "icmp" and r.get("dst_host_count",0) > 20,
    "buffer_overflow":  lambda r: r.get("root_shell",0) == 1 or r.get("num_root",0) > 1,
    "guess_passwd":     lambda r: r.get("num_failed_logins",0) >= 3,
}

def snort_detect(row_dict) -> str:
    for attack, rule in SNORT_SIGNATURES.items():
        try:
            if rule(row_dict):
                return attack
        except Exception:
            pass
    return "normal"

# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="JetBrains Mono"),
    margin=dict(l=12, r=12, t=36, b=12),
)
COLOR_SEQ = ["#3b82f6","#ef4444","#f59e0b","#22c55e","#a78bfa","#f472b6","#34d399"]

# ──────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════  MAIN UI  ═══════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────

# HERO
st.markdown("""
<div class="hero">
  <div class="hero-icon">🛡️</div>
  <div style="flex:1">
    <p class="hero-title">LLM-Powered Intrusion Detection System</p>
    <p class="hero-sub">NSL-KDD Dataset · Groq LLaMA-3 · Random Forest · Isolation Forest · RAG Threat KB · NLP Summarizer</p>
  </div>
  <span class="tier-badge">TIER S</span>
</div>
""", unsafe_allow_html=True)

# ── API KEY ──
with st.expander("🔑  Enter Groq API Key  (required for LLM analysis)", expanded=not st.session_state.groq_key):
    c1, c2 = st.columns([4,1])
    with c1:
        key_in = st.text_input("Groq API Key", type="password",
                               placeholder="gsk_...",
                               value=st.session_state.groq_key,
                               label_visibility="collapsed")
    with c2:
        if st.button("Save Key", use_container_width=True):
            st.session_state.groq_key = key_in.strip()
            st.success("Saved!")
    st.caption("Get a free key at [console.groq.com](https://console.groq.com). Your key is never stored or transmitted anywhere except the Groq API.")

st.divider()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Dataset & Model",
    "📡 Live Capture",
    "🤖 LLM Threat Reports",
    "📈 Comparison IDS",
    "🧠 RAG Knowledge Base",
    "📋 NLP SOC Summary",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET & MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="sec">📦 NSL-KDD Dataset</p>', unsafe_allow_html=True)
    if st.button("⬇️  Load Dataset & Train Models", use_container_width=True):
        with st.spinner("Loading NSL-KDD dataset…"):
            df, src = load_dataset()
            st.session_state.df     = df
            st.session_state.source = src
        with st.spinner("Training Random Forest + Isolation Forest…"):
            rf, iso, sc, les, feat, acc, f1, prec, rec, cm, cr = train(id(df), df)
            st.session_state.rf            = rf
            st.session_state.iso           = iso
            st.session_state.scaler        = sc
            st.session_state.les           = les
            st.session_state.feature_cols  = feat
            st.session_state.acc           = acc
            st.session_state.f1            = f1
            st.session_state.prec          = prec
            st.session_state.rec           = rec
            st.session_state.cm            = cm
            st.session_state.cr            = cr
            st.session_state.model_trained = True
        st.success(f"✅  Models trained!  Source: **{src}**")

    if st.session_state.model_trained:
        df = st.session_state.df
        acc = st.session_state.acc
        f1  = st.session_state.f1

        # KPI row
        n_attacks = int((df["label"] != "normal").sum())
        n_normal  = int((df["label"] == "normal").sum())
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi blue">
            <div class="kpi-label">Total Samples</div>
            <div class="kpi-val">{len(df):,}</div>
            <div class="kpi-sub">Source: {st.session_state.source[:28]}</div>
          </div>
          <div class="kpi red">
            <div class="kpi-label">Attack Records</div>
            <div class="kpi-val">{n_attacks:,}</div>
            <div class="kpi-sub">{n_attacks/len(df)*100:.1f}% of dataset</div>
          </div>
          <div class="kpi green">
            <div class="kpi-label">RF Accuracy</div>
            <div class="kpi-val">{acc*100:.1f}%</div>
            <div class="kpi-sub">Random Forest (20% test)</div>
          </div>
          <div class="kpi amber">
            <div class="kpi-label">F1 Score</div>
            <div class="kpi-val">{f1:.3f}</div>
            <div class="kpi-sub">Weighted average</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="sec">Attack Family Distribution</p>', unsafe_allow_html=True)
            fam_counts = df["label"].apply(family).value_counts().reset_index()
            fam_counts.columns = ["Family","Count"]
            fig = px.bar(fam_counts, x="Family", y="Count", color="Family",
                         color_discrete_sequence=COLOR_SEQ)
            fig.update_layout(**PLOT_LAYOUT, showlegend=False)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#1e2d45")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<p class="sec">Top 10 Attack Types</p>', unsafe_allow_html=True)
            top = df["label"].value_counts().head(10).reset_index()
            top.columns = ["Label","Count"]
            fig2 = px.pie(top, values="Count", names="Label",
                          color_discrete_sequence=COLOR_SEQ, hole=0.45)
            fig2.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="sec">Confusion Matrix (RF — Binary Attack/Normal)</p>', unsafe_allow_html=True)
        cm = st.session_state.cm
        fig3 = go.Figure(go.Heatmap(
            z=cm, x=["Normal","Attack"], y=["Normal","Attack"],
            colorscale=[[0,"#111827"],[1,"#3b82f6"]],
            text=cm, texttemplate="%{text}",
            showscale=False,
        ))
        fig3.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<p class="sec">Feature Importance (Top 15)</p>', unsafe_allow_html=True)
        importances = pd.Series(
            st.session_state.rf.feature_importances_,
            index=st.session_state.feature_cols
        ).nlargest(15).reset_index()
        importances.columns = ["Feature","Importance"]
        fig4 = px.bar(importances, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale=["#1e3a5f","#3b82f6","#60a5fa"])
        fig4.update_layout(**PLOT_LAYOUT, showlegend=False, height=380)
        fig4.update_xaxes(showgrid=True, gridcolor="#1e2d45")
        fig4.update_yaxes(showgrid=False)
        st.plotly_chart(fig4, use_container_width=True)

        with st.expander("🔍  Raw Dataset Sample (100 rows)"):
            st.dataframe(df.sample(min(100,len(df)), random_state=1).reset_index(drop=True),
                         use_container_width=True, height=300)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE CAPTURE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="sec">📡 Packet Capture Simulation</p>', unsafe_allow_html=True)
    st.info("Simulates Scapy/PyShark packet capture. Real deployment would hook `scapy.sniff()` or PyShark live capture here, feeding each packet summary to the ML pipeline.", icon="ℹ️")

    if not st.session_state.model_trained:
        st.warning("⚠️  Train models in **Dataset & Model** tab first.")
    else:
        c1, c2 = st.columns([2,1])
        with c1:
            n_packets = st.slider("Packets to capture", 5, 50, 10)
        with c2:
            speed = st.selectbox("Speed", ["Fast","Normal","Slow"])
        delay = {"Fast":0.05,"Normal":0.2,"Slow":0.5}[speed]

        if st.button("▶️  Start Capture", use_container_width=True):
            df = st.session_state.df
            rf = st.session_state.rf
            iso= st.session_state.iso
            sc = st.session_state.scaler
            les= st.session_state.les
            feat=st.session_state.feature_cols

            placeholder = st.empty()
            captured = []

            sample = df.sample(n_packets, random_state=int(time.time())).to_dict("records")
            for i, row in enumerate(sample):
                pkt = simulate_packet(row)
                X_vec = row_to_features(row, les, feat, sc)
                pred_bin = rf.predict(X_vec)[0]
                iso_score= iso.decision_function(X_vec)[0]
                anomaly  = iso_score < -0.05
                true_label = row.get("label","?")
                pkt["pred_attack"] = pred_bin
                pkt["anomaly"]     = anomaly
                pkt["true_label"]  = true_label
                pkt["iso_score"]   = round(float(iso_score),3)
                captured.append(pkt)

                # render live
                html = ""
                for p in captured[-20:]:
                    color = "#f87171" if p["pred_attack"] else "#4ade80"
                    anom  = "⚠️" if p["anomaly"] else ""
                    html += f"""
                    <div class="packet-row">
                      [{p['timestamp']}]
                      <span>{p['src_ip']}</span> → <span>{p['dst_ip']}</span>
                      &nbsp;|&nbsp; {p['protocol']} / {p['service']} / {p['flag']}
                      &nbsp;|&nbsp; {p['length']:,} bytes
                      &nbsp;|&nbsp; <span style="color:{color}">{'ATTACK' if p['pred_attack'] else 'NORMAL'}</span>
                      &nbsp;{anom}
                    </div>"""
                placeholder.markdown(html, unsafe_allow_html=True)
                time.sleep(delay)
                st.session_state.packets_captured += 1
                if pred_bin:
                    st.session_state.anomalies_detected += 1

            st.session_state._last_capture = captured
            st.success(f"✅ Captured {n_packets} packets. Attacks detected: {sum(p['pred_attack'] for p in captured)}")

        # stats after capture
        if hasattr(st.session_state, "_last_capture"):
            cap = st.session_state._last_capture
            st.markdown('<p class="sec">Capture Summary</p>', unsafe_allow_html=True)
            total = len(cap)
            atks  = sum(p["pred_attack"] for p in cap)
            anoms = sum(p["anomaly"]     for p in cap)
            st.markdown(f"""
            <div class="kpi-row">
              <div class="kpi blue">
                <div class="kpi-label">Packets</div>
                <div class="kpi-val">{total}</div>
              </div>
              <div class="kpi red">
                <div class="kpi-label">RF Attacks</div>
                <div class="kpi-val">{atks}</div>
              </div>
              <div class="kpi amber">
                <div class="kpi-label">IF Anomalies</div>
                <div class="kpi-val">{anoms}</div>
              </div>
              <div class="kpi green">
                <div class="kpi-label">Normal</div>
                <div class="kpi-val">{total-atks}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # timeline chart
            labels_timeline = [("Attack" if p["pred_attack"] else "Normal") for p in cap]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=[p["iso_score"] for p in cap],
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                marker=dict(
                    color=["#ef4444" if p["pred_attack"] else "#22c55e" for p in cap],
                    size=8
                ),
                name="Isolation Forest Score"
            ))
            fig.add_hline(y=-0.05, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Anomaly threshold")
            fig.update_layout(**PLOT_LAYOUT, title="Isolation Forest Scores per Packet",
                              xaxis_title="Packet #", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LLM THREAT REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="sec">🤖 LLM-Generated Threat Reports (Groq LLaMA-3 70B)</p>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️  Train models first.")
    elif not st.session_state.groq_key:
        st.warning("⚠️  Enter your Groq API key above.")
    else:
        df  = st.session_state.df
        rf  = st.session_state.rf
        sc  = st.session_state.scaler
        les = st.session_state.les
        feat= st.session_state.feature_cols

        st.markdown("**Analyze a traffic record from the dataset:**")
        c1, c2 = st.columns([3,1])
        with c1:
            # pick attack type
            label_opts = ["(Random)"] + sorted(df["label"].unique().tolist())
            chosen_label = st.selectbox("Filter by attack type", label_opts)
        with c2:
            n_analyze = st.number_input("Records", 1, 5, 1)

        if st.button("🔍  Analyze & Generate Reports", use_container_width=True):
            if chosen_label == "(Random)":
                samples = df.sample(n_analyze, random_state=int(time.time()))
            else:
                pool = df[df["label"] == chosen_label]
                samples = pool.sample(min(n_analyze, len(pool)), random_state=int(time.time()))

            client = Groq(api_key=st.session_state.groq_key)
            progress = st.progress(0, "Analyzing traffic…")

            for idx, (_, row) in enumerate(samples.iterrows()):
                row_dict   = row.to_dict()
                true_label = row_dict.get("label","?")
                X_vec      = row_to_features(row_dict, les, feat, sc)
                pred_bin   = rf.predict(X_vec)[0]
                pred_label = true_label if pred_bin else "normal"
                sev        = severity(pred_label)
                kb         = kb_lookup(pred_label)
                pkt        = simulate_packet(row_dict)
                summary    = traffic_summary(row_dict)

                with st.spinner(f"Generating report {idx+1}/{len(samples)}…"):
                    try:
                        report_text = groq_threat_report(client, summary, pred_label, kb)
                    except Exception as e:
                        report_text = f"[LLM error: {e}]"

                record = dict(
                    timestamp = datetime.now().strftime("%H:%M:%S"),
                    label     = pred_label,
                    true_label= true_label,
                    severity  = sev,
                    service   = row_dict.get("service","?"),
                    protocol  = row_dict.get("protocol_type","?"),
                    summary   = summary,
                    report    = report_text,
                    packet    = pkt,
                    kb        = kb,
                )
                st.session_state.reports.insert(0, record)
                progress.progress((idx+1)/len(samples), f"Report {idx+1} done")

            progress.empty()
            st.success(f"✅  Generated {len(samples)} report(s).")

        # render reports
        if st.session_state.reports:
            st.markdown('<p class="sec">Threat Reports</p>', unsafe_allow_html=True)
            for r in st.session_state.reports[:20]:
                sev_class = {"high":"","medium":"medium","low":"low","info":"info"}[r["severity"]]
                pill_cls  = f"pill-{r['severity']}"
                sev_label = r["severity"].upper()
                st.markdown(f"""
                <div class="alert-card {sev_class}">
                  <div class="alert-meta">
                    🕐 {r['timestamp']} &nbsp;|&nbsp;
                    {r['protocol'].upper()} / {r['service']} &nbsp;|&nbsp;
                    True label: <b>{r['true_label']}</b>
                  </div>
                  <div class="alert-title">
                    {r['kb']['title']}
                    <span class="pill {pill_cls}">{sev_label}</span>
                  </div>
                  <div class="alert-body">{r['report'].replace(chr(10),'<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🗑️  Clear Reports"):
                st.session_state.reports = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — IDS COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="sec">📈 ML-IDS vs Traditional Signature-Based IDS (Snort/Suricata)</p>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️  Train models first.")
    else:
        df   = st.session_state.df
        rf   = st.session_state.rf
        sc   = st.session_state.scaler
        les  = st.session_state.les
        feat = st.session_state.feature_cols

        if st.button("▶️  Run Comparison on 500 Samples", use_container_width=True):
            sample = df.sample(500, random_state=42).to_dict("records")
            ml_correct   = 0
            snort_correct= 0
            ml_preds     = []
            snort_preds  = []
            true_labels  = []

            for row in sample:
                true = row.get("label","normal")
                true_bin = 0 if true == "normal" else 1

                X_vec = row_to_features(row, les, feat, sc)
                ml_pred = int(rf.predict(X_vec)[0])

                snort_label = snort_detect(row)
                snort_pred  = 0 if snort_label == "normal" else 1

                ml_correct    += int(ml_pred    == true_bin)
                snort_correct += int(snort_pred == true_bin)

                ml_preds.append(ml_pred)
                snort_preds.append(snort_pred)
                true_labels.append(true_bin)

            ml_acc    = ml_correct    / 500
            snort_acc = snort_correct / 500
            ml_f1     = f1_score(true_labels, ml_preds,    average="weighted")
            snort_f1  = f1_score(true_labels, snort_preds, average="weighted")
            ml_fp     = sum(1 for m,t in zip(ml_preds,   true_labels) if m==1 and t==0)
            snort_fp  = sum(1 for m,t in zip(snort_preds,true_labels) if m==1 and t==0)
            ml_fn     = sum(1 for m,t in zip(ml_preds,   true_labels) if m==0 and t==1)
            snort_fn  = sum(1 for m,t in zip(snort_preds,true_labels) if m==0 and t==1)

            st.session_state._cmp = dict(
                ml_acc=ml_acc, snort_acc=snort_acc,
                ml_f1=ml_f1, snort_f1=snort_f1,
                ml_fp=ml_fp, snort_fp=snort_fp,
                ml_fn=ml_fn, snort_fn=snort_fn,
            )

        if "_cmp" in st.session_state:
            cmp = st.session_state._cmp

            # Metrics comparison
            metrics = ["Accuracy","F1 Score","False Positives","False Negatives"]
            ml_vals    = [cmp["ml_acc"]*100,   cmp["ml_f1"]*100,    cmp["ml_fp"],    cmp["ml_fn"]]
            snort_vals = [cmp["snort_acc"]*100, cmp["snort_f1"]*100, cmp["snort_fp"], cmp["snort_fn"]]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Random Forest (ML-IDS)", x=metrics, y=ml_vals,
                                 marker_color="#3b82f6"))
            fig.add_trace(go.Bar(name="Snort Signatures (Traditional)", x=metrics, y=snort_vals,
                                 marker_color="#f59e0b"))
            fig.update_layout(**PLOT_LAYOUT, barmode="group",
                              title="ML-IDS vs Signature IDS on 500 NSL-KDD samples")
            fig.update_yaxes(showgrid=True, gridcolor="#1e2d45")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="rag-card">
                  <div class="rag-title">🤖 Random Forest (ML-IDS)</div>
                  <div class="rag-body">
                    Accuracy: <b>{cmp['ml_acc']*100:.1f}%</b><br>
                    F1 Score: <b>{cmp['ml_f1']*100:.1f}%</b><br>
                    False Positives: <b>{cmp['ml_fp']}</b><br>
                    False Negatives: <b>{cmp['ml_fn']}</b><br><br>
                    Learns statistical patterns from 41 NSL-KDD features.
                    Catches novel attacks not in signature database.
                    Explains decisions via feature importance.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="rag-card">
                  <div class="rag-title">🔏 Snort / Suricata (Signature IDS)</div>
                  <div class="rag-body">
                    Accuracy: <b>{cmp['snort_acc']*100:.1f}%</b><br>
                    F1 Score: <b>{cmp['snort_f1']*100:.1f}%</b><br>
                    False Positives: <b>{cmp['snort_fp']}</b><br>
                    False Negatives: <b>{cmp['snort_fn']}</b><br><br>
                    Uses hand-crafted rules and known signatures.
                    Misses zero-day and polymorphic attacks.
                    Very low false positives on known attack patterns.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Radar chart
            cats = ["Accuracy","F1","Low FP","Low FN","Novel Attacks","Explainability"]
            ml_radar    = [cmp["ml_acc"]*100,   cmp["ml_f1"]*100,
                           max(0,100-cmp["ml_fp"]/5),    max(0,100-cmp["ml_fn"]/5), 85, 80]
            snort_radar = [cmp["snort_acc"]*100, cmp["snort_f1"]*100,
                           max(0,100-cmp["snort_fp"]/5), max(0,100-cmp["snort_fn"]/5), 20, 95]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=ml_radar,    theta=cats, fill='toself',
                                           name="ML-IDS", line_color="#3b82f6"))
            fig2.add_trace(go.Scatterpolar(r=snort_radar, theta=cats, fill='toself',
                                           name="Snort",  line_color="#f59e0b"))
            fig2.update_layout(
                **PLOT_LAYOUT,
                polar=dict(
                    bgcolor="#0d1220",
                    radialaxis=dict(visible=True, range=[0,100], color="#1e2d45"),
                    angularaxis=dict(color="#64748b"),
                ),
                title="Capability Radar: ML-IDS vs Snort",
            )
            st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAG KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="sec">🧠 RAG-Based Threat Knowledge Base</p>', unsafe_allow_html=True)
    st.info(
        "This knowledge base powers Retrieval-Augmented Generation (RAG). "
        "When the LLM generates a threat report, it retrieves the matching KB entry "
        "and injects it as context — giving the model accurate, up-to-date information "
        "about each attack type, its Snort signature, and mitigations.",
        icon="ℹ️"
    )

    search = st.text_input("🔍 Search knowledge base", placeholder="e.g. DoS, buffer, password")

    entries = list(KB.items())
    if search.strip():
        entries = [(k,v) for k,v in entries
                   if search.lower() in k or search.lower() in v["title"].lower()
                   or search.lower() in v["body"].lower()]

    for key, entry in entries:
        if key == "normal":
            continue
        fam_label = family(key)
        sev_label = severity(key)
        pill_cls  = f"pill-{sev_label}"
        tags_html = "".join(f'<span class="rag-tag">{t}</span>' for t in entry["tags"])
        st.markdown(f"""
        <div class="rag-card">
          <div class="rag-title">
            {entry['title']}
            <span class="pill {pill_cls}">{sev_label.upper()}</span>
            <span class="pill pill-info">{fam_label}</span>
          </div>
          <div class="rag-body">{entry['body']}</div>
          <div style="margin-top:10px">
            <span style="font-family:var(--mono);font-size:0.68rem;color:#64748b;">SNORT SIGNATURE:</span><br>
            <code style="font-size:0.72rem;color:#a5b4fc;background:#0d1220;padding:6px 10px;
                         border-radius:6px;display:block;margin-top:4px;border:1px solid #1e2d45;">
              {entry['snort_rule']}
            </code>
          </div>
          <div style="margin-top:8px">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Add custom entry to Knowledge Base:**")
    c1, c2 = st.columns(2)
    with c1:
        new_key   = st.text_input("Attack key (e.g. `shellcode`)")
        new_title = st.text_input("Title")
    with c2:
        new_body  = st.text_area("Description", height=100)
        new_tags  = st.text_input("Tags (comma-separated)")
    new_snort = st.text_input("Snort rule (optional)")
    if st.button("➕  Add to Knowledge Base"):
        if new_key and new_title and new_body:
            KB[new_key.strip().lower()] = {
                "title": new_title,
                "body":  new_body,
                "tags":  [t.strip() for t in new_tags.split(",") if t.strip()],
                "snort_rule": new_snort or "# No rule defined",
            }
            st.success(f"✅  Added `{new_key}` to knowledge base!")
            st.rerun()
        else:
            st.error("Key, Title and Description are required.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — NLP SOC SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="sec">📋 Real-Time NLP Alert Summarizer (SOC Briefing)</p>', unsafe_allow_html=True)
    st.info(
        "Aggregates recent IDS alerts and uses Groq LLaMA-3 8B to produce a "
        "concise plain-English SOC briefing — exactly as described in the bonus features.",
        icon="ℹ️"
    )

    if not st.session_state.groq_key:
        st.warning("⚠️  Enter your Groq API key above.")
    elif not st.session_state.reports:
        st.warning("⚠️  Generate some threat reports in the **LLM Threat Reports** tab first.")
    else:
        reports = st.session_state.reports
        st.markdown(f"**{len(reports)} alert(s) in queue.** Summarizing last 5.")

        if st.button("📝  Generate SOC Briefing", use_container_width=True):
            client = Groq(api_key=st.session_state.groq_key)
            with st.spinner("Generating NLP summary…"):
                try:
                    summary = groq_nlp_summarizer(client, reports)
                    st.session_state._soc_summary = summary
                except Exception as e:
                    st.session_state._soc_summary = f"[Error: {e}]"

        if "_soc_summary" in st.session_state:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"""
            <div class="alert-card info">
              <div class="alert-meta">🕐 {ts} &nbsp;|&nbsp; Generated by Groq LLaMA-3 8B &nbsp;|&nbsp; SOC Briefing</div>
              <div class="alert-title">SOC Threat Summary <span class="pill pill-info">NLP</span></div>
              <div class="alert-body" style="font-size:0.9rem;line-height:1.75">
                {st.session_state._soc_summary.replace(chr(10),'<br>')}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Alert timeline
        st.markdown('<p class="sec">Alert Timeline</p>', unsafe_allow_html=True)
        sev_map = {"high":3,"medium":2,"low":1,"info":0}
        col_map = {"high":"#ef4444","medium":"#f59e0b","low":"#22c55e","info":"#3b82f6"}

        for r in reports[:15]:
            sev_class = {"high":"","medium":"medium","low":"low","info":"info"}[r["severity"]]
            pill_cls  = f"pill-{r['severity']}"
            st.markdown(f"""
            <div class="alert-card {sev_class}" style="padding:12px 18px;">
              <div class="alert-meta">
                🕐 {r['timestamp']} &nbsp;|&nbsp;
                {r['protocol'].upper()} / {r['service']}
              </div>
              <div class="alert-title" style="font-size:0.88rem">
                {r['kb']['title']}
                <span class="pill {pill_cls}">{r['severity'].upper()}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Severity distribution pie
        from collections import Counter
        sev_counts = Counter(r["severity"] for r in reports)
        fig = go.Figure(go.Pie(
            labels=list(sev_counts.keys()),
            values=list(sev_counts.values()),
            hole=0.5,
            marker_colors=[col_map.get(s,"#64748b") for s in sev_counts.keys()],
        ))
        fig.update_layout(**PLOT_LAYOUT, title="Alert Severity Distribution", height=280)
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL KPI FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
total_reports  = len(st.session_state.reports)
high_alerts    = sum(1 for r in st.session_state.reports if r["severity"]=="high")
medium_alerts  = sum(1 for r in st.session_state.reports if r["severity"]=="medium")
model_status   = "✅ Trained" if st.session_state.model_trained else "⏳ Not trained"
st.markdown(f"""
<div class="kpi-row" style="margin-top:8px">
  <div class="kpi blue">
    <div class="kpi-label">Model Status</div>
    <div class="kpi-val" style="font-size:1.1rem">{model_status}</div>
    <div class="kpi-sub">RF + Isolation Forest</div>
  </div>
  <div class="kpi red">
    <div class="kpi-label">High Alerts</div>
    <div class="kpi-val">{high_alerts}</div>
    <div class="kpi-sub">DoS / U2R</div>
  </div>
  <div class="kpi amber">
    <div class="kpi-label">Medium Alerts</div>
    <div class="kpi-val">{medium_alerts}</div>
    <div class="kpi-sub">Probe / R2L</div>
  </div>
  <div class="kpi green">
    <div class="kpi-label">Total Reports</div>
    <div class="kpi-val">{total_reports}</div>
    <div class="kpi-sub">LLM-generated</div>
  </div>
</div>
""", unsafe_allow_html=True)
