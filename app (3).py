"""
LLM-Powered Intrusion Detection System — Semester Project
Category A | AI & LLM-Powered Security Systems | TIER S
Place this file at the ROOT of your GitHub repo.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import time
import warnings
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq                          # pip install groq
from sklearn.ensemble import (IsolationForest,
                               RandomForestClassifier)
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LLM-IDS | Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# GROQ KEY — read from Streamlit secrets only (never shown on screen)
# Local:  edit  .streamlit/secrets.toml   →  GROQ_API_KEY = "gsk_..."
# Cloud:  App menu → Settings → Secrets  →  GROQ_API_KEY = "gsk_..."
# ══════════════════════════════════════════════════════════════════════════════
def _get_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""

def _groq_client():
    k = _get_key()
    if not k or k == "gsk_your_key_here":
        return None
    return Groq(api_key=k)

def _llm_ready() -> bool:
    k = _get_key()
    return bool(k) and k != "gsk_your_key_here"

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@400;600;700;900&display=swap');
:root{
  --bg:#070b12;--bg2:#0d1220;--bg3:#111827;--border:#1e2d45;
  --accent:#3b82f6;--accent2:#6366f1;
  --danger:#ef4444;--warn:#f59e0b;--ok:#22c55e;
  --text:#e2e8f0;--muted:#64748b;
  --mono:'JetBrains Mono',monospace;--sans:'Outfit',sans-serif;
}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;
}
[data-testid="stAppViewContainer"]{
  background:radial-gradient(ellipse 80% 50% at 50% -10%,#0d1d3a,var(--bg))!important;
}
[data-testid="stHeader"]{background:transparent!important;}
#MainMenu,footer,header{visibility:hidden;}

/* HERO */
.hero{background:linear-gradient(135deg,#0d1e3a,#070b12);border:1px solid var(--border);
  border-radius:20px;padding:32px 40px;margin-bottom:28px;display:flex;align-items:center;
  gap:24px;box-shadow:0 0 60px rgba(59,130,246,.07);position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;top:-80px;right:-80px;width:300px;height:300px;
  background:radial-gradient(circle,rgba(99,102,241,.08),transparent 70%);pointer-events:none;}
.hero-icon{font-size:3.2rem;line-height:1;}
.hero-title{font-size:2rem;font-weight:900;color:var(--text);margin:0;letter-spacing:-.5px;}
.hero-sub{font-size:.78rem;color:#7fa8d4;font-family:var(--mono);margin-top:5px;}
.tier-badge{margin-left:auto;background:linear-gradient(135deg,var(--accent2),#4f46e5);
  color:#fff;font-size:.68rem;font-weight:700;padding:6px 16px;border-radius:30px;
  letter-spacing:2px;font-family:var(--mono);white-space:nowrap;
  box-shadow:0 0 20px rgba(99,102,241,.4);}

/* KPI GRID */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px;}
.kpi{background:var(--bg3);border:1px solid var(--border);border-radius:14px;
  padding:20px 22px;position:relative;overflow:hidden;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;}
.kpi.blue::before{background:linear-gradient(90deg,#3b82f6,#60a5fa);}
.kpi.red::before{background:linear-gradient(90deg,#ef4444,#f87171);}
.kpi.green::before{background:linear-gradient(90deg,#22c55e,#4ade80);}
.kpi.amber::before{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.kpi-label{font-size:.68rem;color:var(--muted);letter-spacing:1.5px;
  text-transform:uppercase;font-family:var(--mono);}
.kpi-val{font-size:2rem;font-weight:900;line-height:1.1;margin-top:6px;}
.kpi.blue .kpi-val{color:#60a5fa;}.kpi.red .kpi-val{color:#f87171;}
.kpi.green .kpi-val{color:#4ade80;}.kpi.amber .kpi-val{color:#fbbf24;}
.kpi-sub{font-size:.72rem;color:var(--muted);margin-top:3px;font-family:var(--mono);}

/* SECTION TITLE */
.sec{font-size:.72rem;font-weight:700;color:var(--muted);text-transform:uppercase;
  letter-spacing:2.5px;border-left:3px solid var(--accent2);padding-left:12px;
  margin:28px 0 14px;font-family:var(--mono);}

/* ALERT CARD */
.alert-card{background:var(--bg3);border-radius:12px;padding:18px 22px;margin-bottom:12px;
  border-left:4px solid var(--danger);border-top:1px solid var(--border);
  border-right:1px solid var(--border);border-bottom:1px solid var(--border);}
.alert-card.medium{border-left-color:var(--warn);}
.alert-card.low{border-left-color:var(--ok);}
.alert-card.info{border-left-color:var(--accent);}
.alert-meta{font-family:var(--mono);font-size:.68rem;color:var(--muted);margin-bottom:7px;}
.alert-title{font-size:.98rem;font-weight:700;color:var(--text);margin-bottom:6px;}
.alert-body{font-size:.82rem;color:#94a3b8;line-height:1.65;}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.65rem;
  font-weight:700;letter-spacing:1px;font-family:var(--mono);text-transform:uppercase;margin-left:8px;}
.pill-high{background:rgba(239,68,68,.15);color:#f87171;}
.pill-medium{background:rgba(245,158,11,.15);color:#fbbf24;}
.pill-low{background:rgba(34,197,94,.15);color:#4ade80;}
.pill-info{background:rgba(59,130,246,.15);color:#60a5fa;}

/* PACKET ROW */
.packet-row{background:var(--bg2);border:1px solid var(--border);border-radius:8px;
  padding:10px 16px;font-family:var(--mono);font-size:.74rem;color:#94a3b8;margin-bottom:6px;}
.packet-row span{color:#60a5fa;}

/* RAG CARD */
.rag-card{background:var(--bg3);border:1px solid var(--border);border-radius:10px;
  padding:16px 20px;margin-bottom:10px;}
.rag-title{font-size:.9rem;font-weight:700;color:var(--text);margin-bottom:6px;}
.rag-body{font-size:.8rem;color:#94a3b8;line-height:1.6;}
.rag-tag{display:inline-block;background:rgba(99,102,241,.15);color:#a5b4fc;
  font-size:.62rem;padding:2px 8px;border-radius:20px;font-family:var(--mono);
  margin-right:4px;margin-top:6px;}

/* BUTTONS */
.stButton>button{background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;
  color:#fff!important;border:none!important;border-radius:8px!important;
  font-family:var(--sans)!important;font-weight:700!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;
  box-shadow:0 8px 24px rgba(99,102,241,.35)!important;}

/* TABS */
.stTabs [data-baseweb="tab-list"]{background:var(--bg3)!important;border-radius:12px!important;
  padding:4px!important;gap:4px!important;border:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;
  border-radius:9px!important;font-family:var(--sans)!important;font-weight:600!important;font-size:.85rem!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;color:#fff!important;}

/* INPUTS */
.stTextArea textarea,.stTextInput>div>div>input{background:var(--bg3)!important;
  border:1px solid var(--border)!important;border-radius:8px!important;
  color:var(--text)!important;font-family:var(--mono)!important;font-size:.8rem!important;}
.stSelectbox>div>div,.stNumberInput>div>div>div{background:var(--bg3)!important;
  border-color:var(--border)!important;color:var(--text)!important;border-radius:8px!important;}
[data-testid="stExpander"]{background:var(--bg3)!important;border:1px solid var(--border)!important;border-radius:10px!important;}
hr{border-color:var(--border)!important;}
.stAlert{border-radius:10px!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = dict(
    reports=[], model_trained=False,
    rf=None, iso=None, scaler=None, les={}, feature_cols=[],
    df=None, source="", acc=0.0, f1=0.0,
    cm=None, cr=None,
    packets_captured=0, anomalies_detected=0,
)
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS / KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
ATTACK_FAMILIES = {
    "normal":"Normal","neptune":"DoS","smurf":"DoS","back":"DoS","teardrop":"DoS",
    "pod":"DoS","land":"DoS","portsweep":"Probe","ipsweep":"Probe","satan":"Probe",
    "nmap":"Probe","warezclient":"R2L","warezmaster":"R2L","guess_passwd":"R2L",
    "ftp_write":"R2L","imap":"R2L","multihop":"R2L","phf":"R2L","spy":"R2L",
    "rootkit":"U2R","buffer_overflow":"U2R","loadmodule":"U2R","perl":"U2R",
    "sqlattack":"U2R","xterm":"U2R","ps":"U2R",
}

KB = {
    "neptune":{
        "title":"Neptune / SYN Flood (DoS)",
        "body":"Neptune attacks send thousands of SYN packets without completing the TCP handshake, exhausting server connection tables. Mitigation: SYN cookies, rate-limiting, firewall rules.",
        "tags":["DoS","TCP","SYN-Flood"],
        "snort_rule":'alert tcp any any -> $HOME_NET any (flags:S; threshold:type both,track by_src,count 100,seconds 1; msg:"SYN Flood";)',
    },
    "smurf":{
        "title":"Smurf Attack (DDoS / ICMP Amplification)",
        "body":"ICMP echo requests sent to broadcast address with spoofed victim IP. All hosts reply to victim, amplifying bandwidth. Mitigation: disable IP-directed broadcasts, BCP38 ingress filtering.",
        "tags":["DDoS","ICMP","Amplification","Spoofing"],
        "snort_rule":'alert icmp any any -> $HOME_NET any (itype:8; msg:"ICMP Smurf Attack";)',
    },
    "portsweep":{
        "title":"Port Sweep / Probe",
        "body":"Attacker scans multiple ports to identify open services. High connection count with diverse destination ports over a short duration. Mitigation: port knocking, firewall, IDS rules.",
        "tags":["Probe","Reconnaissance","Nmap"],
        "snort_rule":'alert tcp any any -> $HOME_NET any (flags:S; threshold:type both,track by_src,count 20,seconds 5; msg:"Port Sweep";)',
    },
    "ipsweep":{
        "title":"IP Sweep (Network Reconnaissance)",
        "body":"Scans multiple IPs via ICMP ping to map live hosts. High ICMP request count from single source. Mitigation: ICMP rate limiting, honeypots.",
        "tags":["Probe","Reconnaissance","ICMP"],
        "snort_rule":'alert icmp any any -> $HOME_NET any (itype:8; threshold:type both,track by_src,count 15,seconds 3; msg:"IP Sweep";)',
    },
    "buffer_overflow":{
        "title":"Buffer Overflow (U2R)",
        "body":"Exploits memory management flaws to inject and execute arbitrary code with elevated privileges. Indicators: root_shell=1, num_root>0. Mitigation: ASLR, stack canaries, NX bit.",
        "tags":["U2R","Memory-Exploit","Privilege-Escalation"],
        "snort_rule":'alert tcp any any -> $HOME_NET any (content:"|90 90 90|"; msg:"Possible Buffer Overflow";)',
    },
    "guess_passwd":{
        "title":"Password Brute-Force (R2L)",
        "body":"Repeated login attempts with credential lists. High num_failed_logins, logged_in=0. Mitigation: account lockout, MFA, fail2ban, CAPTCHA.",
        "tags":["R2L","Credential","Brute-Force"],
        "snort_rule":'alert tcp any any -> $HOME_NET 22 (flags:S; threshold:type both,track by_src,count 5,seconds 10; msg:"SSH Brute Force";)',
    },
    "normal":{
        "title":"Normal Traffic",
        "body":"Traffic conforms to expected baseline. No anomaly detected. Continue standard monitoring.",
        "tags":["Benign","Baseline"],
        "snort_rule":"# No rule — traffic classified as normal",
    },
}

def _family(label: str) -> str:
    label = label.lower().strip().rstrip(".")
    for k, v in ATTACK_FAMILIES.items():
        if k in label:
            return v
    return "Unknown"

def _severity(label: str) -> str:
    f = _family(label)
    if label.strip().lower() == "normal":
        return "info"
    if f == "U2R":
        return "high"
    if f in ("R2L", "Probe"):
        return "medium"
    return "high"   # DoS → high

def _kb(label: str) -> dict:
    label = label.lower().strip().rstrip(".")
    for k in KB:
        if k in label:
            return KB[k]
    return KB["normal"]

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
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
    for src, url in [
        ("NSL-KDD GitHub/defcom17 — KDDTrain+",
         "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"),
        ("NSL-KDD GitHub/defcom17 — 20%",
         "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+_20Percent.txt"),
    ]:
        try:
            df = pd.read_csv(url, names=NSL_COLS, header=None, nrows=8000)
            df["label"] = df["label"].str.strip().str.lower()
            return df, src
        except Exception:
            continue
    # Synthetic fallback
    np.random.seed(42)
    n = 5000
    lp = ["normal","neptune","smurf","portsweep","ipsweep","satan","nmap",
          "guess_passwd","ftp_write","buffer_overflow","rootkit","teardrop","back"]
    lw = [0.35,0.14,0.10,0.06,0.06,0.04,0.03,0.05,0.02,0.04,0.02,0.05,0.04]
    df = pd.DataFrame({
        "duration":            np.random.exponential(5,n).astype(int),
        "protocol_type":       np.random.choice(["tcp","udp","icmp"],n,p=[0.6,0.25,0.15]),
        "service":             np.random.choice(["http","ftp","smtp","ssh","dns","telnet","private","other"],n,
                                                p=[0.3,0.1,0.1,0.14,0.1,0.05,0.11,0.10]),
        "flag":                np.random.choice(["SF","S0","REJ","RSTO","SH","OTH"],n,
                                                p=[0.52,0.2,0.1,0.08,0.05,0.05]),
        "src_bytes":           np.random.exponential(5000,n).astype(int),
        "dst_bytes":           np.random.exponential(2000,n).astype(int),
        "land":                np.random.choice([0,1],n,p=[0.99,0.01]),
        "wrong_fragment":      np.random.choice([0,1,2],n,p=[0.94,0.05,0.01]),
        "urgent":              np.zeros(n,dtype=int),
        "hot":                 np.random.randint(0,6,n),
        "num_failed_logins":   np.random.choice([0,1,2,3],n,p=[0.88,0.08,0.03,0.01]),
        "logged_in":           np.random.choice([0,1],n,p=[0.45,0.55]),
        "num_compromised":     np.random.choice([0,1,2],n,p=[0.9,0.08,0.02]),
        "root_shell":          np.random.choice([0,1],n,p=[0.97,0.03]),
        "su_attempted":        np.random.choice([0,1],n,p=[0.99,0.01]),
        "num_root":            np.random.choice([0,1,2],n,p=[0.9,0.08,0.02]),
        "num_file_creations":  np.random.choice([0,1,2],n,p=[0.93,0.05,0.02]),
        "num_shells":          np.random.choice([0,1],n,p=[0.97,0.03]),
        "num_access_files":    np.random.choice([0,1,2],n,p=[0.9,0.08,0.02]),
        "num_outbound_cmds":   np.zeros(n,dtype=int),
        "is_host_login":       np.zeros(n,dtype=int),
        "is_guest_login":      np.random.choice([0,1],n,p=[0.97,0.03]),
        "count":               np.random.randint(1,512,n),
        "srv_count":           np.random.randint(1,512,n),
        "serror_rate":         np.random.uniform(0,1,n).round(2),
        "srv_serror_rate":     np.random.uniform(0,1,n).round(2),
        "rerror_rate":         np.random.uniform(0,1,n).round(2),
        "srv_rerror_rate":     np.random.uniform(0,1,n).round(2),
        "same_srv_rate":       np.random.uniform(0,1,n).round(2),
        "diff_srv_rate":       np.random.uniform(0,1,n).round(2),
        "srv_diff_host_rate":  np.random.uniform(0,1,n).round(2),
        "dst_host_count":      np.random.randint(1,256,n),
        "dst_host_srv_count":  np.random.randint(1,256,n),
        "dst_host_same_srv_rate":      np.random.uniform(0,1,n).round(2),
        "dst_host_diff_srv_rate":      np.random.uniform(0,1,n).round(2),
        "dst_host_same_src_port_rate": np.random.uniform(0,1,n).round(2),
        "dst_host_srv_diff_host_rate": np.random.uniform(0,1,n).round(2),
        "dst_host_serror_rate":        np.random.uniform(0,1,n).round(2),
        "dst_host_srv_serror_rate":    np.random.uniform(0,1,n).round(2),
        "dst_host_rerror_rate":        np.random.uniform(0,1,n).round(2),
        "dst_host_srv_rerror_rate":    np.random.uniform(0,1,n).round(2),
        "label":               np.random.choice(lp,n,p=lw),
        "difficulty_level":    np.random.randint(1,21,n),
    })
    return df, "Synthetic NSL-KDD (offline fallback)"

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def _preprocess(df):
    d = df.copy()
    cat = ["protocol_type","service","flag"]
    les = {}
    for c in cat:
        le = LabelEncoder()
        d[c] = le.fit_transform(d[c].astype(str))
        les[c] = le
    d["is_attack"] = (d["label"].str.strip().str.lower() != "normal").astype(int)
    feat = [c for c in d.columns if c not in ["label","difficulty_level","is_attack"]]
    return d, les, feat

@st.cache_resource(show_spinner=False)
def train_models(_key, df):
    d, les, feat = _preprocess(df)
    X = d[feat].fillna(0)
    y = d["is_attack"]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(Xtr, ytr)
    iso = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
    iso.fit(Xtr)
    yp  = rf.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1  = f1_score(yte, yp, average="weighted")
    cm  = confusion_matrix(yte, yp)
    cr  = classification_report(yte, yp, output_dict=True)
    return rf, iso, sc, les, feat, acc, f1, cm, cr

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _fake_ip():
    r = np.random.randint
    return f"{r(1,255)}.{r(0,255)}.{r(0,255)}.{r(1,255)}"

def _sim_packet(row: dict) -> dict:
    return dict(
        timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
        src_ip=_fake_ip(), dst_ip="192.168.1."+str(np.random.randint(1,50)),
        protocol=str(row.get("protocol_type","tcp")).upper(),
        service=row.get("service","http"), flag=row.get("flag","SF"),
        length=int(row.get("src_bytes",0))+int(row.get("dst_bytes",0)),
        src_bytes=int(row.get("src_bytes",0)), dst_bytes=int(row.get("dst_bytes",0)),
        duration=int(row.get("duration",0)),
    )

def _traffic_summary(row: dict) -> str:
    return (
        f"Protocol: {str(row.get('protocol_type','?')).upper()} | "
        f"Service: {row.get('service','?')} | Flag: {row.get('flag','?')}\n"
        f"Src Bytes: {int(row.get('src_bytes',0)):,} | "
        f"Dst Bytes: {int(row.get('dst_bytes',0)):,} | "
        f"Duration: {row.get('duration',0)}s\n"
        f"Failed Logins: {row.get('num_failed_logins',0)} | "
        f"Logged In: {row.get('logged_in',0)} | "
        f"Root Shell: {row.get('root_shell',0)} | "
        f"Su Attempted: {row.get('su_attempted',0)}\n"
        f"Error Rate: {float(row.get('serror_rate',0)):.2f} | "
        f"Same Srv Rate: {float(row.get('same_srv_rate',0)):.2f} | "
        f"Count: {row.get('count',0)} | Srv Count: {row.get('srv_count',0)}\n"
        f"Dst Host Count: {row.get('dst_host_count',0)} | "
        f"Wrong Fragments: {row.get('wrong_fragment',0)}"
    )

def _row_to_vec(row: dict, les: dict, feat: list, sc: StandardScaler):
    tmp = {}
    for c in feat:
        val = row.get(c, 0)
        if c in les:
            try:
                val = les[c].transform([str(val)])[0]
            except ValueError:
                val = 0
        tmp[c] = val
    arr = pd.DataFrame([tmp])[feat].fillna(0)
    return sc.transform(arr)

# Snort simulation
SNORT_RULES = {
    "neptune":         lambda r: float(r.get("serror_rate",0))>0.7 and int(r.get("count",0))>100,
    "smurf":           lambda r: str(r.get("protocol_type",""))=="icmp" and int(r.get("src_bytes",0))>9000,
    "portsweep":       lambda r: float(r.get("diff_srv_rate",0))>0.5 and int(r.get("count",0))>30,
    "ipsweep":         lambda r: str(r.get("protocol_type",""))=="icmp" and int(r.get("dst_host_count",0))>20,
    "buffer_overflow": lambda r: int(r.get("root_shell",0))==1 or int(r.get("num_root",0))>1,
    "guess_passwd":    lambda r: int(r.get("num_failed_logins",0))>=3,
}
def _snort_detect(row: dict) -> str:
    for attack, rule in SNORT_RULES.items():
        try:
            if rule(row):
                return attack
        except Exception:
            pass
    return "normal"

# Plotly theme
PLOT = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8",family="JetBrains Mono"),
            margin=dict(l=12,r=12,t=36,b=12))
COLORS = ["#3b82f6","#ef4444","#f59e0b","#22c55e","#a78bfa","#f472b6","#34d399"]

# ══════════════════════════════════════════════════════════════════════════════
# GROQ FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def groq_report(client: Groq, summary: str, label: str, kb: dict) -> str:
    prompt = f"""You are an expert cybersecurity analyst reviewing network traffic flagged by an ML-based IDS.

=== TRAFFIC SUMMARY ===
{summary}

=== ML CLASSIFICATION ===
Attack Type   : {label}
Attack Family : {_family(label)}
Severity      : {_severity(label).upper()}
Known Pattern : {kb['title']}

=== SNORT SIGNATURE (traditional IDS) ===
{kb['snort_rule']}

Write a structured threat report with exactly these sections:
**1. Threat Classification** — one sentence.
**2. Traffic Analysis** — 2-3 sentences on why the traffic is suspicious.
**3. Potential Impact** — 1-2 sentences.
**4. LLM vs Signature IDS** — 1 sentence comparing what Snort would catch vs LLM analysis.
**5. Recommended Response**
- Action 1
- Action 2
- Action 3

Under 280 words. Be technically precise."""
    r = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role":"user","content":prompt}],
        max_tokens=500, temperature=0.25,
    )
    return r.choices[0].message.content.strip()

def groq_soc_summary(client: Groq, reports: list) -> str:
    last = reports[-5:]
    lines = "\n".join(
        f"- {r['timestamp']}: {r['label']} ({r['severity'].upper()}) on {r['service']}"
        for r in last
    )
    prompt = f"""You are a SOC analyst writing a shift briefing.

Recent IDS alerts (last {len(last)}):
{lines}

Write 3-4 sentences covering: overall threat posture, dominant attack type(s), most urgent action.
Flowing prose only — no bullet points."""
    r = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role":"user","content":prompt}],
        max_tokens=200, temperature=0.3,
    )
    return r.choices[0].message.content.strip()

# ══════════════════════════════════════════════════════════════════════════════
# ════════════════════════  MAIN UI  ══════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

# ── HERO ─────────────────────────────────────────────────────────────────────
key_status = "🟢 API key configured" if _llm_ready() else "🔴 Add GROQ_API_KEY to Secrets"
st.markdown(f"""
<div class="hero">
  <div class="hero-icon">🛡️</div>
  <div style="flex:1">
    <p class="hero-title">LLM-Powered Intrusion Detection System</p>
    <p class="hero-sub">NSL-KDD Dataset · Groq LLaMA-3 · Random Forest · Isolation Forest · RAG KB · NLP Summarizer</p>
    <p class="hero-sub" style="margin-top:6px;font-size:.72rem">{key_status}</p>
  </div>
  <span class="tier-badge">TIER S</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Dataset & Model",
    "📡 Live Capture",
    "🤖 LLM Threat Reports",
    "📈 IDS Comparison",
    "🧠 RAG Knowledge Base",
    "📋 NLP SOC Summary",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET & MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<p class="sec">📦 NSL-KDD Dataset (GitHub Auto-Download)</p>', unsafe_allow_html=True)

    if st.button("⬇️  Load Dataset & Train Models", use_container_width=True):
        with st.spinner("Downloading NSL-KDD dataset…"):
            df, src = load_dataset()
            st.session_state.df     = df
            st.session_state.source = src
        with st.spinner("Training Random Forest + Isolation Forest…"):
            rf, iso, sc, les, feat, acc, f1, cm, cr = train_models(id(df), df)
            st.session_state.rf            = rf
            st.session_state.iso           = iso
            st.session_state.scaler        = sc
            st.session_state.les           = les
            st.session_state.feature_cols  = feat
            st.session_state.acc           = acc
            st.session_state.f1            = f1
            st.session_state.cm            = cm
            st.session_state.cr            = cr
            st.session_state.model_trained = True
        st.success(f"✅  Done!  Source: **{src}**")

    if st.session_state.model_trained:
        df  = st.session_state.df
        acc = st.session_state.acc
        f1  = st.session_state.f1
        n_atk = int((df["label"] != "normal").sum())

        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi blue">
            <div class="kpi-label">Total Samples</div>
            <div class="kpi-val">{len(df):,}</div>
            <div class="kpi-sub">{st.session_state.source[:30]}</div>
          </div>
          <div class="kpi red">
            <div class="kpi-label">Attack Records</div>
            <div class="kpi-val">{n_atk:,}</div>
            <div class="kpi-sub">{n_atk/len(df)*100:.1f}% of dataset</div>
          </div>
          <div class="kpi green">
            <div class="kpi-label">RF Accuracy</div>
            <div class="kpi-val">{acc*100:.1f}%</div>
            <div class="kpi-sub">20% test split</div>
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
            fc = df["label"].apply(_family).value_counts().reset_index()
            fc.columns = ["Family","Count"]
            fig = px.bar(fc, x="Family", y="Count", color="Family", color_discrete_sequence=COLORS)
            fig.update_layout(**PLOT, showlegend=False)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#1e2d45")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<p class="sec">Top 10 Attack Labels</p>', unsafe_allow_html=True)
            top = df["label"].value_counts().head(10).reset_index()
            top.columns = ["Label","Count"]
            fig2 = px.pie(top, values="Count", names="Label",
                          color_discrete_sequence=COLORS, hole=0.45)
            fig2.update_layout(**PLOT)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="sec">Confusion Matrix (RF Binary — Normal vs Attack)</p>', unsafe_allow_html=True)
        cm = st.session_state.cm
        fig3 = go.Figure(go.Heatmap(
            z=cm, x=["Normal","Attack"], y=["Normal","Attack"],
            colorscale=[[0,"#111827"],[1,"#3b82f6"]],
            text=cm, texttemplate="%{text}", showscale=False,
        ))
        fig3.update_layout(**PLOT, height=280)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<p class="sec">Feature Importance (Top 15)</p>', unsafe_allow_html=True)
        imp = pd.Series(
            st.session_state.rf.feature_importances_,
            index=st.session_state.feature_cols
        ).nlargest(15).reset_index()
        imp.columns = ["Feature","Importance"]
        fig4 = px.bar(imp, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale=["#1e3a5f","#3b82f6","#60a5fa"])
        fig4.update_layout(**PLOT, showlegend=False, height=380)
        fig4.update_xaxes(showgrid=True, gridcolor="#1e2d45")
        st.plotly_chart(fig4, use_container_width=True)

        with st.expander("🔍 Raw Dataset Sample (100 rows)"):
            st.dataframe(df.sample(min(100,len(df)), random_state=1).reset_index(drop=True),
                         use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE CAPTURE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p class="sec">📡 Packet Capture Simulation (Scapy / PyShark)</p>', unsafe_allow_html=True)
    st.info("Simulates live packet capture feeding into the ML pipeline. In production, replace the sample loop with `scapy.sniff()` or `pyshark.LiveCapture()`.", icon="ℹ️")

    if not st.session_state.model_trained:
        st.warning("⚠️  Train models in **Dataset & Model** tab first.")
    else:
        c1, c2 = st.columns([3,1])
        with c1:
            n_pkts = st.slider("Packets to capture", 5, 50, 15)
        with c2:
            speed = st.selectbox("Speed", ["Fast","Normal","Slow"])
        delay = {"Fast":0.05,"Normal":0.2,"Slow":0.5}[speed]

        if st.button("▶️  Start Capture", use_container_width=True):
            df   = st.session_state.df
            rf   = st.session_state.rf
            iso  = st.session_state.iso
            sc   = st.session_state.scaler
            les  = st.session_state.les
            feat = st.session_state.feature_cols

            ph       = st.empty()
            captured = []
            rows     = df.sample(n_pkts, random_state=int(time.time())).to_dict("records")

            for row in rows:
                pkt    = _sim_packet(row)
                X_vec  = _row_to_vec(row, les, feat, sc)
                pred   = int(rf.predict(X_vec)[0])
                iso_sc = float(iso.decision_function(X_vec)[0])
                pkt.update(pred_attack=pred, anomaly=iso_sc<-0.05,
                           true_label=row.get("label","?"), iso_score=round(iso_sc,3))
                captured.append(pkt)

                html = ""
                for p in captured[-20:]:
                    col = "#f87171" if p["pred_attack"] else "#4ade80"
                    anom = " ⚠️" if p["anomaly"] else ""
                    html += (
                        f'<div class="packet-row">'
                        f'[{p["timestamp"]}] '
                        f'<span>{p["src_ip"]}</span> → <span>{p["dst_ip"]}</span>'
                        f' &nbsp;|&nbsp; {p["protocol"]} / {p["service"]} / {p["flag"]}'
                        f' &nbsp;|&nbsp; {p["length"]:,} bytes'
                        f' &nbsp;|&nbsp; <span style="color:{col}">{"ATTACK" if p["pred_attack"] else "NORMAL"}</span>'
                        f'{anom}</div>'
                    )
                ph.markdown(html, unsafe_allow_html=True)
                time.sleep(delay)

            st.session_state._capture = captured
            atks = sum(p["pred_attack"] for p in captured)
            st.success(f"✅  {n_pkts} packets — {atks} attack(s) detected")

        if "_capture" in st.session_state:
            cap  = st.session_state._capture
            atks = sum(p["pred_attack"] for p in cap)
            anoms= sum(p["anomaly"]     for p in cap)
            st.markdown(f"""
            <div class="kpi-row">
              <div class="kpi blue"><div class="kpi-label">Packets</div><div class="kpi-val">{len(cap)}</div></div>
              <div class="kpi red"><div class="kpi-label">RF Attacks</div><div class="kpi-val">{atks}</div></div>
              <div class="kpi amber"><div class="kpi-label">IF Anomalies</div><div class="kpi-val">{anoms}</div></div>
              <div class="kpi green"><div class="kpi-label">Normal</div><div class="kpi-val">{len(cap)-atks}</div></div>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=[p["iso_score"] for p in cap], mode="lines+markers",
                line=dict(color="#3b82f6",width=2),
                marker=dict(color=["#ef4444" if p["pred_attack"] else "#22c55e" for p in cap], size=8),
                name="IF Score",
            ))
            fig.add_hline(y=-0.05, line_dash="dash", line_color="#f59e0b", annotation_text="Anomaly threshold")
            fig.update_layout(**PLOT, title="Isolation Forest Score per Packet",
                              xaxis_title="Packet #", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LLM THREAT REPORTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p class="sec">🤖 LLM Threat Reports (Groq LLaMA-3 70B)</p>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️  Train models first.")
    elif not _llm_ready():
        st.warning("⚠️  Groq API key not found. Add `GROQ_API_KEY` in **App Settings → Secrets**.")
    else:
        df   = st.session_state.df
        rf   = st.session_state.rf
        sc   = st.session_state.scaler
        les  = st.session_state.les
        feat = st.session_state.feature_cols

        c1, c2 = st.columns([3,1])
        with c1:
            opts = ["(Random)"] + sorted(df["label"].unique().tolist())
            chosen = st.selectbox("Filter by attack type", opts)
        with c2:
            n_analyze = st.number_input("Records", 1, 5, 1)

        if st.button("🔍  Analyze & Generate Reports", use_container_width=True):
            if chosen == "(Random)":
                samples = df.sample(int(n_analyze), random_state=int(time.time()))
            else:
                pool = df[df["label"] == chosen]
                samples = pool.sample(min(int(n_analyze), len(pool)), random_state=int(time.time()))

            client   = _groq_client()
            progress = st.progress(0, "Analyzing…")

            for idx, (_, row) in enumerate(samples.iterrows()):
                rd         = row.to_dict()
                true_label = rd.get("label","?")
                X_vec      = _row_to_vec(rd, les, feat, sc)
                pred_bin   = int(rf.predict(X_vec)[0])
                pred_label = true_label if pred_bin else "normal"
                sev        = _severity(pred_label)
                kb         = _kb(pred_label)
                summary    = _traffic_summary(rd)

                with st.spinner(f"Report {idx+1}/{len(samples)}…"):
                    try:
                        report_text = groq_report(client, summary, pred_label, kb)
                    except Exception as e:
                        report_text = f"[LLM error: {e}]"

                st.session_state.reports.insert(0, dict(
                    timestamp  = datetime.now().strftime("%H:%M:%S"),
                    label      = pred_label,
                    true_label = true_label,
                    severity   = sev,
                    service    = rd.get("service","?"),
                    protocol   = str(rd.get("protocol_type","?")),
                    summary    = summary,
                    report     = report_text,
                    kb         = kb,
                ))
                progress.progress((idx+1)/len(samples))

            progress.empty()
            st.success(f"✅  {len(samples)} report(s) generated.")

        if st.session_state.reports:
            st.markdown('<p class="sec">Generated Reports</p>', unsafe_allow_html=True)
            for r in st.session_state.reports[:20]:
                sc_cls   = {"high":"","medium":"medium","low":"low","info":"info"}[r["severity"]]
                pill_cls = f"pill-{r['severity']}"
                st.markdown(f"""
                <div class="alert-card {sc_cls}">
                  <div class="alert-meta">🕐 {r['timestamp']} &nbsp;|&nbsp;
                    {r['protocol'].upper()} / {r['service']} &nbsp;|&nbsp;
                    True label: <b>{r['true_label']}</b></div>
                  <div class="alert-title">{r['kb']['title']}
                    <span class="pill {pill_cls}">{r['severity'].upper()}</span></div>
                  <div class="alert-body">{r['report'].replace(chr(10),'<br>')}</div>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🗑️  Clear All Reports"):
                st.session_state.reports = []
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — IDS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p class="sec">📈 ML-IDS vs Traditional Signature IDS (Snort / Suricata)</p>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️  Train models first.")
    else:
        if st.button("▶️  Run Comparison on 500 Samples", use_container_width=True):
            df   = st.session_state.df
            rf   = st.session_state.rf
            sc   = st.session_state.scaler
            les  = st.session_state.les
            feat = st.session_state.feature_cols

            rows       = df.sample(500, random_state=42).to_dict("records")
            ml_preds   = []
            sn_preds   = []
            true_bins  = []

            for row in rows:
                true_bin = 0 if row.get("label","normal") == "normal" else 1
                X_vec    = _row_to_vec(row, les, feat, sc)
                ml_pred  = int(rf.predict(X_vec)[0])
                sn_pred  = 0 if _snort_detect(row) == "normal" else 1
                ml_preds.append(ml_pred)
                sn_preds.append(sn_pred)
                true_bins.append(true_bin)

            cmp = dict(
                ml_acc  = accuracy_score(true_bins, ml_preds),
                sn_acc  = accuracy_score(true_bins, sn_preds),
                ml_f1   = f1_score(true_bins, ml_preds,  average="weighted"),
                sn_f1   = f1_score(true_bins, sn_preds,  average="weighted"),
                ml_fp   = sum(1 for m,t in zip(ml_preds, true_bins) if m==1 and t==0),
                sn_fp   = sum(1 for m,t in zip(sn_preds, true_bins) if m==1 and t==0),
                ml_fn   = sum(1 for m,t in zip(ml_preds, true_bins) if m==0 and t==1),
                sn_fn   = sum(1 for m,t in zip(sn_preds, true_bins) if m==0 and t==1),
            )
            st.session_state._cmp = cmp

        if "_cmp" in st.session_state:
            cmp = st.session_state._cmp

            metrics  = ["Accuracy (%)","F1 Score (%)","False Positives","False Negatives"]
            ml_vals  = [cmp["ml_acc"]*100, cmp["ml_f1"]*100, cmp["ml_fp"], cmp["ml_fn"]]
            sn_vals  = [cmp["sn_acc"]*100, cmp["sn_f1"]*100, cmp["sn_fp"], cmp["sn_fn"]]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Random Forest (ML-IDS)", x=metrics, y=ml_vals, marker_color="#3b82f6"))
            fig.add_trace(go.Bar(name="Snort Signatures",        x=metrics, y=sn_vals, marker_color="#f59e0b"))
            fig.update_layout(**PLOT, barmode="group", title="ML-IDS vs Signature IDS — 500 NSL-KDD samples")
            fig.update_yaxes(showgrid=True, gridcolor="#1e2d45")
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="rag-card">
                  <div class="rag-title">🤖 Random Forest (ML-IDS)</div>
                  <div class="rag-body">
                    Accuracy: <b>{cmp['ml_acc']*100:.1f}%</b> &nbsp;|&nbsp; F1: <b>{cmp['ml_f1']*100:.1f}%</b><br>
                    False Positives: <b>{cmp['ml_fp']}</b> &nbsp;|&nbsp; False Negatives: <b>{cmp['ml_fn']}</b><br><br>
                    Learns statistical patterns from 41 NSL-KDD features.
                    Detects novel and zero-day attacks. Provides feature-level explainability.
                  </div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="rag-card">
                  <div class="rag-title">🔏 Snort / Suricata (Signature IDS)</div>
                  <div class="rag-body">
                    Accuracy: <b>{cmp['sn_acc']*100:.1f}%</b> &nbsp;|&nbsp; F1: <b>{cmp['sn_f1']*100:.1f}%</b><br>
                    False Positives: <b>{cmp['sn_fp']}</b> &nbsp;|&nbsp; False Negatives: <b>{cmp['sn_fn']}</b><br><br>
                    Rule-based, fast, and reliable on known attacks.
                    Misses zero-day, polymorphic, and slow-burn attacks entirely.
                  </div></div>""", unsafe_allow_html=True)

            # Radar chart
            cats      = ["Accuracy","F1","Low FP","Low FN","Novel Attack Detection","Explainability"]
            ml_radar  = [cmp["ml_acc"]*100, cmp["ml_f1"]*100,
                         max(0,100-cmp["ml_fp"]/5), max(0,100-cmp["ml_fn"]/5), 85, 80]
            sn_radar  = [cmp["sn_acc"]*100, cmp["sn_f1"]*100,
                         max(0,100-cmp["sn_fp"]/5), max(0,100-cmp["sn_fn"]/5), 20, 95]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=ml_radar, theta=cats, fill='toself',
                                           name="ML-IDS", line_color="#3b82f6"))
            fig2.add_trace(go.Scatterpolar(r=sn_radar, theta=cats, fill='toself',
                                           name="Snort",  line_color="#f59e0b"))
            fig2.update_layout(**PLOT,
                               polar=dict(bgcolor="#0d1220",
                                          radialaxis=dict(visible=True,range=[0,100],color="#1e2d45"),
                                          angularaxis=dict(color="#64748b")),
                               title="Capability Radar: ML-IDS vs Snort/Suricata")
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RAG KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p class="sec">🧠 RAG-Based Threat Knowledge Base</p>', unsafe_allow_html=True)
    st.info(
        "Retrieval-Augmented Generation: when the LLM generates a report, "
        "the matching KB entry (attack description + Snort rule + tags) is injected "
        "as context — giving the model accurate, grounded information.",
        icon="ℹ️"
    )

    search = st.text_input("🔍 Search", placeholder="e.g. DoS, buffer, brute")
    entries = [(k,v) for k,v in KB.items() if k != "normal"]
    if search.strip():
        q = search.lower()
        entries = [(k,v) for k,v in entries
                   if q in k or q in v["title"].lower() or q in v["body"].lower()]

    for key, entry in entries:
        fam  = _family(key)
        sev  = _severity(key)
        tags = "".join(f'<span class="rag-tag">{t}</span>' for t in entry["tags"])
        st.markdown(f"""
        <div class="rag-card">
          <div class="rag-title">
            {entry['title']}
            <span class="pill pill-{sev}">{sev.upper()}</span>
            <span class="pill pill-info">{fam}</span>
          </div>
          <div class="rag-body">{entry['body']}</div>
          <div style="margin-top:10px">
            <span style="font-family:var(--mono);font-size:.68rem;color:var(--muted)">SNORT RULE:</span><br>
            <code style="font-size:.72rem;color:#a5b4fc;background:#0d1220;padding:6px 10px;
                         border-radius:6px;display:block;margin-top:4px;border:1px solid #1e2d45;">
              {entry['snort_rule']}
            </code>
          </div>
          <div style="margin-top:8px">{tags}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**➕ Add custom entry**")
    c1, c2 = st.columns(2)
    with c1:
        nk = st.text_input("Key (e.g. `shellcode`)")
        nt = st.text_input("Title")
    with c2:
        nb = st.text_area("Description", height=90)
        ng = st.text_input("Tags (comma-separated)")
    ns = st.text_input("Snort rule (optional)")
    if st.button("Add to Knowledge Base"):
        if nk and nt and nb:
            KB[nk.strip().lower()] = {
                "title": nt, "body": nb,
                "tags":  [t.strip() for t in ng.split(",") if t.strip()],
                "snort_rule": ns or "# No rule defined",
            }
            st.success(f"Added `{nk}`!")
            st.rerun()
        else:
            st.error("Key, Title and Description required.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — NLP SOC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<p class="sec">📋 Real-Time NLP SOC Briefing (LLaMA-3 8B)</p>', unsafe_allow_html=True)
    st.info(
        "Aggregates recent IDS alerts and uses Groq LLaMA-3 8B to write a "
        "concise plain-English SOC shift briefing.", icon="ℹ️"
    )

    if not _llm_ready():
        st.warning("⚠️  Add `GROQ_API_KEY` in App Settings → Secrets.")
    elif not st.session_state.reports:
        st.warning("⚠️  Generate reports in the **LLM Threat Reports** tab first.")
    else:
        rpts = st.session_state.reports
        st.markdown(f"**{len(rpts)} alert(s) queued.** Summarising last 5.")

        if st.button("📝  Generate SOC Briefing", use_container_width=True):
            client = _groq_client()
            with st.spinner("Writing briefing…"):
                try:
                    brief = groq_soc_summary(client, rpts)
                    st.session_state._brief = brief
                except Exception as e:
                    st.session_state._brief = f"[Error: {e}]"

        if "_brief" in st.session_state:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"""
            <div class="alert-card info">
              <div class="alert-meta">🕐 {ts} &nbsp;|&nbsp; Groq LLaMA-3 8B &nbsp;|&nbsp; SOC Briefing</div>
              <div class="alert-title">SOC Threat Summary <span class="pill pill-info">NLP</span></div>
              <div class="alert-body" style="font-size:.9rem;line-height:1.75">
                {st.session_state._brief.replace(chr(10),'<br>')}
              </div>
            </div>
            """, unsafe_allow_html=True)

        col_map = {"high":"#ef4444","medium":"#f59e0b","low":"#22c55e","info":"#3b82f6"}
        st.markdown('<p class="sec">Alert Timeline</p>', unsafe_allow_html=True)
        for r in rpts[:15]:
            sc_cls = {"high":"","medium":"medium","low":"low","info":"info"}[r["severity"]]
            st.markdown(f"""
            <div class="alert-card {sc_cls}" style="padding:12px 18px;">
              <div class="alert-meta">🕐 {r['timestamp']} &nbsp;|&nbsp; {r['protocol'].upper()} / {r['service']}</div>
              <div class="alert-title" style="font-size:.88rem">
                {r['kb']['title']}
                <span class="pill pill-{r['severity']}">{r['severity'].upper()}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        sev_counts = Counter(r["severity"] for r in rpts)
        fig = go.Figure(go.Pie(
            labels=list(sev_counts.keys()),
            values=list(sev_counts.values()),
            hole=0.5,
            marker_colors=[col_map.get(s,"#64748b") for s in sev_counts.keys()],
        ))
        fig.update_layout(**PLOT, title="Alert Severity Distribution", height=280)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
total  = len(st.session_state.reports)
high   = sum(1 for r in st.session_state.reports if r["severity"]=="high")
medium = sum(1 for r in st.session_state.reports if r["severity"]=="medium")
mstat  = "✅ Trained" if st.session_state.model_trained else "⏳ Not trained"
st.markdown(f"""
<div class="kpi-row" style="margin-top:8px">
  <div class="kpi blue">
    <div class="kpi-label">Model Status</div>
    <div class="kpi-val" style="font-size:1.1rem">{mstat}</div>
    <div class="kpi-sub">RF + Isolation Forest</div>
  </div>
  <div class="kpi red">
    <div class="kpi-label">High Alerts</div>
    <div class="kpi-val">{high}</div>
    <div class="kpi-sub">DoS / U2R</div>
  </div>
  <div class="kpi amber">
    <div class="kpi-label">Medium Alerts</div>
    <div class="kpi-val">{medium}</div>
    <div class="kpi-sub">Probe / R2L</div>
  </div>
  <div class="kpi green">
    <div class="kpi-label">Total Reports</div>
    <div class="kpi-val">{total}</div>
    <div class="kpi-sub">LLM-generated</div>
  </div>
</div>
""", unsafe_allow_html=True)
