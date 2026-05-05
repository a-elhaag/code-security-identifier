import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Code Security Identifier", layout="wide")

WEIGHTS_DIR = Path("weights")
GCB_CHECKPOINT = WEIGHTS_DIR / "graphCodeBert.pt"
DUAL_CHECKPOINT = WEIGHTS_DIR / "dualEncoder.pt"
GCB_MODEL_NAME = "microsoft/graphcodebert-base"
VULB_MODEL_NAME = "claudios/VulBERTa-MLP-D2A"
MAX_LENGTH = 512
NUM_CLASSES = 8

CWE_LABELS = [
    "CWE-077", "CWE-601", "CWE-022", "CWE-094",
    "CWE-089", "CWE-352", "CWE-079", "unknown",
]

CWE_EXPLANATIONS = {
    "CWE-077": {"name": "Command Injection",         "description": "The code may allow untrusted input to influence an OS command."},
    "CWE-601": {"name": "Open Redirect",             "description": "The code may redirect users to a URL controlled by an attacker."},
    "CWE-022": {"name": "Path Traversal",            "description": "The code may allow input such as ../ to access files outside the intended directory."},
    "CWE-094": {"name": "Code Injection",            "description": "The code may execute dynamically constructed code from unsafe input."},
    "CWE-089": {"name": "SQL Injection",             "description": "The code may build SQL queries using untrusted input without safe parameterization."},
    "CWE-352": {"name": "Cross-Site Request Forgery","description": "The code may allow a forged request to perform actions as an authenticated user."},
    "CWE-079": {"name": "Cross-Site Scripting",      "description": "The code may place untrusted input into a web page without proper escaping."},
    "unknown":  {"name": "No Vulnerability Detected","description": "The model did not map this snippet to any of the trained CWE classes."},
}

CONTRIBUTORS = [
    ("Anas Ahmed",       "192200122"),
    ("Youstina Adel",    "192200148"),
    ("Sohaila Tamer",    "192200251"),
    ("Farida Hassan",    "192200221"),
    ("Jomana Mekheimar", "192200297"),
    ("Menna Reda",       "192200325"),
    ("Hend Elhout",      "192300146"),
    ("Hesham Elshimy",   "192200154"),
]

# ======================
# MODEL CLASSES
# ======================

class GraphCodeBERTLoRACWEModel(nn.Module):
    def __init__(self, use_lora: bool = True, lora_r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1):
        super().__init__()
        encoder = AutoModel.from_pretrained(GCB_MODEL_NAME)
        if use_lora:
            if get_peft_model is None:
                raise RuntimeError("peft is not installed.")
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["query", "key", "value"], bias="none",
            )
            encoder = get_peft_model(encoder, lora_cfg)
        self.encoder = encoder
        hidden = 768
        self.cwe_head = nn.ModuleDict({
            "norm": nn.LayerNorm(hidden),
            "fc1":  nn.Linear(hidden, hidden // 2),
            "fc2":  nn.Linear(hidden // 2, NUM_CLASSES),
        })
        self.act = nn.GELU()

    @staticmethod
    def _mean_pool(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        x = self.cwe_head["norm"](pooled)
        x = self.act(self.cwe_head["fc1"](x))
        return {"logits": self.cwe_head["fc2"](x)}


class _FusionProjection(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
        )

    def forward(self, x):
        return self.projection(x)


class VulBERTaFusionModel(nn.Module):
    def __init__(self, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        gcb = AutoModel.from_pretrained(GCB_MODEL_NAME)
        if get_peft_model is None:
            raise RuntimeError("peft is not installed.")
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"], bias="none",
        )
        self.graphcodebert = get_peft_model(gcb, lora_cfg)
        self.vulberta = RobertaModel.from_pretrained(VULB_MODEL_NAME)
        for p in self.vulberta.parameters():
            p.requires_grad = False
        self.fusion = _FusionProjection()
        self.classifier = nn.Sequential(
            nn.Linear(768, 384), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(384, NUM_CLASSES),
        )

    def forward(self, gcb_input_ids, gcb_attention_mask, vulb_input_ids, vulb_attention_mask):
        gcb_cls  = self.graphcodebert(input_ids=gcb_input_ids,  attention_mask=gcb_attention_mask).last_hidden_state[:, 0]
        vulb_cls = self.vulberta(     input_ids=vulb_input_ids, attention_mask=vulb_attention_mask).last_hidden_state[:, 0]
        fused = self.fusion(torch.cat([gcb_cls, vulb_cls], dim=-1))
        return {"logits": self.classifier(fused)}


# ======================
# DEVICE & LOADERS
# ======================

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource(show_spinner="Loading GraphCodeBERT model...")
def load_gcb_assets():
    device = _get_device()
    if not GCB_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {GCB_CHECKPOINT}")
    ckpt = torch.load(GCB_CHECKPOINT, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    cfg   = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    max_len = int(cfg.get("max_length", MAX_LENGTH))
    lora_r  = int(cfg.get("lora_r", 8))

    model = GraphCodeBERTLoRACWEModel(use_lora=True, lora_r=lora_r)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        st.warning(f"GCB load: missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys[:5]}")
    model.to(device).eval()
    tokenizer = RobertaTokenizer.from_pretrained(GCB_MODEL_NAME)
    return tokenizer, model, device, max_len


@st.cache_resource(show_spinner="Loading Dual Encoder model...")
def load_dual_assets():
    device = _get_device()
    if not DUAL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {DUAL_CHECKPOINT}")
    ckpt  = torch.load(DUAL_CHECKPOINT, map_location=device, weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    key_map = {
        "fusion.proj.weight":  "fusion.projection.0.weight",
        "fusion.proj.bias":    "fusion.projection.0.bias",
        "fusion.norm.weight":  "fusion.projection.1.weight",
        "fusion.norm.bias":    "fusion.projection.1.bias",
    }
    state = {key_map.get(k, k): v for k, v in state.items()}

    model = VulBERTaFusionModel()
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        st.warning(f"Dual load: missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys[:5]}")
    model.to(device).eval()
    gcb_tok  = RobertaTokenizer.from_pretrained(GCB_MODEL_NAME)
    vulb_tok = AutoTokenizer.from_pretrained(VULB_MODEL_NAME)
    return gcb_tok, vulb_tok, model, device


# ======================
# INFERENCE
# ======================

def predict_gcb(code: str):
    tokenizer, model, device, max_len = load_gcb_assets()
    enc = tokenizer(code, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])["logits"]
        probs  = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    idx = probs.index(max(probs))
    return CWE_LABELS[idx], {CWE_LABELS[i]: round(p * 100, 1) for i, p in enumerate(probs)}


def predict_dual(code: str):
    gcb_tok, vulb_tok, model, device = load_dual_assets()
    gcb_enc  = gcb_tok( code, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(device)
    vulb_enc = vulb_tok(code, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(
            gcb_enc["input_ids"],  gcb_enc["attention_mask"],
            vulb_enc["input_ids"], vulb_enc["attention_mask"],
        )["logits"]
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    idx = probs.index(max(probs))
    return CWE_LABELS[idx], {CWE_LABELS[i]: round(p * 100, 1) for i, p in enumerate(probs)}


# ======================
# FUNCTION SPLITTING & MULTI-FUNCTION PIPELINE
# ======================

def split_functions(code: str) -> list[str]:
    parts = re.split(r"^(?=(?:async\s+)?def\s)", code, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]


def _aggregate_probs(all_probs: list[dict]) -> dict:
    avg = {lbl: 0.0 for lbl in CWE_LABELS}
    for p in all_probs:
        for lbl, val in p.items():
            avg[lbl] += val
    n = len(all_probs)
    return {lbl: round(v / n, 1) for lbl, v in avg.items()}


def analyze_file_gcb(code: str):
    funcs = split_functions(code) or [code]
    per_func = []
    for fn in funcs:
        pred, probs = predict_gcb(fn)
        per_func.append({"code": fn, "prediction": pred, "probs": probs})
    avg_probs = _aggregate_probs([f["probs"] for f in per_func])
    overall = max(avg_probs, key=avg_probs.get)
    return per_func, avg_probs, overall


def analyze_file_dual(code: str):
    funcs = split_functions(code) or [code]
    per_func = []
    for fn in funcs:
        pred, probs = predict_dual(fn)
        per_func.append({"code": fn, "prediction": pred, "probs": probs})
    avg_probs = _aggregate_probs([f["probs"] for f in per_func])
    overall = max(avg_probs, key=avg_probs.get)
    return per_func, avg_probs, overall


# ======================
# STYLING
# ======================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap');

        :root {
            --dark-teal: #102a2e;
            --teal: #49c5b1;
            --light-green: #d4ec8e;
            --canvas: #FBFBFA;
            --surface: #FFFFFF;
            --border: #EAEAEA;
            --text-primary: #111111;
            --text-secondary: #787774;
            --radius: 12px;
            --font-sans: 'SF Pro Display', 'Helvetica Neue', 'Switzer', sans-serif;
            --font-serif: 'Newsreader', 'Instrument Serif', 'Playfair Display', serif;
            --font-mono: 'SF Mono', 'JetBrains Mono', 'Geist Mono', monospace;
        }

        html, body, [class*="css"] {
            font-family: var(--font-sans) !important;
            color: var(--text-primary);
        }

        .stApp {
            background: var(--canvas);
        }

        /* ── Header ── */
        .app-header {
            border-bottom: 1px solid var(--border);
            padding: 2.5rem 0 2rem;
            margin-bottom: 2.5rem;
        }

        .app-header h1 {
            font-family: var(--font-serif);
            font-size: 2.4rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            line-height: 1.1;
            color: var(--dark-teal);
            margin: 0 0 0.4rem;
        }

        .app-header p {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin: 0;
            letter-spacing: 0.01em;
        }

        /* ── Textarea ── */
        .stTextArea textarea {
            border-radius: var(--radius) !important;
            padding: 1rem 1.25rem !important;
            border: 1px solid var(--border) !important;
            background: var(--surface) !important;
            font-family: var(--font-mono) !important;
            font-size: 0.875rem !important;
            color: var(--text-primary) !important;
            box-shadow: none !important;
            transition: border-color 160ms ease !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 3px rgba(73,197,177,0.08) !important;
        }

        /* ── Primary button ── */
        .stButton > button {
            width: 100%;
            border-radius: 6px !important;
            padding: 0.6rem 1.5rem !important;
            background: var(--dark-teal) !important;
            color: #ffffff !important;
            border: none !important;
            font-family: var(--font-sans) !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.01em !important;
            box-shadow: none !important;
            transition: background 160ms ease, transform 100ms ease !important;
        }

        .stButton > button:hover {
            background: #1d3d42 !important;
        }

        .stButton > button:active {
            transform: scale(0.98) !important;
        }

        /* ── Result card ── */
        .result-container {
            background: var(--surface);
            padding: 1.75rem 2rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            margin-top: 1.5rem;
        }

        .result-container h3 {
            font-family: var(--font-serif);
            font-size: 1.25rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--dark-teal);
            margin: 0 0 1rem;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: transparent !important;
            border-bottom: 1px solid var(--border) !important;
            gap: 0 !important;
            padding: 0 !important;
        }

        button[data-baseweb="tab"] {
            font-family: var(--font-sans) !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
            background: transparent !important;
            border-radius: 0 !important;
            border-bottom: 2px solid transparent !important;
            padding: 0.6rem 1.25rem !important;
            transition: color 160ms ease !important;
        }

        button[data-baseweb="tab"]:hover { color: var(--text-primary) !important; }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--dark-teal) !important;
            border-bottom-color: var(--teal) !important;
            font-weight: 600 !important;
        }

        /* ── Alerts ── */
        div[data-testid="stAlert"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border) !important;
            background: var(--surface) !important;
            font-size: 0.875rem !important;
        }

        /* ── Expander ── */
        div[data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            background: var(--surface) !important;
        }

        /* ── File uploader ── */
        div[data-testid="stFileUploader"] {
            border-radius: var(--radius) !important;
        }

        /* ── Code blocks ── */
        code, pre { font-family: var(--font-mono) !important; font-size: 0.82rem !important; }

        /* ── Divider ── */
        hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 2rem 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================
# HEADER
# ======================
st.markdown(
    """
    <div class="app-header">
        <h1>Code Security Identifier</h1>
        <p>Static vulnerability detection via GraphCodeBERT &middot; 8 CWE classes</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ======================
# SHARED RESULT RENDERER
# ======================
def _render_result(prediction: str, probs: dict):
    explanation = CWE_EXPLANATIONS[prediction]
    is_safe = prediction == "unknown"

    if is_safe:
        badge_bg, badge_text = "#EDF3EC", "#346538"
    else:
        badge_bg, badge_text = "#FDEBEC", "#9F2F2D"

    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    bar_color = "#49c5b1" if is_safe else "#e05c5c"
    top3_html = "".join(
        f'<div style="display:flex;align-items:center;gap:12px;margin:6px 0;">'
        f'<span style="width:80px;font-size:0.78rem;font-weight:600;color:var(--text-secondary);font-family:var(--font-mono);letter-spacing:0.02em;">{lbl}</span>'
        f'<div style="flex:1;background:#F3F3F2;border-radius:3px;height:5px;">'
        f'<div style="width:{pct}%;background:{bar_color};border-radius:3px;height:5px;transition:width 400ms ease;"></div>'
        f'</div>'
        f'<span style="font-size:0.78rem;color:var(--text-secondary);width:38px;text-align:right;">{pct}%</span>'
        f'</div>'
        for lbl, pct in top3
    )

    st.markdown(
        f"""
        <div class="result-container">
            <h3>Analysis Result</h3>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.75rem;">
                <span style="background:{badge_bg};color:{badge_text};padding:3px 10px;border-radius:9999px;font-size:0.72rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;font-family:var(--font-sans);">{prediction}</span>
                <span style="font-weight:600;font-size:1.05rem;color:var(--text-primary);">{explanation['name']}</span>
            </div>
            <p style="color:var(--text-secondary);line-height:1.65;font-size:0.9rem;margin:0 0 1.25rem;">{explanation['description']}</p>
            <div style="border-top:1px solid var(--border);padding-top:1rem;">
                <p style="font-size:0.72rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:var(--text-secondary);margin:0 0 0.6rem;">Top predictions</p>
                {top3_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================
# HELPER: per-function breakdown renderer
# ======================

def _render_per_function(per_func: list[dict]):
    st.markdown(
        '<p style="font-size:0.72rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:var(--text-secondary);margin:1.5rem 0 0.5rem;">Per-function breakdown</p>',
        unsafe_allow_html=True,
    )
    for i, item in enumerate(per_func, 1):
        pred = item["prediction"]
        expl = CWE_EXPLANATIONS[pred]
        is_safe = pred == "unknown"
        badge_bg  = "#EDF3EC" if is_safe else "#FDEBEC"
        badge_txt = "#346538" if is_safe else "#9F2F2D"
        with st.expander(f"fn {i}  ·  {pred}  ·  {expl['name']}"):
            st.markdown(
                f'<span style="background:{badge_bg};color:{badge_txt};padding:2px 9px;border-radius:9999px;font-size:0.7rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;">{pred}</span>'
                f'&nbsp;&nbsp;<span style="font-size:0.85rem;color:var(--text-secondary);">{expl["description"]}</span>',
                unsafe_allow_html=True,
            )
            st.code(item["code"], language="python")


# ======================
# TABS
# ======================
tab_gcb, tab_dual = st.tabs(["GraphCodeBERT", "Dual Encoder"])

# ── GraphCodeBERT tab ──────────────────────────────────────────────────────
with tab_gcb:
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown(
            '<p style="font-size:0.72rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:var(--text-secondary);margin:0 0 0.4rem;">Model</p>'
            '<p style="font-size:1rem;font-weight:600;color:var(--dark-teal);margin:0 0 0.75rem;">GraphCodeBERT</p>'
            '<p style="font-size:0.82rem;color:var(--text-secondary);line-height:1.55;margin:0 0 1.25rem;">Single encoder fine-tuned with LoRA on 8 CWE classes. Upload a file to get per-function scores averaged into one result.</p>',
            unsafe_allow_html=True,
        )
        gcb_btn = st.button("Run Security Scan", key="btn_gcb", type="primary")

    with col1:
        gcb_sub_paste, gcb_sub_upload = st.tabs(["Paste Code", "Upload File"])
        with gcb_sub_paste:
            gcb_paste = st.text_area(
                "gcb_paste", height=300,
                placeholder="Paste your code snippet here...",
                label_visibility="collapsed", key="gcb_paste",
            )
        with gcb_sub_upload:
            gcb_file = st.file_uploader("Upload .py file", type=["py"], key="gcb_file", label_visibility="collapsed")

    if gcb_btn:
        code = ""
        if gcb_file is not None:
            code = gcb_file.read().decode("utf-8")
        elif gcb_paste.strip():
            code = gcb_paste
        if not code.strip():
            st.warning("Please paste code or upload a file.")
        else:
            try:
                funcs = split_functions(code)
                is_multifunction = len(funcs) > 1
                with st.spinner(f"Analyzing {len(funcs)} function(s)..." if is_multifunction else "Analyzing..."):
                    if is_multifunction:
                        per_func, avg_probs, overall = analyze_file_gcb(code)
                    else:
                        overall, avg_probs = predict_gcb(code)
                        per_func = None
                st.markdown(f"**{len(funcs)} function(s) detected — overall result (averaged):**" if is_multifunction else "**Result:**")
                _render_result(overall, avg_probs)
                if per_func:
                    _render_per_function(per_func)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ── Dual Encoder tab ───────────────────────────────────────────────────────
with tab_dual:
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown(
            '<p style="font-size:0.72rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:var(--text-secondary);margin:0 0 0.4rem;">Model</p>'
            '<p style="font-size:1rem;font-weight:600;color:var(--dark-teal);margin:0 0 0.75rem;">Dual Encoder</p>'
            '<p style="font-size:0.82rem;color:var(--text-secondary);line-height:1.55;margin:0 0 1.25rem;">Fusion of GraphCodeBERT (structural) and VulBERTa (security-specific). Upload a file to get per-function scores averaged into one result.</p>',
            unsafe_allow_html=True,
        )
        dual_btn = st.button("Run Security Scan", key="btn_dual", type="primary")

    with col1:
        dual_sub_paste, dual_sub_upload = st.tabs(["Paste Code", "Upload File"])
        with dual_sub_paste:
            dual_paste = st.text_area(
                "dual_paste", height=300,
                placeholder="Paste your code snippet here...",
                label_visibility="collapsed", key="dual_paste",
            )
        with dual_sub_upload:
            dual_file = st.file_uploader("Upload .py file", type=["py"], key="dual_file", label_visibility="collapsed")

    if dual_btn:
        code = ""
        if dual_file is not None:
            code = dual_file.read().decode("utf-8")
        elif dual_paste.strip():
            code = dual_paste
        if not code.strip():
            st.warning("Please paste code or upload a file.")
        else:
            try:
                funcs = split_functions(code)
                is_multifunction = len(funcs) > 1
                with st.spinner(f"Analyzing {len(funcs)} function(s)..." if is_multifunction else "Analyzing..."):
                    if is_multifunction:
                        per_func, avg_probs, overall = analyze_file_dual(code)
                    else:
                        overall, avg_probs = predict_dual(code)
                        per_func = None
                st.markdown(f"**{len(funcs)} function(s) detected — overall result (averaged):**" if is_multifunction else "**Result:**")
                _render_result(overall, avg_probs)
                if per_func:
                    _render_per_function(per_func)
            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ======================
# FOOTER
# ======================
cards_html = "".join(
    f'<div style="padding:16px 20px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);">'
    f'<div style="font-size:0.875rem;font-weight:600;color:var(--text-primary);">{name}</div>'
    f'<div style="font-size:0.78rem;color:var(--text-secondary);margin-top:2px;font-family:var(--font-mono);">{cid}</div>'
    f'</div>'
    for name, cid in CONTRIBUTORS
)
st.markdown(
    f"""
    <div style="border-top:1px solid var(--border);padding-top:2.5rem;margin-top:3rem;">
        <p style="font-size:0.72rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:var(--text-secondary);margin:0 0 1.25rem;">Contributors</p>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.75rem;">
            {cards_html}
        </div>
        <p style="font-size:0.78rem;color:var(--text-secondary);margin-top:2rem;">
            GraphCodeBERT Security Module &middot; 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
