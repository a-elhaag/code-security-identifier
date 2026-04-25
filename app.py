import streamlit as st
import pandas as pd
import time
import ast
import re
import html

# ======================
# CONFIG
# ======================
st.set_page_config(
    page_title="Code Intelligence Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CLEAN MODERN CSS
# ======================
st.markdown("""
<style>
.main-title {
    font-size:34px;
    font-weight:700;
    margin-bottom:5px;
}
.subtitle {
    color:gray;
    margin-bottom:25px;
}
.section {
    margin-top:35px;
    margin-bottom:15px;
    font-size:22px;
    font-weight:600;
}
.card {
    background:#1c1f26;
    padding:18px;
    border-radius:12px;
    text-align:center;
}
.metric {
    font-size:26px;
    font-weight:bold;
}
.code-box {
    background:#0e1117;
    padding:20px;
    border-radius:12px;
    overflow:auto;
}
.badge {
    background:#2d2f36;
    padding:6px 10px;
    border-radius:8px;
    font-size:12px;
    margin-right:6px;
}
.legend span {
    margin-right:12px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# HEADER
# ======================
st.markdown("<div class='main-title'>🚀 Code Intelligence Studio</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze • Highlight • Understand Code Structure</div>", unsafe_allow_html=True)

# ======================
# SIDEBAR (CLEAN GROUPED)
# ======================
st.sidebar.markdown("## ⚙️ Controls")

st.sidebar.markdown("### Mode")
use_demo = st.sidebar.checkbox("Demo Data")
use_input = st.sidebar.checkbox("Manual Input")

st.sidebar.markdown("### Search")
search_word = st.sidebar.text_input("Keyword")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("final_dataset_ready.csv")

if use_input:
    st.markdown("<span class='badge'>✍️ Manual Mode</span>", unsafe_allow_html=True)

    user_code = st.text_area("Enter your code:", height=250)

    df = pd.DataFrame([{
        "code": user_code,
        "functions": "[]",
        "variables": "[]",
        "classes": "[]"
    }])

elif use_demo:
    st.markdown("<span class='badge'>🧪 Demo Mode</span>", unsafe_allow_html=True)

    df = pd.DataFrame([
        {
            "code": """class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        print(self.name)

x = Dog("Max")
x.bark()
""",
            "functions": "['__init__','bark']",
            "variables": "['name','x']",
            "classes": "['Dog']"
        }
    ])

else:
    with st.spinner("Loading dataset..."):
        time.sleep(1)
        df = load_data()

# ======================
# INDEX
# ======================
index = st.sidebar.slider("Select Code", 0, len(df)-1, 0)
row = df.iloc[index]

# ======================
# PARSE
# ======================
def safe_parse(val):
    try:
        return ast.literal_eval(val)
    except:
        return []

functions = safe_parse(row.get("functions", "[]"))
variables = safe_parse(row.get("variables", "[]"))
classes = safe_parse(row.get("classes", "[]"))

code = str(row.get("code", ""))

# ======================
# AUTO EXTRACTION
# ======================
if use_input and code.strip():
    functions = re.findall(r'def\s+(\w+)', code)
    classes = re.findall(r'class\s+(\w+)', code)
    variables = re.findall(r'(\w+)\s*=', code)

# ======================
# HIGHLIGHT
# ======================
def replace_word(text, word, color):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(pattern,
        f"<span style='color:{color}; font-weight:bold'>{word}</span>",
        text)

def highlight(code):
    for c in classes:
        code = replace_word(code, c, "red")
    for f in functions:
        code = replace_word(code, f, "cyan")
    for v in variables:
        code = replace_word(code, v, "lightgreen")

    if search_word:
        code = replace_word(code, search_word, "yellow")

    return code

if not code.strip():
    st.info("No code provided")
    st.stop()

escaped = html.escape(code)
highlighted = highlight(escaped)

# ======================
# METRICS SECTION
# ======================
st.markdown("<div class='section'>📊 Code Metrics</div>", unsafe_allow_html=True)

lines = len(code.split("\n"))

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"<div class='card'><div>Classes</div><div class='metric'>{len(classes)}</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='card'><div>Functions</div><div class='metric'>{len(functions)}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='card'><div>Variables</div><div class='metric'>{len(variables)}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='card'><div>Lines</div><div class='metric'>{lines}</div></div>", unsafe_allow_html=True)

# ======================
# MAIN LAYOUT
# ======================
left, right = st.columns([3,1])

# ======================
# CODE VIEW
# ======================
with left:
    st.markdown("<div class='section'>💻 Code Viewer</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='legend'>
        <span style='color:red'>■ Classes</span>
        <span style='color:cyan'>■ Functions</span>
        <span style='color:lightgreen'>■ Variables</span>
        <span style='color:yellow'>■ Search</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='code-box'><pre>{highlighted}</pre></div>", unsafe_allow_html=True)

# ======================
# INSIGHTS
# ======================
with right:
    st.markdown("<div class='section'>🧠 Insights</div>", unsafe_allow_html=True)

    if not classes:
        st.caption("No classes detected")
    if not functions:
        st.caption("No functions detected")
    if not variables:
        st.caption("No variables detected")

    st.markdown("### Tips")
    st.caption("• Use meaningful names")
    st.caption("• Avoid deep nesting")
    st.caption("• Add comments")

# ======================
# CWE SECTION
# ======================
st.markdown("<div class='section'>🔐 Security Insights (CWE)</div>", unsafe_allow_html=True)

cwe_data = {
    "CWE-79 (XSS)": "Sanitize user input before rendering.",
    "CWE-89 (SQL Injection)": "Use parameterized queries.",
    "CWE-20 (Input Validation)": "Validate all inputs.",
    "CWE-200 (Info Exposure)": "Protect sensitive data.",
    "CWE-22 (Path Traversal)": "Validate file paths.",
    "CWE-287 (Auth Bypass)": "Enforce authentication.",
    "CWE-352 (CSRF)": "Use CSRF tokens."
}

choice = st.selectbox("Select CWE", list(cwe_data.keys()))

st.markdown(f"""
<div class='card'>
<b>{choice}</b><br><br>
{cwe_data[choice]}
</div>
""", unsafe_allow_html=True)

# ======================
# FOOTER
# ======================
st.markdown("---")
st.caption("Built with Streamlit • Code Analysis Tool")
