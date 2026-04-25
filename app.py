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

cwe_data = [
    {
        "id": "CWE-79",
        "name": "Cross-Site Scripting (XSS)",
        "severity": "High",
        "color": "#f0a500",
        "description": "Attackers inject malicious scripts into web pages viewed by other users. The browser executes the script as if it came from a trusted source, enabling session hijacking, credential theft, or page defacement.",
        "fix": "Sanitize and escape all user input before rendering it in HTML. Use libraries like `bleach` in Python.",
        "example": 'name = request.args.get("name")\nreturn f"<h1>Hello {name}</h1>"  # ❌ vulnerable'
    },
    {
        "id": "CWE-89",
        "name": "SQL Injection",
        "severity": "Critical",
        "color": "#e05252",
        "description": "Malicious SQL statements inserted into input fields allow attackers to read, modify, or delete database content — or even execute admin operations.",
        "fix": "Always use parameterized queries or ORM methods. Never concatenate user input into SQL strings.",
        "example": 'query = "SELECT * FROM users WHERE id = " + user_id  # ❌ vulnerable'
    },
    {
        "id": "CWE-94",
        "name": "Code Injection",
        "severity": "Critical",
        "color": "#e05252",
        "description": "Attacker-controlled input is passed to functions like eval() or exec(), causing arbitrary code to run with the application's privileges.",
        "fix": "Never use eval() or exec() on user input. Use safe alternatives like ast.literal_eval() for data parsing.",
        "example": "eval(request.args.get('expr'))  # ❌ vulnerable"
    },
    {
        "id": "CWE-601",
        "name": "Open Redirect",
        "severity": "Medium",
        "color": "#4fc3f7",
        "description": "The application redirects users to an attacker-controlled URL, enabling phishing attacks that appear to originate from a trusted domain.",
        "fix": "Validate redirect URLs against an allowlist of trusted domains. Never redirect to arbitrary user-supplied URLs.",
        "example": 'next_url = request.args.get("next")\nreturn redirect(next_url)  # ❌ vulnerable'
    },
    {
        "id": "CWE-22",
        "name": "Path Traversal",
        "severity": "High",
        "color": "#f0a500",
        "description": "Attackers use sequences like '../' in file paths to escape the intended directory and access sensitive system files like /etc/passwd.",
        "fix": "Use os.path.realpath() and verify the resolved path starts with your allowed base directory.",
        "example": 'open("/var/www/uploads/" + filename)  # ❌ vulnerable'
    },
    {
        "id": "CWE-78",
        "name": "OS Command Injection (RCE)",
        "severity": "Critical",
        "color": "#e05252",
        "description": "User input is passed directly to shell commands, allowing attackers to execute arbitrary operating system commands on the server.",
        "fix": "Use subprocess with a list of arguments (not shell=True) and never pass user input to os.system().",
        "example": 'os.system("ping " + user_input)  # ❌ vulnerable'
    },
    {
        "id": "CWE-352",
        "name": "Cross-Site Request Forgery (CSRF)",
        "severity": "Medium",
        "color": "#4fc3f7",
        "description": "Attackers trick authenticated users into unknowingly submitting malicious requests, exploiting the trust a site has in the user's browser.",
        "fix": "Use CSRF tokens on all state-changing forms. Frameworks like Django include this by default.",
        "example": '# No CSRF token on form submission  # ❌ vulnerable'
    },
]

severity_icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡"}

# Dropdown to pick CWE
cwe_names = [f"{c['id']} — {c['name']}" for c in cwe_data]
selected = st.selectbox("Select a vulnerability type to learn more:", cwe_names)

# Find selected CWE
selected_cwe = next(c for c in cwe_data if f"{c['id']} — {c['name']}" == selected)

# Display card
st.markdown(f"""
<div class='card' style='text-align:left; border-left: 4px solid {selected_cwe["color"]}; padding: 20px;'>
    <div style='font-size:20px; font-weight:700; margin-bottom:8px;'>
        {severity_icon[selected_cwe["severity"]]} {selected_cwe["id"]} — {selected_cwe["name"]}
    </div>
    <div style='margin-bottom:6px;'>
        <span class='badge'>Severity: {selected_cwe["severity"]}</span>
    </div>
    <br>
    <div style='margin-bottom:12px;'>
        <b>📖 What is it?</b><br>{selected_cwe["description"]}
    </div>
    <div style='margin-bottom:12px;'>
        <b>🛠️ How to fix it:</b><br>{selected_cwe["fix"]}
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<b>❌ Vulnerable code example:</b>", unsafe_allow_html=True)
st.code(selected_cwe["example"], language="python")

# Overview grid
st.markdown("<br><b>📋 All 7 Vulnerability Types</b>", unsafe_allow_html=True)
cols = st.columns(7)
for i, cwe in enumerate(cwe_data):
    with cols[i]:
        st.markdown(f"""
        <div class='card' style='border-top: 3px solid {cwe["color"]}; padding:10px;'>
            <div style='font-size:11px; font-weight:700;'>{cwe["id"]}</div>
            <div style='font-size:10px; color:gray;'>{severity_icon[cwe["severity"]]}</div>
        </div>
        """, unsafe_allow_html=True)