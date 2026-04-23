import streamlit as st
import pandas as pd
import time
import ast
import re
import html

# ======================
# Page config (optional but professional)
# ======================
st.set_page_config(page_title="Code Highlighter", layout="wide")

st.title("Code Identifier Highlighter 🔍")

# ======================
# Cache dataset loading (performance fix)
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("final_dataset_ready.csv")

with st.spinner("Loading dataset..."):
    time.sleep(1)  # optional UX delay
    df = load_data()

st.success("Dataset loaded successfully!")

# ======================
# Input selection
# ======================
index = st.number_input(
    "Choose code index",
    min_value=0,
    max_value=len(df) - 1,
    step=1,
    key="code_index"
)

row = df.iloc[index]

# ======================
# Safe parsing (replace eval)
# ======================
def safe_parse(value):
    try:
        return ast.literal_eval(value)
    except:
        return []

functions = safe_parse(row.get("functions", "[]"))
variables = safe_parse(row.get("variables", "[]"))
classes = safe_parse(row.get("classes", "[]"))

code = str(row.get("code", ""))

# ======================
# Safe highlighting function (regex-based)
# ======================
def safe_replace(text, word, color):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.sub(
        pattern,
        f"<span style='color:{color}; font-weight:bold'>{word}</span>",
        text
    )

def highlight_code(code, classes, functions, variables):
    # order matters (avoid overwriting)
    for c in classes:
        code = safe_replace(code, c, "red")

    for f in functions:
        code = safe_replace(code, f, "blue")

    for v in variables:
        code = safe_replace(code, v, "green")

    return code

# ======================
# Edge cases
# ======================
if not code.strip():
    st.warning("No code available for this index.")
    st.stop()

if not functions:
    st.info("No functions found in this code.")
if not variables:
    st.info("No variables found in this code.")
if not classes:
    st.info("No classes found in this code.")

# ======================
# Escape + highlight
# ======================
escaped_code = html.escape(code)
highlighted_code = highlight_code(escaped_code, classes, functions, variables)

# ======================
# Display
# ======================
st.markdown("### Highlighted Code")

if len(code) > 1000:
    st.text_area("Scrollable Code View", highlighted_code, height=300)
else:
    st.markdown(
        f"<div style='background-color:#111; padding:10px; border-radius:8px; color:white'>"
        f"<pre>{highlighted_code}</pre></div>",
        unsafe_allow_html=True
    )

# ======================
# Fix suggestions
# ======================
st.markdown("### Fix Suggestions")

fix_suggestions = [
    "Use clear and consistent variable names.",
    "Add comments to explain complex logic.",
    "Handle exceptions using try/except.",
    "Optimize loops for better performance.",
    "Use functions to avoid repetition.",
    "Validate input data before processing.",
    "Follow PEP8 style guidelines."
]

for s in fix_suggestions:
    st.write("•", s)

# ======================
# Example snippets
# ======================
st.markdown("### Example Code Snippets")

example_snippets = [
    ("Function Example", "def greet(name):\n    return f'Hello, {name}!'"),
    ("Loop Example", "for i in range(5):\n    print(i)"),
    ("Class Example", "class Dog:\n    def __init__(self, name):\n        self.name = name"),
    ("List Comprehension", "[x**2 for x in range(10)]"),
    ("Dictionary Example", "student = {'name': 'Hend', 'age': 22}"),
    ("Try/Except", "try:\n    x = 1/0\nexcept ZeroDivisionError:\n    print('Error')"),
    ("File Handling", "with open('file.txt', 'w') as f:\n    f.write('Hello')")
]

for title, snippet in example_snippets:
    st.markdown(f"**{title}**")
    st.code(snippet, language="python")