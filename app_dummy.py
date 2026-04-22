import random
from typing import Dict, List

import streamlit as st


st.set_page_config(
	page_title="Code Security Identifier - Dummy UI",
	page_icon="Shield",
	layout="wide",
)


SAMPLE_SNIPPETS: Dict[str, str] = {
	"SQL query builder": """def find_user(conn, username):
	query = f\"SELECT * FROM users WHERE username = '{username}'\"
	return conn.execute(query).fetchall()
""",
	"Safe parameterized query": """def find_user(conn, username):
	query = \"SELECT * FROM users WHERE username = ?\"
	return conn.execute(query, (username,)).fetchall()
""",
	"Command execution": """import os

def run_backup(target):
	os.system(f\"tar -czf /tmp/backup.tar.gz {target}\")
""",
}


def hardcoded_prediction(selected_profile: str) -> Dict[str, object]:
	"""Return deterministic-looking dummy predictions for UI testing."""
	profiles = {
		"High risk": {
			"vulnerability_score": 0.91,
			"is_vulnerable": True,
			"top_cwe": "CWE-089: SQL Injection",
			"severity": "High",
			"confidence": 0.88,
			"line_risk": [0.22, 0.74, 0.95, 0.31, 0.11],
		},
		"Medium risk": {
			"vulnerability_score": 0.57,
			"is_vulnerable": True,
			"top_cwe": "CWE-078: OS Command Injection",
			"severity": "Medium",
			"confidence": 0.71,
			"line_risk": [0.17, 0.48, 0.68, 0.42, 0.23],
		},
		"Low risk": {
			"vulnerability_score": 0.14,
			"is_vulnerable": False,
			"top_cwe": "None detected",
			"severity": "Low",
			"confidence": 0.93,
			"line_risk": [0.05, 0.07, 0.11, 0.08, 0.04],
		},
	}
	result = dict(profiles[selected_profile])

	# Small random jitter keeps interactions realistic while staying bounded.
	jitter = random.uniform(-0.03, 0.03)
	score = min(max(result["vulnerability_score"] + jitter, 0.0), 1.0)
	result["vulnerability_score"] = round(score, 3)
	result["confidence"] = round(min(max(result["confidence"] + jitter, 0.0), 1.0), 3)
	return result


def render_header() -> None:
	st.markdown(
		"""
		<style>
		.title-wrap {
			padding: 1rem 1.25rem;
			border-radius: 12px;
			background: linear-gradient(120deg, #0f4c5c 0%, #1f7a8c 100%);
			color: #ffffff;
			margin-bottom: 1rem;
		}
		.subtitle {
			opacity: 0.9;
			margin-top: 0.2rem;
		}
		</style>
		<div class=\"title-wrap\">
			<h2 style=\"margin:0;\">Code Security Identifier - Streamlit Dummy UI</h2>
			<p class=\"subtitle\">Use hardcoded predictions to validate layout and interactivity before model integration.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_sidebar() -> Dict[str, object]:
	st.sidebar.header("Test Controls")
	profile = st.sidebar.selectbox("Prediction profile", ["High risk", "Medium risk", "Low risk"])
	show_probs = st.sidebar.toggle("Show line risk chart", value=True)
	seed = st.sidebar.slider("Random seed", min_value=0, max_value=100, value=42)
	random.seed(seed)
	return {"profile": profile, "show_probs": show_probs}


def render_inputs() -> str:
	col1, col2 = st.columns([3, 2])
	with col1:
		st.subheader("Input Code")
		selected_sample = st.selectbox("Load sample snippet", list(SAMPLE_SNIPPETS.keys()))
		code_text = st.text_area(
			"Code snippet",
			value=SAMPLE_SNIPPETS[selected_sample],
			height=260,
			placeholder="Paste Python code here...",
		)
	with col2:
		st.subheader("Model Mode")
		st.radio("Backend", ["Dummy only", "Future real model"], index=0)
		st.caption("Current run uses hardcoded outputs to test UX behavior.")
	return code_text


def render_prediction(result: Dict[str, object], show_probs: bool) -> None:
	st.subheader("Prediction Output")
	m1, m2, m3, m4 = st.columns(4)
	m1.metric("Vulnerability Score", f"{result['vulnerability_score']:.2f}")
	m2.metric("Vulnerable", "Yes" if result["is_vulnerable"] else "No")
	m3.metric("Severity", str(result["severity"]))
	m4.metric("Confidence", f"{result['confidence']:.2f}")

	st.info(f"Top CWE Prediction: {result['top_cwe']}")

	if show_probs:
		st.caption("Dummy line-level risk probabilities")
		chart_data: List[float] = result["line_risk"]
		st.bar_chart(chart_data)


def main() -> None:
	render_header()
	controls = render_sidebar()
	code_text = render_inputs()

	predict_clicked = st.button("Run Dummy Prediction", type="primary", use_container_width=True)

	if predict_clicked:
		if not code_text.strip():
			st.error("Please provide a code snippet first.")
			return

		with st.spinner("Generating dummy predictions..."):
			result = hardcoded_prediction(controls["profile"])
		render_prediction(result, controls["show_probs"])
	else:
		st.warning("Click 'Run Dummy Prediction' to test UI interactions.")


if __name__ == "__main__":
	main()
