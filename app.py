import streamlit as st


st.set_page_config(
	page_title="Code Security Identifier",
	page_icon="🛡️",
	layout="wide",
)


def main() -> None:
	st.title("Code Security Identifier")
	st.caption("Basic Streamlit layout with code input area.")

	col_left, col_right = st.columns([2, 1])

	with col_left:
		st.subheader("Input Code")
		code = st.text_area(
			"Paste your code here",
			height=280,
			placeholder="Write or paste code snippet...",
		)

		if st.button("Submit"):
			if code.strip():
				st.success("Code received successfully.")
			else:
				st.warning("Please enter some code before submitting.")

	with col_right:
		st.subheader("Preview")
		st.info("This is a simple UI skeleton for later model integration.")
		st.write("Current input length:", len(code))
		st.write("Lines:", len(code.splitlines()) if code else 0)

	st.divider()
	st.subheader("Notes")
	st.markdown(
		"- This app is only a UI starter.\n"
		"- You can connect a model later.\n"
		"- Run it with: `python -m streamlit run app.py`"
	)


if __name__ == "__main__":
	main()
