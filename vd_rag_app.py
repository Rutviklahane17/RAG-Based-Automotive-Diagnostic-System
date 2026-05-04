import streamlit as st
from src.rag.rag_pipeline_ollama import get_diagnosis

st.set_page_config(page_title="Vehicle Diagnostic Assistant using RAG", page_icon="🔧")
st.title(" Vehicle Diagnostic Assistant ")
st.write("Enter a description of the issue or abnormalities you're experiencing with your vehicle, and the app will provide a diagnosis based on the knowledge base.")

st.subheader("Vehicle Issue Description")
user_input = st.text_area("Example: Engine temperature is high and coolant level is low", height=130)
if st.button("Get Analysis"):
    if user_input.strip() == "":
        st.warning("Please enter a valid query or description of the issue.")
    else:
        with st.spinner("Analyzing..."):
            diagnosis, results = get_diagnosis(user_input)
        st.subheader("Diagnosis Result")
        st.write(diagnosis)
        st.subheader("Source Information")
        for r in results:
            st.write(f"**Context from knowledge base:** {r['content']}")
            st.write(f"**Source:** {r['source']}")
            
            



st.markdown("<small>Note: This is a learning project to demonstrate the use of Retrieval-Augmented Generation (RAG) for vehicle diagnostics. " \
"The responses are based on a predefined knowledge base and may not cover all possible issues.</small>", unsafe_allow_html=True)
