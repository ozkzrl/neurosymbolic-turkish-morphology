import streamlit as st
import requests
import plotly.express as px

st.set_page_config(page_title="Hybrid NLP Academic Panel", layout="wide")

CORE_API_URL = "http://nlp-core:8000"

st.title("🔬 Hybrid Neuro-Symbolic NLP Engine")
st.subheader("Academic Evaluation & Analysis Dashboard")
st.markdown("---")

tab1, tab2 = st.tabs(["📌 Morphotactic FSA Parser", "🧠 Semantic Vector Search (RAG)"])

# TAB 1: MORFOLOJİK OTOMAT ANALİZİ
with tab1:
    st.header("Deterministic Finite State Automata Parsing")
    input_text = st.text_area("Enter Turkish text for morphological validation:", "kitapları kalemleri")
    
    if st.button("Run FSA Parser"):
        try:
            response = requests.post(f"{CORE_API_URL}/analyze-and-index", json={"text": input_text, "doc_id": 999})
            if response.status_code == 200:
                results = response.json().get("symbolic_fsa_analysis", [])
                
                for r in results:
                    if r["status"] == "ACCEPTED":
                        st.success(f"**Word:** {r['word']} -> **Root:** {r['root']} (ACCEPTED)")
                        st.caption(f"Transition Path: {' -> '.join(r['automated_path'])}")
                    else:
                        st.error(f"**Word:** {r['word']} -> REJECTED")
                        st.caption(f"Reason: {r.get('reason')}")
        except Exception as e:
            st.error(f"Cannot connect to NLP Core Service: {e}")

# TAB 2: VEKTÖREL ARAMA VE SEMANTİK SKORLAR
with tab2:
    st.header("Neural Semantic Search Evaluation")
    search_query = st.text_input("Enter semantic query:", "kütüphane kaynakları")
    
    if st.button("Search Vector Space"):
        try:
            response = requests.post(f"{CORE_API_URL}/semantic-search", json={"query": search_query})
            if response.status_code == 200:
                matches = response.json().get("matches", [])
                
                if matches:
                    texts = [m["text"] for m in matches]
                    scores = [m["score"] for m in matches]
                    
                    fig = px.bar(x=scores, y=texts, orientation='h', labels={'x':'Cosine Similarity Score', 'y':'Document'}, title="Semantic Proximity Analysis", range_x=[0,1])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    for m in matches:
                        st.info(f"**Doc ID:** {m['doc_id']} | **Similarity Score:** {m['score']:.4f}")
                        st.write(f"Text: *{m['text']}*")
                else:
                    st.warning("No matches found in Qdrant vector space.")
        except Exception as e:
            st.error(f"Error connecting to Vector Search API: {e}")

