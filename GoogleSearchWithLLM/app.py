from search import SemanticSearch, GoogleSearch, Document
import streamlit as st
from model import RAGModel, load_configs


def run_on_start():
    
    if "configs" not in st.session_state:
        st.session_state.configs = configs = load_configs(config_file="rag.configs.yml")
    if "model" not in st.session_state:
        st.session_state.model = RAGModel(configs)
   
run_on_start()


def search(query):
    g = GoogleSearch(query)
    data = g.all_page_data
    d = Document(data, min_char_len=st.session_state.configs["document"]["min_char_length"])
    st.session_state.doc = d.doc()


st.title("Search Here Instead of Google")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc" not in st.session_state:
    st.session_state.doc = None

if "refresh" not in st.session_state:
    st.session_state.refresh = True 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Search Here insetad of Google"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    configs = st.session_state.configs
    if st.session_state.refresh:
        st.session_state.refresh = False 
        search(prompt)

    s = SemanticSearch(
        st.session_state.doc,
        configs["model"]["embeding_model"],
        configs["model"]["device"],
    )
    topk, u = s.semantic_search(query=prompt, k=32)
    output = st.session_state.model.answer_query(query=prompt, topk_items=topk)
    response = output
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    