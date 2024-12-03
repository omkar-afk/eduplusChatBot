import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Eduplus Chatbot")

with st.sidebar:
    with st.form(key='my_form'):
        query = st.sidebar.text_area(
            label="Ask me any Query?",
            max_chars=50,
            key="query"
            )
        "[View the source code](https://github.com/omkar-afk/eduplusChatBot/tree/main)"
        submit_button = st.form_submit_button(label='Submit')

if query :
    
        db = lch.create_db()
        response, docs = lch.get_response_from_query(db, query)
        st.subheader("Answer:")
        st.text(response)
        st.subheader("docs:")
        st.text(docs)