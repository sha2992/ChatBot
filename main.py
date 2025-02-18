import streamlit as st 
import chat_ready as l

def main():
    st.title("Ask me anything!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for messages in st.session_state.messages:
        st.chat_message(messages['role']).markdown(messages['content'])


    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':"user",'content':prompt})

        response = l.answer_query(prompt)
        # print(response)
        # response = "Hi I'm Chitti"

        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role':"assistant",'content':response})

if __name__=='__main__':
    main()