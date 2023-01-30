import streamlit as st

if st.button('app.py code view'):
    code = '''
        st.text('enter code🍀')
    '''
    st.code(code, language='python')