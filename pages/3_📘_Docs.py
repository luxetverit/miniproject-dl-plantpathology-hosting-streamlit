import streamlit as st

if st.button('app.py code view'):
    code = '''python
        st.text('enter code🍀')
    '''
    st.code(code, language='python')