import streamlit as st

st.set_page_config(
    page_title='4leaf',
    layout='centered',
)

st.header('Code Viewer🧱')

if st.button('app.py code view'):
    code = '''
        st.text('enter code🍀')
    '''
    st.code(code, language='python')