import streamlit as st

st.title("Лабораторная работа №3 по численным методам")

st.write("""
Выполнил: Дударь Юрий М8О-309Б-22. 
""")

st.sidebar.title("Выбор метода")

pages_method = {
    "--------": [
        st.Page("3.1.py", title="3.1", icon="🏦"),
        st.Page("3.2.py", title="3.2", icon="🏦"),
        st.Page("3.3.py", title="3.3", icon="🏦"),
        st.Page("3.4.py", title="3.4", icon="🏦"),
        st.Page("3.5.py", title="3.5", icon="🏦"),
    ]
}

pg_methood = st.navigation(pages_method)
pg_methood.run()

