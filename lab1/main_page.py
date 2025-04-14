import streamlit as st

st.title("Лабораторная работа №1 по численным методам")

st.write("""
Выполнил: Дударь Юрий М8О-309Б-22. 
""")

st.sidebar.title("Выбор метода")

pages_method = {
    "--------": [
        st.Page("lu_decomposition.py", title="lu_decomposition", icon="🏦"),
        st.Page("progonka.py", title="progonka", icon="🛢️"),
        st.Page("seidel_iteration_method.py", title="seidel_iteration_method", icon="⛓️"),
        st.Page("jacobi_rotation_method.py", title="jacobi_rotation_method", icon="🪜"),
        st.Page("qr_decomposition.py", title="qr_decomposition", icon="🧮"),
    ],
}

pg_methood = st.navigation(pages_method)
pg_methood.run()

