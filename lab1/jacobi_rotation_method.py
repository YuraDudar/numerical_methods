import streamlit as st
import numpy as np
import pandas as pd
import math
import time

# --- Функции для Метода Вращений Якоби ---
def is_symmetric(A, tol=1e-8):
    """Проверяет, является ли матрица A симметричной с заданной точностью."""
    return np.allclose(A, A.T, atol=tol)

def find_max_off_diagonal(A):
    """
    Находит индексы (p, q) максимального по модулю наддиагонального элемента матрицы A.
    Возвращает p, q, max_val
    """
    n = A.shape[0]
    max_val = 0.0
    p, q = -1, -1
    for i in range(n):
        for j in range(i + 1, n): # Смотрим только наддиагональные элементы
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j
    return p, q, max_val

def jacobi_rotation(A_input, epsilon, max_iterations=100):
    """
    Находит собственные значения и векторы симметричной матрицы A
    методом вращений Якоби.

    Возвращает:
        eigenvalues (np.array): Найденные собственные значения.
        eigenvectors (np.array): Матрица, столбцы которой - собственные векторы.
        iterations (int): Количество выполненных итераций (вращений).
        converged (bool): Сошелся ли метод.
        history (dict): История максимальных внедиагональных элементов.
        elapsed_time (float): Затраченное время.
    """
    if not is_symmetric(A_input):
        raise ValueError("Метод Якоби применим только к симметричным матрицам.")

    n = A_input.shape[0]
    A = A_input.astype(float).copy() # Рабочая копия матрицы
    V = np.identity(n)              # Матрица собственных векторов (изначально единичная)
    iterations = 0
    history = {'iter': [], 'max_off_diag': []}
    converged = False

    start_time = time.time()

    while iterations < max_iterations:
        p, q, max_off_diag_val = find_max_off_diagonal(A)

        # Запись в историю ДО проверки на сходимость
        history['iter'].append(iterations)
        history['max_off_diag'].append(max_off_diag_val)


        # Проверка условия остановки
        if max_off_diag_val < epsilon:
            converged = True
            break

        # Элементы для расчета угла вращения
        a_pp = A[p, p]
        a_qq = A[q, q]
        a_pq = A[p, q]

        # Расчет c = cos(theta) и s = sin(theta)
        if abs(a_pq) < 1e-20: # Если элемент уже почти ноль, пропускаем (маловероятно из-за find_max_off_diagonal)
             c = 1.0
             s = 0.0
        else:
            tau = (a_qq - a_pp) / (2.0 * a_pq)
            if tau >= 0:
                t = 1.0 / (tau + math.sqrt(1.0 + tau**2))
            else:
                t = -1.0 / (-tau + math.sqrt(1.0 + tau**2))
            c = 1.0 / math.sqrt(1.0 + t**2)
            s = c * t

        # Обновление матрицы A (только затрагиваемые элементы)
        A_new_pp = c*c*a_pp - 2*s*c*a_pq + s*s*a_qq
        A_new_qq = s*s*a_pp + 2*s*c*a_pq + c*c*a_qq

        A[p, p] = A_new_pp
        A[q, q] = A_new_qq
        A[p, q] = 0.0 # Обнуляем элемент
        A[q, p] = 0.0 # И симметричный ему

        # Обновление остальных элементов в строках/столбцах p и q
        for k in range(n):
            if k != p and k != q:
                a_pk = A[p, k]
                a_qk = A[q, k]
                A[p, k] = c * a_pk - s * a_qk
                A[q, k] = s * a_pk + c * a_qk
                A[k, p] = A[p, k] # Поддерживаем симметричность
                A[k, q] = A[q, k]

        # Обновление матрицы собственных векторов V
        for k in range(n):
            v_kp = V[k, p]
            v_kq = V[k, q]
            V[k, p] = c * v_kp - s * v_kq
            V[k, q] = s * v_kp + c * v_kq

        iterations += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    eigenvalues = np.diag(A) # Собственные значения на диагонали A
    # Отсортируем собственные значения и соответствующие векторы по убыванию
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = V[:,idx]


    # Добавляем финальное значение ошибки (если сошлось)
    if converged:
         _, _, final_max_off_diag = find_max_off_diagonal(A)
         history['iter'].append(iterations)
         history['max_off_diag'].append(final_max_off_diag)


    return eigenvalues, eigenvectors, iterations, converged, history, elapsed_time

# --- GUI ---
st.title("4. Метод вращений Якоби")
st.markdown("""
Нахождение собственных значений (**λ**) и собственных векторов (**v**) **симметричной** матрицы **A** (удовлетворяющих уравнению **Av = λv**) методом вращений Якоби.

Метод последовательно обнуляет внедиагональные элементы матрицы **A** с помощью матриц вращения **R(p, q, θ)**, приводя её к диагональному виду:
**Λ = ... R₂ᵀ R₁ᵀ A R₁ R₂ ...**
Диагональные элементы полученной матрицы **Λ** являются собственными значениями.
Столбцы матрицы **V = R₁ R₂ ...** являются соответствующими собственными векторами.

**Условие применимости:** Матрица **A** должна быть **симметричной**.
""")
st.divider()

# --- Функция ввода матрицы 3x3 и параметров ---
def input_matrix_3x3_jacobi():
    st.subheader("Ввод симметричной матрицы 3×3 и параметров")
    A = np.zeros((3, 3), dtype=float)

    # Значения по умолчанию из примера
    default_A = [
        [9, 2, -7],
        [2, -4, -1],
        [-7, -1, 1]
    ]

    cols_header = st.columns(3)
    headers = [f"Столбец {j+1}" for j in range(3)]
    for col, header in zip(cols_header, headers):
        col.markdown(f"**{header}**")

    input_container = st.container()
    with input_container:
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                # Предлагаем значение по умолчанию
                default_val = float(default_A[i][j])
                # Позволяем пользователю ввести свое
                A[i, j] = cols[j].number_input(
                    f"A[{i},{j}]", value=default_val,
                    label_visibility="collapsed", key=f"A_{i}_{j}"
                )

    # Ввод точности и макс итераций
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        epsilon_input = st.number_input(
            "Требуемая точность (ε):",
            min_value=1e-15, max_value=1.0, value=1e-6, format="%.1e", step=1e-6,
            help="Критерий остановки: максимальный по модулю внедиагональный элемент < ε"
        )
    with col_param2:
        max_iter_input = st.number_input(
            "Максимальное число итераций:",
            min_value=1, max_value=10000, value=100, step=10,
            help="Предотвращение бесконечного цикла."
        )

    return A, epsilon_input, int(max_iter_input)

# --- Ввод данных ---
A_input, epsilon, max_iterations = input_matrix_3x3_jacobi()

st.divider()

# --- Кнопка Решить ---
if st.button("Найти собственные значения и векторы", type="primary"):

    st.subheader("Исходная матрица и параметры:")
    col_mat, col_param = st.columns([2,1])
    with col_mat:
        st.markdown("**Матрица A:**")
        st.dataframe(pd.DataFrame(A_input))
    with col_param:
         st.metric("Точность ε", f"{epsilon:.1e}")
         st.metric("Макс. итераций", max_iterations)

    st.divider()
    st.subheader("Проверка условий и Ход решения:")

    # 1. Проверка на симметричность
    if not is_symmetric(A_input):
        st.error(f"Ошибка: Введенная матрица не является симметричной (с точностью {1e-8}). Метод Якоби неприменим.")
    else:
        st.success("Матрица является симметричной.")

        # 2. Запуск метода Якоби
        try:
            eigenvalues, eigenvectors, iterations, converged, history, elapsed_time = jacobi_rotation(
                A_input, epsilon, max_iterations
            )

            st.markdown(f"**Результаты после {iterations} итераций (время: {elapsed_time:.4f} сек):**")

            if converged:
                st.success(f"Метод сошелся за {iterations} итераций.")
            else:
                st.warning(f"Метод не сошелся за {max_iterations} итераций. Достигнутая точность (макс. внедиаг. элемент): {history['max_off_diag'][-1]:.2e}")

            # Отображение результатов
            col_val, col_vec = st.columns(2)
            with col_val:
                st.markdown("**Найденные собственные значения (λ):**")
                st.dataframe(pd.DataFrame(eigenvalues, columns=['Значение']).style.format("{:.6f}"))
            with col_vec:
                st.markdown("**Найденные собственные векторы (V - столбцы):**")
                st.dataframe(pd.DataFrame(eigenvectors, columns=[f"v{i+1}" for i in range(len(eigenvalues))]).style.format("{:.6f}"))

            # Анализ погрешности (в expander)
            st.divider()
            with st.expander("Анализ сходимости (Погрешность от числа итераций)"):
                if history['iter']:
                     hist_df = pd.DataFrame({
                        'Итерация': history['iter'],
                        'Макс. внедиаг. элемент |A[p,q]|': history['max_off_diag']
                    })
                     st.dataframe(hist_df.style.format({'Макс. внедиаг. элемент |A[p,q]|': '{:.4e}'}))

                     # Построение графика сходимости
                     st.line_chart(hist_df.rename(columns={'Макс. внедиаг. элемент |A[p,q]|':'Max Off-Diagonal Element'}).set_index('Итерация'))
                     st.caption("График показывает уменьшение максимального внедиагонального элемента с каждой итерацией (в линейном масштабе).")
                else:
                     st.info("История итераций пуста (возможно, метод сошелся за 0 итераций).")


            # Дополнительная проверка (опционально)
            with st.expander("Проверка AV ≈ VΛ"):
                 Lambda = np.diag(eigenvalues)
                 AV = A_input @ eigenvectors
                 VL = eigenvectors @ Lambda
                 diff = np.linalg.norm(AV - VL)
                 st.write("Норма разности ||AV - VΛ||:", diff)
                 st.write("AV:")
                 st.dataframe(pd.DataFrame(AV).style.format("{:.6f}"))
                 st.write("VΛ:")
                 st.dataframe(pd.DataFrame(VL).style.format("{:.6f}"))


        except ValueError as e:
            st.error(f"Ошибка выполнения: {e}")
        except Exception as e:
             st.error(f"Непредвиденная ошибка: {e}")
             # import traceback
             # st.error(traceback.format_exc())
