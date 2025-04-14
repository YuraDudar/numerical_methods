import streamlit as st
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt

# --- Функции для итерационных методов ---
def check_diagonal_dominance(A):
    """
    Проверяет условие строгого диагонального преобладания матрицы A.
    Возвращает: Bool (True/False), str (Сообщение)
    """
    n = A.shape[0]
    is_dominant = True
    message = "Проверка строгого диагонального преобладания:\n"
    details = []
    for i in range(n):
        diag_element = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag_element
        check = diag_element > off_diag_sum
        details.append(f"Строка {i}: |A[{i},{i}]| = {diag_element:.4f}, Сумма остальных = {off_diag_sum:.4f}. Преобладание: {check}")
        if not check:
            is_dominant = False
    if is_dominant:
        message += "\n".join(details)
        message += "\n\nУсловие строгого диагонального преобладания ВЫПОЛНЕНО. Оба метода (простых итераций и Зейделя) должны сойтись."
        status = "success"
    else:
        message += "\n".join(details)
        message += "\n\nУсловие строгого диагонального преобладания НЕ ВЫПОЛНЕНО. Сходимость методов не гарантирована (но возможна)."
        status = "warning"
    return is_dominant, message, status

def check_zero_diagonal(A):
    """Проверяет наличие нулей на главной диагонали."""
    if np.any(np.abs(np.diag(A)) < 1e-15): # Сравнение с малым числом
        zero_indices = np.where(np.abs(np.diag(A)) < 1e-15)[0]
        return True, f"Ошибка: Нулевые или близкие к нулю элементы на главной диагонали в позициях: {zero_indices}. Методы неприменимы в текущем виде."
    return False, ""

# --- Метод Простых Итераций (Якоби) ---
def simple_iteration(A, b, epsilon, max_iterations=500, initial_guess=None):
    """
    Решает СЛАУ Ax = b методом простых итераций (Якоби).
    """
    n = len(b)
    # Проверка диагонали (уже сделана снаружи, но можно и здесь для надежности)
    has_zeros, msg = check_zero_diagonal(A)
    if has_zeros:
        raise ValueError(msg)

    if initial_guess is None:
        x = np.zeros(n) # Начальное приближение - нулевой вектор
    else:
        x = initial_guess.astype(float).copy()

    # Преобразование системы к виду x = Bx + f
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    # Итерационная форма: x_new = D^(-1) * (b - L_plus_U * x_old)
    D_inv = np.diag(1.0 / np.diag(D)) # Обратная к диагональной - просто 1/элементы

    iterations = 0
    history = {'iter': [], 'x': [], 'error': []}
    converged = False

    start_time = time.time()

    while iterations < max_iterations:
        x_old = x.copy()
        x = D_inv @ (b - L_plus_U @ x_old) # Главная формула Якоби

        # Оценка погрешности (норма разности между итерациями)
        # Используем норму бесконечности (максимальное абсолютное значение компоненты)
        error = np.linalg.norm(x - x_old, ord=np.inf)

        history['iter'].append(iterations + 1)
        history['x'].append(x.copy())
        history['error'].append(error)

        iterations += 1

        if error < epsilon:
            converged = True
            break

        # Проверка на расходимость (если значения становятся слишком большими)
        if np.any(np.isinf(x)) or np.any(np.isnan(x)) or error > 1e10:
             st.warning(f"Метод простых итераций: Возможно, расходится (iter={iterations}, error={error:.2e})")
             converged = False
             break


    end_time = time.time()
    elapsed_time = end_time - start_time

    return x, iterations, converged, history, elapsed_time

# --- Метод Зейделя ---
def gauss_seidel(A, b, epsilon, max_iterations=500, initial_guess=None):
    """
    Решает СЛАУ Ax = b методом Зейделя.
    """
    n = len(b)
    # Проверка диагонали
    has_zeros, msg = check_zero_diagonal(A)
    if has_zeros:
        raise ValueError(msg)

    if initial_guess is None:
        x = np.zeros(n) # Начальное приближение
    else:
        x = initial_guess.astype(float).copy()


    iterations = 0
    history = {'iter': [], 'x': [], 'error': []}
    converged = False

    start_time = time.time()

    while iterations < max_iterations:
        x_old = x.copy() # Сохраняем предыдущее значение для расчета ошибки

        # Итерация по компонентам x
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])       # Используем уже обновленные x[j] для j < i
            sum2 = np.dot(A[i, i+1:], x_old[i+1:]) # Используем x_old[j] для j > i
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Оценка погрешности
        error = np.linalg.norm(x - x_old, ord=np.inf)

        history['iter'].append(iterations + 1)
        history['x'].append(x.copy())
        history['error'].append(error)

        iterations += 1

        if error < epsilon:
            converged = True
            break

         # Проверка на расходимость
        if np.any(np.isinf(x)) or np.any(np.isnan(x)) or error > 1e10:
            st.warning(f"Метод Зейделя: Возможно, расходится (iter={iterations}, error={error:.2e})")
            converged = False
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    return x, iterations, converged, history, elapsed_time


# --- GUI ---
st.title("3. Итерационные методы: Простых Итераций (Якоби) и Зейделя")
st.markdown("""
Решение СЛАУ **Ax = b** итерационными методами. Эти методы строят последовательность приближений **x^(k)**, которая (в идеале) сходится к точному решению **x**.

- **Метод Простых Итераций (Якоби):** Все компоненты вектора **x^(k+1)** вычисляются на основе *всех* компонент предыдущего приближения **x^(k)**.
- **Метод Зейделя:** При вычислении компоненты **xᵢ^(k+1)** используются *уже вычисленные* на текущем шаге компоненты **xⱼ^(k+1)** (где j < i). Обычно сходится быстрее Якоби, если сходится вообще.

**Условие сходимости (достаточное):** Строгое диагональное преобладание матрицы **A**. Если оно не выполнено, сходимость не гарантирована. Методы также не работают при нулевых диагональных элементах.
""")
st.divider()

# --- Функция ввода СЛАУ 4x4 и параметров ---
def input_slu_4x4_iterative():
    st.subheader("Ввод коэффициентов СЛАУ 4×4 (Ax = b) и параметров")
    A = np.zeros((4, 4), dtype=float)
    b = np.zeros(4, dtype=float)

    # Значения по умолчанию из примера
    default_A = [
        [23, -6, -5, 9],
        [8, 22, -2, 5],
        [7, -6, 18, -1],
        [3, 5, 5, -19]
    ]
    default_b = [232, -82, 202, -57]

    cols_header = st.columns([1] * 4 + [0.2, 1]) # Колонки для A и b + разделитель
    headers = [f"x{j+1}" for j in range(4)] + ["", "="]
    for col, header in zip(cols_header, headers):
        col.markdown(f"**{header}**")

    input_container = st.container()
    with input_container:
        for i in range(4):
            cols = st.columns([1] * 4 + [0.2, 1]) # 4 для A, 1 пустая, 1 для b
            for j in range(4):
                A[i, j] = cols[j].number_input(
                    f"A[{i},{j}]", value=float(default_A[i][j]),
                    label_visibility="collapsed", key=f"A_{i}_{j}"
                )
            cols[4].write(" = ")
            b[i] = cols[5].number_input(
                f"b[{i}]", value=float(default_b[i]),
                 label_visibility="collapsed", key=f"b_{i}"
            )

    # Ввод точности и макс итераций
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        epsilon_input = st.number_input(
            "Требуемая точность (ε):",
            min_value=1e-15, max_value=1.0, value=1e-4, format="%.1e", step=1e-5,
            help="Критерий остановки: ||x^(k) - x^(k-1)|| < ε"
        )
    with col_param2:
        max_iter_input = st.number_input(
            "Максимальное число итераций:",
            min_value=1, max_value=10000, value=100, step=10,
            help="Предотвращение бесконечного цикла при расходимости."
        )

    return A, b, epsilon_input, int(max_iter_input)

# --- Ввод данных ---
A_input, b_input, epsilon, max_iterations = input_slu_4x4_iterative()

st.divider()

# --- Кнопка Решить ---
if st.button("Решить методами итераций", type="primary"):

    st.subheader("Исходная система и параметры:")
    col1_disp, col2_disp, col3_disp = st.columns([2,1,1])
    with col1_disp:
        st.markdown("**Матрица A:**")
        st.dataframe(pd.DataFrame(A_input, columns=[f"x{j+1}" for j in range(4)]))
    with col2_disp:
        st.markdown("**Вектор b:**")
        st.dataframe(pd.DataFrame(b_input, columns=["Значение"]))
    with col3_disp:
         st.metric("Точность ε", f"{epsilon:.1e}")
         st.metric("Макс. итераций", max_iterations)

    st.divider()
    st.subheader("Проверка условий сходимости:")

    # 1. Проверка на нули на диагонали
    has_zeros, zero_diag_msg = check_zero_diagonal(A_input)
    if has_zeros:
        st.error(zero_diag_msg)
    else:
        st.success("На главной диагонали нет нулевых элементов.")
        # 2. Проверка диагонального преобладания
        is_dominant, dominance_msg, dominance_status = check_diagonal_dominance(A_input)
        if dominance_status == "success":
            st.success(dominance_msg)
        else:
            st.warning(dominance_msg)
            st.info("Продолжаем вычисления, но сходимость не гарантирована.")

        st.divider()
        st.subheader("Ход решения и Результаты:")

        # --- Запуск и отображение результатов ---
        col_jacobi, col_seidel = st.columns(2)

        # --- Метод Простых Итераций (Якоби) ---
        with col_jacobi:
            st.markdown("#### Метод Простых Итераций (Якоби)")
            try:
                x_jacobi, iter_jacobi, converged_jacobi, history_jacobi, time_jacobi = simple_iteration(
                    A_input, b_input, epsilon, max_iterations
                )

                if converged_jacobi:
                    st.success(f"Сошелся за {iter_jacobi} итераций (время: {time_jacobi:.4f} сек).")
                    st.markdown("**Решение x:**")
                    st.dataframe(pd.DataFrame(x_jacobi.reshape(-1, 1), index=[f'x{i+1}' for i in range(len(x_jacobi))], columns=['Значение']).style.format("{:.6f}"))
                else:
                    st.error(f"Не сошелся за {max_iterations} итераций. Достигнутая точность: {history_jacobi['error'][-1]:.2e}")
                    st.markdown("**Последнее приближение x:**")
                    st.dataframe(pd.DataFrame(x_jacobi.reshape(-1, 1), index=[f'x{i+1}' for i in range(len(x_jacobi))], columns=['Значение']).style.format("{:.6f}"))

                # Отображение истории итераций (в expander)
                with st.expander("Показать историю итераций (Якоби)"):
                    hist_df_jacobi = pd.DataFrame({
                        'Итерация': history_jacobi['iter'],
                        'x': [f"[{', '.join(f'{val:.4f}' for val in vec)}]" for vec in history_jacobi['x']], # Форматируем вектор x
                        'Ошибка ||x_k - x_{k-1}||': history_jacobi['error']
                    })
                    st.dataframe(hist_df_jacobi.style.format({'Ошибка ||x_k - x_{k-1}||': '{:.4e}'}))

                    # Опционально: График сходимости
                    try:
                        fig, ax = plt.subplots()
                        ax.semilogy(history_jacobi['iter'], history_jacobi['error'], marker='.') # Логарифмическая шкала по Y
                        ax.set_xlabel("Итерация")
                        ax.set_ylabel("Ошибка (log scale)")
                        ax.set_title("Сходимость Якоби")
                        ax.grid(True)
                        st.pyplot(fig)
                    except Exception as plot_e:
                        st.warning(f"Не удалось построить график: {plot_e}")


            except ValueError as e:
                st.error(f"Ошибка выполнения (Якоби): {e}")
            except Exception as e:
                st.error(f"Непредвиденная ошибка (Якоби): {e}")


        # --- Метод Зейделя ---
        with col_seidel:
            st.markdown("#### Метод Зейделя")
            try:
                x_seidel, iter_seidel, converged_seidel, history_seidel, time_seidel = gauss_seidel(
                    A_input, b_input, epsilon, max_iterations
                )

                if converged_seidel:
                    st.success(f"Сошелся за {iter_seidel} итераций (время: {time_seidel:.4f} сек).")
                    st.markdown("**Решение x:**")
                    st.dataframe(pd.DataFrame(x_seidel.reshape(-1, 1), index=[f'x{i+1}' for i in range(len(x_seidel))], columns=['Значение']).style.format("{:.6f}"))
                else:
                    st.error(f"Не сошелся за {max_iterations} итераций. Достигнутая точность: {history_seidel['error'][-1]:.2e}")
                    st.markdown("**Последнее приближение x:**")
                    st.dataframe(pd.DataFrame(x_seidel.reshape(-1, 1), index=[f'x{i+1}' for i in range(len(x_seidel))], columns=['Значение']).style.format("{:.6f}"))

                # Отображение истории итераций (в expander)
                with st.expander("Показать историю итераций (Зейдель)"):
                    hist_df_seidel = pd.DataFrame({
                        'Итерация': history_seidel['iter'],
                        'x': [f"[{', '.join(f'{val:.4f}' for val in vec)}]" for vec in history_seidel['x']],
                        'Ошибка ||x_k - x_{k-1}||': history_seidel['error']
                    })
                    st.dataframe(hist_df_seidel.style.format({'Ошибка ||x_k - x_{k-1}||': '{:.4e}'}))

                    # Опционально: График сходимости
                    try:
                        fig, ax = plt.subplots()
                        ax.semilogy(history_seidel['iter'], history_seidel['error'], marker='.', color='orange')
                        ax.set_xlabel("Итерация")
                        ax.set_ylabel("Ошибка (log scale)")
                        ax.set_title("Сходимость Зейделя")
                        ax.grid(True)
                        st.pyplot(fig)
                    except Exception as plot_e:
                        st.warning(f"Не удалось построить график: {plot_e}")

            except ValueError as e:
                st.error(f"Ошибка выполнения (Зейдель): {e}")
            except Exception as e:
                st.error(f"Непредвиденная ошибка (Зейдель): {e}")
