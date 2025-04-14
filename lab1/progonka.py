import streamlit as st
import numpy as np
import pandas as pd

# --- Функции для метода прогонки (Алгоритм Томаса) ---
def check_tridiagonal_conditions(a, c, b):
    """
    Проверяет достаточное условие применимости метода прогонки
    (строгое диагональное преобладание).
    Возвращает: Bool, str (Сообщение о проверке)
    """
    n = len(c)
    is_dominant = True
    message = "Проверка условия диагонального преобладания:\n"

    # Первая строка
    if abs(c[0]) <= abs(b[0]):
        is_dominant = False
        message += f"- |c[0]|={abs(c[0]):.2f} <= |b[0]|={abs(b[0]):.2f} (Нарушено для i=0)\n"
    else:
         message += f"- |c[0]|={abs(c[0]):.2f} > |b[0]|={abs(b[0]):.2f} (Выполнено для i=0)\n"

    # Средние строки
    for i in range(1, n - 1):
        cond = abs(c[i]) > abs(a[i-1]) + abs(b[i])
        if not cond:
            is_dominant = False
            message += f"- |c[{i}]|={abs(c[i]):.2f} <= |a[{i-1}]|={abs(a[i-1]):.2f} + |b[{i}]|={abs(b[i]):.2f} (Нарушено для i={i})\n"
        else:
            message += f"- |c[{i}]|={abs(c[i]):.2f} > |a[{i-1}]|={abs(a[i-1]):.2f} + |b[{i}]|={abs(b[i]):.2f} (Выполнено для i={i})\n"


    # Последняя строка
    if abs(c[n-1]) <= abs(a[n-2]):
         is_dominant = False
         message += f"- |c[{n-1}]|={abs(c[n-1]):.2f} <= |a[{n-2}]|={abs(a[n-2]):.2f} (Нарушено для i={n-1})\n"
    else:
        message += f"- |c[{n-1}]|={abs(c[n-1]):.2f} > |a[{n-2}]|={abs(a[n-2]):.2f} (Выполнено для i={n-1})\n"


    if is_dominant:
        final_message = "Условие строгого диагонального преобладания ВЫПОЛНЕНО. Метод прогонки должен быть устойчив."
        status = "success"
    else:
        final_message = "Условие строгого диагонального преобладания НЕ ВЫПОЛНЕНО. Устойчивость метода не гарантирована (но решение может существовать, если не возникнет деления на ноль)."
        status = "warning"

    return is_dominant, final_message, status, message # Возвращаем детальное сообщение тоже


def thomas_algorithm(a, c, b, d):
    """
    Решает СЛАУ Ax=d для трехдиагональной матрицы A методом прогонки (Алгоритм Томаса).
    a: поддиагональ (размер n-1, a[i] соответствует a_{i+1} в уравнении i+1)
    c: главная диагональ (размер n)
    b: наддиагональ (размер n-1, b[i] соответствует b_{i} в уравнении i)
    d: вектор правых частей (размер n)

    Возвращает: x (решение), alpha, beta (прогоночные коэффициенты) или None, None, None в случае ошибки
    """
    n = len(c)
    if not (len(a) == n - 1 and len(b) == n - 1 and len(d) == n):
        raise ValueError("Размеры векторов диагоналей и правой части неверны.")

    alpha = np.zeros(n)
    beta = np.zeros(n)
    x = np.zeros(n)
    epsilon = 1e-12 # Точность для сравнения с нулем

    # --- Прямой ход ---
    # Первая строка
    if abs(c[0]) < epsilon:
        st.error(f"Ошибка: Нулевой элемент на главной диагонали c[0]={c[0]}. Метод прогонки неприменим (деление на ноль).")
        return None, None, None
    alpha[0] = -b[0] / c[0]
    beta[0] = d[0] / c[0]

    # Остальные строки прямого хода
    for i in range(1, n):
        denominator = c[i] + a[i-1] * alpha[i-1]
        if abs(denominator) < epsilon:
            st.error(f"Ошибка: Знаменатель ({denominator:.2e}) близок к нулю на шаге i={i} прямого хода. Метод прогонки не может продолжить.")
            return None, None, None

        # Для последней строки alpha не нужен, но для единообразия можно посчитать beta
        beta[i] = (d[i] - a[i-1] * beta[i-1]) / denominator
        if i < n - 1: # alpha нужен только до предпоследнего
             alpha[i] = -b[i] / denominator


    # --- Обратный ход ---
    # Последняя строка
    x[n-1] = beta[n-1] # Т.к. x_n = alpha_{n-1}*x_{n+1} + beta_{n-1}, а x_{n+1} нет (или alpha_{n-1} = 0)

    # Остальные строки обратного хода
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i+1] + beta[i]

    return x, alpha, beta

# --- GUI ---

st.title("2. Метод прогонки (Алгоритм Томаса)")
st.markdown("""
Решение СЛАУ **Ax = d** для **трехдиагональной** матрицы **A** методом прогонки.
Матрица A имеет вид:
Вводятся три диагонали:
- **a**: поддиагональ (элементы `a[0]`...`a[n-2]`)
- **c**: главная диагональ (элементы `c[0]`...`c[n-1]`)
- **b**: наддиагональ (элементы `b[0]`...`b[n-2]`)
и вектор правых частей **d**.

**Условие применимости (достаточное):** Строгое диагональное преобладание.
`|c[i]| > |a[i-1]| + |b[i]|` для всех `i` (считая `a[-1]=0`, `b[n-1]=0`).
Если условие не выполнено, метод может быть неустойчив или невозможен (деление на ноль). Программа проверяет деление на ноль в ходе вычислений.
""")
st.divider()

# --- Функция ввода СЛАУ 5x5 ---
def input_slu_tridiagonal_5x5():
    st.subheader("Ввод ненулевых диагоналей матрицы 5×5 и вектора d")
    n = 5
    # Инициализация массивов для диагоналей и вектора d
    # Используем списки Python для удобства добавления в number_input
    a_input_vals = [0.0] * (n - 1) # Поддиагональ a (a_1 ... a_{n-1} в ур-ях)
    c_input_vals = [0.0] * n      # Главная диагональ c (c_0 ... c_{n-1})
    b_input_vals = [0.0] * (n - 1) # Наддиагональ b (b_0 ... b_{n-2})
    d_input_vals = [0.0] * n      # Вектор d (d_0 ... d_{n-1})

    # Значения по умолчанию из примера
    default_c = [6, 16, -17, 22, -13]
    default_b = [-5, 9, -3, -8] # n-1 элементов
    default_a = [-6, 9, 8, 6]   # n-1 элементов
    default_d = [-58, 161, -114, -90, -55]

    # Создаем колонки для ввода
    cols = st.columns([1, 1, 1, 0.3, 1]) # a, c, b, spacer, d
    headers = ["Поддиагональ (a)", "Главная диагональ (c)", "Наддиагональ (b)", "", "Правая часть (d)"]
    for col, h in zip(cols, headers):
        col.markdown(f"**{h}**")

    # Поля ввода
    for i in range(n):
        row_cols = st.columns([1, 1, 1, 0.3, 1])

        # Поддиагональ 'a' (начинается со второй строки, индекс i-1 в массиве a)
        if i > 0:
            a_input_vals[i-1] = row_cols[0].number_input(f"a[{i-1}]", value=float(default_a[i-1]), key=f"a_{i-1}", label_visibility="collapsed")
        else:
            row_cols[0].markdown("_(нет)_") # Нет поддиагонали в первой строке

        # Главная диагональ 'c'
        c_input_vals[i] = row_cols[1].number_input(f"c[{i}]", value=float(default_c[i]), key=f"c_{i}", label_visibility="collapsed")

        # Наддиагональ 'b' (заканчивается на предпоследней строке, индекс i в массиве b)
        if i < n - 1:
             b_input_vals[i] = row_cols[2].number_input(f"b[{i}]", value=float(default_b[i]), key=f"b_{i}", label_visibility="collapsed")
        else:
             row_cols[2].markdown("_(нет)_") # Нет наддиагонали в последней строке

        # Вектор 'd'
        d_input_vals[i] = row_cols[4].number_input(f"d[{i}]", value=float(default_d[i]), key=f"d_{i}", label_visibility="collapsed")

    # Конвертируем в numpy массивы
    a_np = np.array(a_input_vals)
    c_np = np.array(c_input_vals)
    b_np = np.array(b_input_vals)
    d_np = np.array(d_input_vals)

    return a_np, c_np, b_np, d_np

# --- Ввод данных ---
a_in, c_in, b_in, d_in = input_slu_tridiagonal_5x5()

# --- Отображение полной матрицы (для наглядности) ---
st.subheader("Полная матрица A и вектор d:")
n_matrix = len(c_in)
A_full = np.zeros((n_matrix, n_matrix))
for i in range(n_matrix):
    A_full[i, i] = c_in[i]
    if i > 0:
        A_full[i, i-1] = a_in[i-1]
    if i < n_matrix - 1:
        A_full[i, i+1] = b_in[i]

col_mat, col_vec = st.columns([3, 1])
with col_mat:
    st.markdown("**Матрица A:**")
    st.dataframe(pd.DataFrame(A_full, index=[f"Eq{i}" for i in range(n_matrix)], columns=[f"x{i}" for i in range(n_matrix)]))
with col_vec:
    st.markdown("**Вектор d:**")
    st.dataframe(pd.DataFrame(d_in, index=[f"Eq{i}" for i in range(n_matrix)], columns=["Значение"]))

st.divider()

# --- Кнопка Решить ---
if st.button("Решить СЛАУ", type="primary"):

    st.subheader("Проверка условий и Ход решения:")

    # 1. Проверка достаточного условия (диагональное преобладание)
    is_dominant, condition_message, status, details_message = check_tridiagonal_conditions(a_in, c_in, b_in)

    with st.expander("Детали проверки диагонального преобладания", expanded=False):
         st.text(details_message)

    if status == "success":
        st.success(condition_message)
    else:
        st.warning(condition_message)
        st.info("Продолжаем вычисления, но проверяем на деление на ноль...")


    # 2. Выполнение метода прогонки
    try:
        x_sol, alpha_coeff, beta_coeff = thomas_algorithm(a_in, c_in, b_in, d_in)

        # Проверяем, вернула ли функция решение или None (ошибка)
        if x_sol is not None:
            st.markdown("**Прогоночные коэффициенты (Прямой ход):**")
            coeffs_df = pd.DataFrame({'alpha': alpha_coeff, 'beta': beta_coeff})
            st.dataframe(coeffs_df)

            st.divider()
            st.subheader("Результат:")
            st.markdown("**Вектор решения x (Обратный ход):**")
            solution_df = pd.DataFrame(x_sol.reshape(-1, 1), index=[f'x{i}' for i in range(len(x_sol))], columns=['Значение'])
            st.dataframe(solution_df)
            st.success("Решение успешно найдено методом прогонки.")

        else:
            # Ошибка уже была выведена внутри thomas_algorithm через st.error()
            st.error("Метод прогонки не смог найти решение из-за ошибки (см. сообщение выше).")

    except ValueError as e:
        st.error(f"Ошибка в данных: {e}")
    except Exception as e:
         st.error(f"Непредвиденная ошибка: {e}")
         # import traceback
         # st.error(traceback.format_exc())
