import streamlit as st
import numpy as np
import pandas as pd

# --- Функции для LU-разложения ---
def lu_decomposition_pivot(A):
    """
    Выполняет LU-разложение матрицы A с частичным выбором главного элемента (по строке).
    PA = LU
    Возвращает: L, U, P, swaps (количество перестановок строк)
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    L = np.identity(n)
    U = A.astype(float).copy()
    P = np.identity(n)
    swaps = 0
    epsilon = 1e-10

    for k in range(n - 1):
        # --- Выбор главного элемента ---
        pivot_row = k + np.argmax(np.abs(U[k:, k])) # Индекс строки с макс. элементом в столбце k (относительно среза k:)
        if pivot_row != k:
            # Меняем строки в U
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            # Меняем строки в L (только уже вычисленную часть!)
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
             # Меняем строки в P
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            swaps += 1

        # --- Проверка на сингулярность ---
        if abs(U[k, k]) < epsilon:
            # Если главный элемент близок к нулю после возможной перестановки,
            # матрица сингулярна или близка к ней.
             # Можно было бы продолжить, но это приведет к большим ошибкам.
            st.warning(f"Предупреждение: Главный элемент U[{k},{k}] ({U[k,k]:.2e}) близок к нулю. "
                       f"Матрица может быть сингулярной. Точность решения может быть низкой.")
            # raise ValueError(f"Матрица сингулярна или близка к ней (нулевой или близкий к нулю ведущий элемент U[{k},{k}]).")


        # --- Вычисление элементов L и U ---
        for i in range(k + 1, n):
            if U[k,k] == 0: # Дополнительная проверка на всякий случай
                 st.error(f"Ошибка деления на ноль при вычислении L[{i},{k}]. Матрица сингулярна.")
                 return None, None, None, -1 # Сигнал ошибки

            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            # U[i, k] = 0 # Теоретически уже ноль из-за вычитания, но можно для надежности

    # Проверка последнего диагонального элемента U
    if abs(U[n-1, n-1]) < epsilon:
         st.warning(f"Предупреждение: Последний диагональный элемент U[{n-1},{n-1}] ({U[n-1,n-1]:.2e}) близок к нулю. "
                       f"Матрица может быть сингулярной.")

    # Обнулим значения под диагональю U, которые могли остаться из-за погрешностей float
    for i in range(n):
        for j in range(i):
            U[i,j] = 0.0
    print(L)
    print(U)
    return L, U, P, swaps

def forward_substitution(L, Pb):
    """Решает систему Ly = Pb (прямой ход)"""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        # L[i, i] всегда 1 для нашего LU
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
    return y

def backward_substitution(U, y):
    """Решает систему Ux = y (обратный ход)"""
    n = U.shape[0]
    x = np.zeros(n)
    epsilon = 1e-10 # Малое число для сравнения с нулем

    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < epsilon:
            raise ValueError(f"Деление на близкий к нулю диагональный элемент U[{i},{i}]={U[i, i]}. Система может не иметь единственного решения.")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def calculate_determinant(U, swaps):
    """Вычисляет определитель на основе U и числа перестановок"""
    # Используем тот факт, что det(A) = det(P^-1LU) = ±det(U).
    det = np.prod(np.diag(U)) * ((-1) ** swaps)
    return det

def calculate_inverse(L, U, P):
    """Вычисляет обратную матрицу A^-1, решая AX = I (или LUX = P)"""
    n = L.shape[0]
    I = np.identity(n)
    A_inv = np.zeros((n, n))

    # Решаем n систем: Ax_i = e_i, где e_i - столбцы единичной матрицы
    # Это эквивалентно LUx_i = Pe_i
    for i in range(n):
        Pe_i = np.dot(P, I[:, i])  # Применяем перестановку к столбцу единичной матрицы
        y = forward_substitution(L, Pe_i)
        x_i = backward_substitution(U, y)
        A_inv[:, i] = x_i # Записываем найденное решение как i-й столбец обратной матрицы

    return A_inv


# --- GUI ---
st.title("1. LU-разложение с выбором главного элемента")
st.markdown("""
Решение СЛАУ **Ax = b** методом LU-разложения с частичным выбором главного элемента (по строке).
Алгоритм факторизует матрицу A с учетом перестановок строк P так, что **PA = LU**, где:
- **P**: Матрица перестановок.
- **L**: Нижняя треугольная матрица с единицами на диагонали.
- **U**: Верхняя треугольная матрица.

Затем решаются две системы с треугольными матрицами:
1.  **Ly = Pb** (прямая подстановка)
2.  **Ux = y** (обратная подстановка)

Также вычисляется определитель `det(A) = (-1)^swaps * det(U)` и обратная матрица `A⁻¹`.
""")

st.divider()

# --- Функция ввода СЛАУ 4x4 ---
def input_slu_4x4():
    st.subheader("Ввод коэффициентов СЛАУ 4×4 (Ax = b)")
    A = np.zeros((4, 4), dtype=float)
    b = np.zeros(4, dtype=float)

    # Значения по умолчанию из примера
    default_A = [
        [1, 2, -1, -7],
        [8, 0, -9, -3],
        [2, -3, 7, 1],
        [1, -5, -6, 8]
    ]
    default_b = [-23, 39, -7, 30]

    cols_header = st.columns([1] * 4 + [0.2, 1]) # Колонки для A и b + разделитель
    headers = [f"x{j+1}" for j in range(4)] + ["", "="]
    for col, header in zip(cols_header, headers):
        col.markdown(f"**{header}**")

    # Используем контейнеры для лучшего выравнивания
    input_container = st.container()
    with input_container:
        for i in range(4):
            cols = st.columns([1] * 4 + [0.2, 1]) # 4 для A, 1 пустая, 1 для b
            for j in range(4):
                A[i, j] = cols[j].number_input(
                    f"A[{i},{j}]",
                    value=float(default_A[i][j]),
                    label_visibility="collapsed", # Скрываем стандартный label
                    key=f"A_{i}_{j}"
                )
            cols[4].write(" = ") # Визуальный разделитель "="
            b[i] = cols[5].number_input(
                f"b[{i}]",
                value=float(default_b[i]),
                 label_visibility="collapsed",
                key=f"b_{i}"
            )
    return A, b

# --- Ввод данных ---
A_input, b_input = input_slu_4x4()

st.divider()

# --- Кнопка Решить ---
if st.button("Решить СЛАУ", type="primary"):

    st.subheader("Исходная система:")
    # Отображаем исходную матрицу A и вектор b красиво
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Матрица A:**")
        st.dataframe(pd.DataFrame(A_input, index=[f"Eq{i+1}" for i in range(4)], columns=[f"x{j+1}" for j in range(4)]))
    with col2:
        st.markdown("**Вектор b:**")
        st.dataframe(pd.DataFrame(b_input, index=[f"Eq{i+1}" for i in range(4)], columns=["Значение"]))


    st.divider()
    st.subheader("Ход решения:")

    try:
        # 1. LU-разложение
        L, U, P, swaps = lu_decomposition_pivot(A_input)

        if L is None: # Проверка на ошибку деления на ноль из функции
             st.error("Разложение не удалось из-за деления на ноль. Матрица сингулярна.")

        elif swaps == -1: # Другой сигнал ошибки
             st.error("Произошла ошибка при LU-разложении.")

        else:
            st.markdown("**1. LU-разложение (PA = LU)**")
            col_l, col_u, col_p = st.columns(3)
            with col_l:
                st.markdown("**Матрица L:**")
                st.dataframe(pd.DataFrame(L))
            with col_u:
                st.markdown("**Матрица U:**")
                st.dataframe(pd.DataFrame(U))
            with col_p:
                st.markdown("**Матрица P:**")
                st.dataframe(pd.DataFrame(P))
                st.write(f"(Количество перестановок строк: {swaps})")

            # Проверка PA == LU (для отладки и демонстрации)
            PA = np.dot(P, A_input)
            LU = np.dot(L, U)
            st.write("Проверка PA == LU:", np.allclose(PA, LU))


            # 2. Решение Ly = Pb
            st.markdown("**2. Решение Ly = Pb (прямая подстановка)**")
            Pb = np.dot(P, b_input) # Применяем перестановки к вектору b
            y = forward_substitution(L, Pb)
            st.markdown(f"**Вектор Pb** (b с перестановками):")
            st.dataframe(pd.DataFrame(Pb, columns=['Pb']))
            st.markdown(f"**Промежуточный вектор y:**")
            st.dataframe(pd.DataFrame(y, columns=['y']))


            # 3. Решение Ux = y
            st.markdown("**3. Решение Ux = y (обратная подстановка)**")
            x = backward_substitution(U, y)


            st.divider()
            st.subheader("Результаты:")

            # Отображение решения x
            st.markdown("**Вектор решения x:**")
            st.dataframe(pd.DataFrame(x.reshape(-1, 1), index=[f'x{i+1}' for i in range(len(x))], columns=['Значение']))
            # Или так:
            # st.write("x =", x)


            # 4. Вычисление определителя
            st.markdown("**Определитель det(A):**")
            determinant = calculate_determinant(U, swaps)
            if abs(determinant) < 1e-10:
                 st.warning(f"Определитель близок к нулю ({determinant:.2e}). Матрица может быть плохо обусловлена или сингулярна.")
            st.write(f"`det(A) = {determinant:.6f}`")

            # 5. Вычисление обратной матрицы
            st.markdown("**Обратная матрица A⁻¹:**")
            if abs(determinant) < 1e-10:
                 st.error("Определитель близок к нулю, обратную матрицу вычислить невозможно (или она будет очень неточной).")
            else:
                try:
                    A_inv = calculate_inverse(L, U, P)
                    st.dataframe(pd.DataFrame(A_inv))

                    # Проверка A * A_inv = I (для отладки)
                    I_check = np.dot(A_input, A_inv)
                    st.write("Проверка A * A⁻¹ ≈ I :", np.allclose(I_check, np.identity(A_input.shape[0]), atol=1e-8))

                except ValueError as e_inv:
                     st.error(f"Ошибка при вычислении обратной матрицы: {e_inv}")
                except Exception as e_gen:
                     st.error(f"Непредвиденная ошибка при вычислении обратной матрицы: {e_gen}")


    except ValueError as e:
        st.error(f"Ошибка: {e}")
    except np.linalg.LinAlgError as e:
        # Хотя мы не используем np.linalg напрямую для решения,
        # ошибки могут возникнуть при операциях numpy
        st.error(f"Ошибка линейной алгебры (возможно, сингулярная матрица): {e}")
    except Exception as e:
         st.error(f"Непредвиденная ошибка: {e}")
         # import traceback
         # st.error(traceback.format_exc())
