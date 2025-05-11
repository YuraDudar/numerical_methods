import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, expand, lambdify, latex, Poly
import pandas as pd
import traceback


# --- Математические функции для интерполяции ---

def calculate_divided_differences(x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
    """
    Вычисляет разделенные разности для многочлена Ньютона.
    Возвращает массив коэффициентов f[x_0], f[x_0,x_1], ..., f[x_0,...,x_n].
    Эти коэффициенты являются главной диагональю (первой строкой в реализации)
    таблицы разделенных разностей.
    """
    n = len(x_nodes)  # Количество узлов интерполяции

    # Проверка на соответствие размеров входных массивов
    if n != len(y_nodes):
        raise ValueError("Количество узлов X и Y должно совпадать.")
    if n == 0: # Если нет узлов, возвращаем пустой массив
        return np.array([])

    # Инициализируем таблицу разностей значениями y (первый столбец)
    # Копируем y_nodes, чтобы не изменять оригинальный массив
    pyramid = np.zeros([n, n])
    pyramid[:, 0] = y_nodes.copy()

    # Заполняем остальные столбцы таблицы разделенных разностей
    for j in range(1, n):  # j - порядок разности (столбец)
        for i in range(n - j):  # i - строка
            # Формула разделенной разности:
            # f[x_i, ..., x_{i+j}] = (f[x_{i+1}, ..., x_{i+j}] - f[x_i, ..., x_{i+j-1}]) / (x_{i+j} - x_i)
            # В терминах нашей пирамиды:
            # pyramid[i, j] = (pyramid[i+1, j-1] - pyramid[i, j-1]) / (x_nodes[i+j] - x_nodes[i])
            numerator = pyramid[i + 1, j - 1] - pyramid[i, j - 1]
            denominator = x_nodes[i + j] - x_nodes[i]
            if np.isclose(denominator, 0):
                # Эта ошибка должна быть перехвачена ранее проверкой уникальности x_nodes.
                # Однако, если узлы очень близки, но не идентичны, проблема может возникнуть здесь.
                raise ValueError(
                    f"Узлы x_nodes[{i + j}] ({x_nodes[i + j]}) и x_nodes[{i}] ({x_nodes[i]}) "
                    f"слишком близки (разница: {denominator:.2e}), что приводит к делению на ноль "
                    "или близкое к нулю значение при вычислении разделенных разностей."
                )
            pyramid[i, j] = numerator / denominator

    # Возвращаем первую строку таблицы (f[x_0], f[x_0,x_1], f[x_0,x_1,x_2], ...)
    return pyramid[0, :]


def newton_polynomial_value(x_nodes: np.ndarray, div_diffs: np.ndarray, x_eval: float) -> float:
    """
    Вычисляет значение многочлена Ньютона в точке x_eval, используя ранее вычисленные
    разделенные разности (div_diffs).
    Формула многочлена Ньютона:
    P(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + ...
    """
    n = len(div_diffs)  # Количество коэффициентов (равно количеству узлов).
    if n == 0: return 0.0  # Если нет коэффициентов, значение полинома 0.

    # Первый член полинома - это f[x_0] (div_diffs[0]).
    result = div_diffs[0]
    # term_product будет накапливать произведения (x-x_0), (x-x_0)(x-x_1), ...
    term_product = 1.0
    # Итерируем по остальным коэффициентам, начиная с f[x_0,x_1].
    for i in range(1, n):
        term_product *= (x_eval - x_nodes[i - 1])  # Обновляем произведение (x-x_k)
        result += div_diffs[i] * term_product      # Добавляем следующий член полинома
    return result


def newton_polynomial_symbolic(x_nodes: np.ndarray, div_diffs: np.ndarray, x_sym: Symbol = None):
    """
    Возвращает символьное представление многочлена Ньютона.
    """
    if x_sym is None:
        x_sym = Symbol('x')

    n = len(div_diffs)
    if n == 0: return 0  # sympy 0

    # Начинаем строить символьное выражение для полинома.
    # Первый член - f[x_0]. Убеждаемся, что это float для корректной работы с Sympy.
    poly_expr = float(div_diffs[0])
    # term_poly_expr будет накапливать символьные произведения (x_sym - x_0), (x_sym - x_0)(x_sym - x_1), ...
    term_poly_expr = 1 # Начинаем с символьной единицы.
    for i in range(1, n):
        # Умножаем на (x_sym - x_nodes[i-1]). x_nodes[i-1] преобразуется в float.
        term_poly_expr *= (x_sym - float(x_nodes[i - 1]))
        # Добавляем следующий член: div_diffs[i] * term_poly_expr.
        # div_diffs[i] также преобразуется в float.
        poly_expr += float(div_diffs[i]) * term_poly_expr

    # expand() упрощает выражение, раскрывая скобки.
    # Poly(poly_expr, x_sym).as_expr() преобразует выражение в стандартную полиномиальную форму Sympy,
    # что может помочь в упрощении перед expand, хотя expand часто справляется сам.
    return expand(Poly(poly_expr, x_sym).as_expr())


def lagrange_basis_polynomial_symbolic(x_nodes: np.ndarray, j: int, x_sym: Symbol = None):
    """
    Возвращает символьное представление j-го базисного полинома Лагранжа l_j(x).
    Формула l_j(x) = П_{k=0, k!=j}^{n-1} (x - x_k) / (x_j - x_k)
    """
    if x_sym is None:
        x_sym = Symbol('x')

    lj_expr = 1 # Начинаем с символьной единицы для произведения.
    xj_val = float(x_nodes[j]) # Значение x_j, преобразованное в float.

    # Итерируем по всем узлам для построения произведения.
    for k in range(len(x_nodes)):
        if k != j: # Пропускаем случай k=j, т.к. он не входит в произведение.
            xk_val = float(x_nodes[k]) # Значение x_k, преобразованное в float.
            # Добавляем множитель (x_sym - xk_val) / (xj_val - xk_val) к выражению.
            lj_expr *= (x_sym - xk_val) / (xj_val - xk_val)
    # expand() упрощает полученное символьное выражение.
    return expand(lj_expr)


def lagrange_polynomial_symbolic(x_nodes: np.ndarray, y_nodes: np.ndarray, x_sym: Symbol = None):
    """
    Возвращает символьное представление многочлена Лагранжа L(x).
    Формула L(x) = Σ_{j=0}^{n-1} y_j * l_j(x)
    """
    if x_sym is None:
        x_sym = Symbol('x')

    if len(x_nodes) == 0: return 0

    poly_expr = 0 # Начинаем с символьного нуля для суммы.
    # Итерируем по всем узлам, чтобы сложить y_j * l_j(x).
    for j in range(len(y_nodes)): # len(y_nodes) должно быть равно len(x_nodes)
        # Получаем символьное выражение для j-го базисного полинома.
        lj_symbolic = lagrange_basis_polynomial_symbolic(x_nodes, j, x_sym=x_sym)
        yj_val = float(y_nodes[j]) # Значение y_j, преобразованное в float.
        # Добавляем член yj_val * lj_symbolic к общей сумме.
        poly_expr += yj_val * lj_symbolic

    # expand() и Poly().as_expr() для упрощения и стандартизации, как и в Ньютоне.
    return expand(Poly(poly_expr, x_sym).as_expr())


def lagrange_polynomial_value(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: float) -> float:
    """
    Вычисляет значение многочлена Лагранжа в точке x_eval.
    L(x_eval) = Σ_{j=0}^{n-1} y_j * l_j(x_eval)
    """
    n = len(x_nodes)
    if n != len(y_nodes):
        raise ValueError("Количество узлов X и Y должно совпадать.")
    if n == 0: return 0.0

    total_sum = 0.0 # Инициализируем сумму.
    # Итерируем по каждому члену суммы y_j * l_j(x_eval).
    for j in range(n):
        # Вычисляем значение j-го базисного полинома l_j(x_eval)
        basis_poly_val = 1.0 # Инициализируем произведение для l_j.
        for k in range(n):
            if k != j: # Пропускаем k=j.
                numerator = x_eval - x_nodes[k]
                denominator = x_nodes[j] - x_nodes[k]
                # Проверка на деление на ноль.
                if np.isclose(denominator, 0):
                    raise ValueError(
                        f"Узлы x_nodes[{j}] ({x_nodes[j]}) и x_nodes[{k}] ({x_nodes[k]}) "
                        f"совпадают или очень близки (разница: {denominator:.2e}). "
                        "Деление на ноль в базисном полиноме Лагранжа."
                    )
                basis_poly_val *= numerator / denominator
        # Добавляем y_j * l_j(x_eval) к общей сумме.
        total_sum += y_nodes[j] * basis_poly_val
    return total_sum


# --- Streamlit UI ---

def section_3_1():
    st.header("3.1. Интерполяционные многочлены Лагранжа и Ньютона")

    # Определяем функцию для генерации Y и ее имя/LaTeX представление
    func_to_interpolate = np.exp
    func_name_display = "e^x"
    func_latex_display = r"e^x"

    st.sidebar.subheader("Настройки для пункта 3.1")

    # Опции выбора источника данных X_i
    data_source_options = [
        "Предустановка а) X=[-2, -1, 0, 1]",
        "Предустановка б) X=[-2, -1, 0.2, 1]",
        "Пользовательский ввод"
    ]
    # Использование st.session_state для сохранения выбора пользователя
    if 's31_data_source' not in st.session_state:
        st.session_state.s31_data_source = data_source_options[0]

    # Radio для выбора источника X_i
    st.session_state.s31_data_source = st.sidebar.radio(
        "Выберите источник данных для $X_i$:",
        data_source_options,
        index=data_source_options.index(st.session_state.s31_data_source),
        key="s31_data_source_radio"  # Уникальный ключ
    )

    num_points_required = 4  # Фиксированное количество точек i=0, ..., 3

    # Предустановленные значения
    X_i_default_a = np.array([-2.0, -1.0, 0.0, 1.0])
    X_i_default_b = np.array([-2.0, -1.0, 0.2, 1.0])
    X_star_default = -0.5

    # Инициализация или обновление значений в session_state в зависимости от выбора
    # Это помогает сохранить введенные пользователем данные при переключении опций, если они не перезаписываются предустановками
    if 's31_x_nodes_str' not in st.session_state:
        st.session_state.s31_x_nodes_str = ", ".join(map(str, X_i_default_a))
    if 's31_y_nodes_manual_str' not in st.session_state:
        st.session_state.s31_y_nodes_manual_str = ""  # Изначально пусто для ручного ввода
    if 's31_x_star_str' not in st.session_state:
        st.session_state.s31_x_star_str = str(X_star_default)
    if 's31_y_source_choice' not in st.session_state:
        st.session_state.s31_y_source_choice = f"Вычислить из $y=f(x)={func_name_display}$"

    # Установка значений и состояния редактируемости в зависимости от выбора источника X
    if st.session_state.s31_data_source == data_source_options[0]:  # Предустановка а)
        st.session_state.s31_x_nodes_str = ", ".join(map(str, X_i_default_a))
        st.session_state.s31_x_star_str = str(X_star_default)
        is_editable = False
        st.session_state.s31_y_source_choice = f"Вычислить из $y=f(x)={func_name_display}$"  # Y всегда из функции для предустановок
    elif st.session_state.s31_data_source == data_source_options[1]:  # Предустановка б)
        st.session_state.s31_x_nodes_str = ", ".join(map(str, X_i_default_b))
        st.session_state.s31_x_star_str = str(X_star_default)
        is_editable = False
        st.session_state.s31_y_source_choice = f"Вычислить из $y=f(x)={func_name_display}$"
    else:  # Пользовательский ввод
        # Значения s31_x_nodes_str, s31_x_star_str, s31_y_source_choice остаются теми, что были (возможно, изменены пользователем)
        is_editable = True

    st.subheader("Входные данные")
    input_cols = st.columns(2)
    with input_cols[0]:
        st.session_state.s31_x_nodes_str = st.text_input(
            f"Узлы $X_i$ (через запятую, {num_points_required} значения):",
            value=st.session_state.s31_x_nodes_str,
            disabled=not is_editable,
            key="s31_x_nodes_input"
        )
    with input_cols[1]:
        y_source_options_list = [f"Вычислить из $y=f(x)={func_name_display}$", "Ввести вручную"]

        # Убедимся, что текущий выбор y_source_choice валиден
        if st.session_state.s31_y_source_choice not in y_source_options_list:
            st.session_state.s31_y_source_choice = y_source_options_list[0]

        current_y_source_idx = 0 if not is_editable else y_source_options_list.index(
            st.session_state.s31_y_source_choice)

        st.session_state.s31_y_source_choice = st.radio(
            "Источник $Y_i$:",
            y_source_options_list,
            index=current_y_source_idx,
            key="s31_y_source_radio",
            horizontal=True,  # Расположить радиокнопки горизонтально
            disabled=not is_editable  # Отключаем, если не пользовательский ввод X
        )

        # Поле для ручного ввода Y_i, если выбрана соответствующая опция и режим редактируемый
        if st.session_state.s31_y_source_choice == "Ввести вручную" and is_editable:
            st.session_state.s31_y_nodes_manual_str = st.text_input(
                f"Значения $Y_i$ (через запятую, {num_points_required} значения):",
                value=st.session_state.s31_y_nodes_manual_str,
                key="s31_y_nodes_manual_input"
            )

    # Поле для ввода X*
    st.session_state.s31_x_star_str = st.text_input(
        "Точка интерполяции $X^*$:",
        value=st.session_state.s31_x_star_str,
        key="s31_x_star_input"
    )

    # Кнопка для запуска расчетов
    if st.button("📈 Выполнить расчеты для Пункта 3.1", key="s31_run_button"):
        try:
            # --- 1. Парсинг и валидация входных данных ---
            # Получаем X_nodes из сохраненного состояния
            x_nodes_parsed_str = st.session_state.s31_x_nodes_str
            try:
                current_X_nodes = np.array([float(x.strip()) for x in x_nodes_parsed_str.split(',')])
            except ValueError:
                st.error(
                    f"Ошибка парсинга узлов $X_i$. Убедитесь, что это числа, разделенные запятыми (например: -2, -1, 0, 1). Введено: '{x_nodes_parsed_str}'")
                return

            if len(current_X_nodes) != num_points_required:
                st.error(
                    f"Необходимо ввести ровно {num_points_required} значения для $X_i$. Введено: {len(current_X_nodes)}.")
                return

            if len(set(current_X_nodes)) != len(current_X_nodes):
                st.error("Узлы $X_i$ должны быть уникальными. Обнаружены дубликаты.")
                return

            # Сортировка узлов X и соответствующая подготовка Y
            sorted_indices = np.argsort(current_X_nodes)
            current_X_nodes = current_X_nodes[sorted_indices]

            # Получение Y_nodes
            actual_y_source = st.session_state.s31_y_source_choice
            if not is_editable:  # Если выбрана предустановка, Y всегда вычисляются
                actual_y_source = f"Вычислить из $y=f(x)={func_name_display}$"

            if actual_y_source == f"Вычислить из $y=f(x)={func_name_display}$":
                current_Y_nodes = func_to_interpolate(current_X_nodes)
                y_data_source_info = f"вычислены по функции $y={func_name_display}$"
            else:  # Ручной ввод (и is_editable == True)
                y_nodes_manual_parsed_str = st.session_state.s31_y_nodes_manual_str
                if not y_nodes_manual_parsed_str.strip():  # Проверка на пустую строку
                    st.error(
                        f"Пожалуйста, введите значения $Y_i$ для ручного ввода, или выберите опцию 'Вычислить из $y=f(x)={func_name_display}$'.")
                    return
                try:
                    y_nodes_list_parsed = [float(y.strip()) for y in y_nodes_manual_parsed_str.split(',')]
                except ValueError:
                    st.error(
                        f"Ошибка парсинга значений $Y_i$. Убедитесь, что это числа, разделенные запятыми. Введено: '{y_nodes_manual_parsed_str}'")
                    return

                if len(y_nodes_list_parsed) != num_points_required:
                    st.error(
                        f"Необходимо ввести ровно {num_points_required} значения для $Y_i$. Введено: {len(y_nodes_list_parsed)}.")
                    return
                current_Y_nodes = np.array(y_nodes_list_parsed)
                current_Y_nodes = current_Y_nodes[sorted_indices]  # Сортируем Y в соответствии с X
                y_data_source_info = "введены вручную"

            # Парсинг X_star
            try:
                current_X_star = float(st.session_state.s31_x_star_str)
            except ValueError:
                st.error(
                    f"Ошибка парсинга значения $X^*$. Убедитесь, что это число. Введено: '{st.session_state.s31_x_star_str}'")
                return

            st.markdown("---")
            st.subheader("🔍 Результаты интерполяции")

            st.markdown("#### Входные данные для расчетов:")
            df_input = pd.DataFrame({'i': range(num_points_required), 'X_i': current_X_nodes, 'Y_i': current_Y_nodes})
            st.dataframe(df_input.style.format({'X_i': "{:.4f}", 'Y_i': "{:.7f}"}))
            st.write(f"Значения $Y_i$ {y_data_source_info}.")
            st.write(f"Точка для вычисления значения и погрешности $X^* = {current_X_star:.4f}$")

            # Проверка X* на принадлежность отрезку интерполяции
            min_X_node, max_X_node = np.min(current_X_nodes), np.max(current_X_nodes)
            if not (min_X_node <= current_X_star <= max_X_node):
                st.warning(
                    f"⚠️ **Предупреждение:** Точка $X^*={current_X_star:.4f}$ находится вне основного отрезка интерполяции "
                    f"$[{min_X_node:.4f}, {max_X_node:.4f}]$. Выполняется экстраполяция, результаты могут быть менее точными."
                )

            # Символьная переменная для sympy
            x_symbol = Symbol('x')

            # --- 2. Многочлен Лагранжа ---
            st.markdown("#### 1. Многочлен Лагранжа ($L(x)$)")

            with st.expander("Показать базисные полиномы Лагранжа $l_j(x)$"):
                for j_idx in range(num_points_required):
                    lj_sym_expr = lagrange_basis_polynomial_symbolic(current_X_nodes, j_idx, x_sym=x_symbol)
                    st.latex(f"l_{{{j_idx}}}(x) = {latex(lj_sym_expr)}")

            L_poly_symbolic_expr = lagrange_polynomial_symbolic(current_X_nodes, current_Y_nodes, x_sym=x_symbol)
            st.latex(f"L(x) = {latex(L_poly_symbolic_expr)}")

            L_value_at_X_star = lagrange_polynomial_value(current_X_nodes, current_Y_nodes, current_X_star)
            st.markdown(
                f"Значение многочлена Лагранжа в $X^*={current_X_star:.4f}$:  **$L(X^*) = {L_value_at_X_star:.7f}$**")

            # --- 3. Многочлен Ньютона ---
            st.markdown("#### 2. Многочлен Ньютона ($N(x)$)")

            # Вычисление и отображение разделенных разностей
            divided_differences = calculate_divided_differences(current_X_nodes, current_Y_nodes)
            with st.expander("Показать разделенные разности (коэффициенты $a_k = f[x_0, \\dots, x_k]$)"):
                diff_df_data = []
                for i_dd, dd_val in enumerate(divided_differences):
                    nodes_str = ", ".join([f"x_{k}" for k in range(i_dd + 1)])
                    diff_df_data.append({"k": i_dd, f"a_k = f[{nodes_str}]": dd_val})
                diff_df = pd.DataFrame(diff_df_data)
                st.dataframe(diff_df.style.format({diff_df.columns[1]: "{:.7f}"}))

            N_poly_symbolic_expr = newton_polynomial_symbolic(current_X_nodes, divided_differences, x_sym=x_symbol)
            st.latex(f"N(x) = {latex(N_poly_symbolic_expr)}")

            N_value_at_X_star = newton_polynomial_value(current_X_nodes, divided_differences, current_X_star)
            st.markdown(
                f"Значение многочлена Ньютона в $X^*={current_X_star:.4f}$: **$N(X^*) = {N_value_at_X_star:.7f}$**")

            # --- 4. Сравнение и Погрешность ---
            st.markdown("#### 3. Сравнение полиномов и вычисление погрешности")
            diff_L_N_at_X_star = abs(L_value_at_X_star - N_value_at_X_star)
            st.write(
                f"Теоретически, $L(x)$ и $N(x)$ представляют один и тот же единственный интерполяционный многочлен. "
                f"Разница их вычисленных значений в $X^*$: $|L(X^*) - N(X^*)| = {diff_L_N_at_X_star:.2e}$."
            )
            if diff_L_N_at_X_star > 1e-9:  # Допустимая погрешность для численных методов
                st.warning("Разница между значениями $L(X^*)$ и $N(X^*)$ больше ожидаемой.")

            # Вычисление погрешности относительно истинной функции y=e^x
            # Это имеет смысл только если Y были сгенерированы из этой функции
            if actual_y_source == f"Вычислить из $y=f(x)={func_name_display}$":
                f_true_at_X_star = func_to_interpolate(current_X_star)
                st.write(
                    f"Истинное значение функции $f(X^*) = {func_name_display}({current_X_star:.4f}) = {f_true_at_X_star:.7f}$")

                abs_error = abs(f_true_at_X_star - L_value_at_X_star)  # Используем L(X*), т.к. L(X*) ~ N(X*)
                rel_error_percent = (abs_error / abs(f_true_at_X_star) * 100) if not np.isclose(f_true_at_X_star,
                                                                                                0) else (
                    0 if np.isclose(abs_error, 0) else float('inf'))

                st.success(f"Абсолютная погрешность интерполяции: $|f(X^*) - P(X^*)| = {abs_error:.7f}$")
                st.success(
                    f"Относительная погрешность интерполяции: ${rel_error_percent:.5f}\\%$ (если $f(X^*) \\neq 0$)")
            else:
                st.info(
                    f"Поскольку значения $Y_i$ были введены вручную, точное значение 'истинной' функции $f(X^*)$ неизвестно, и погрешность интерполяции относительно нее не вычисляется.")

            # --- 5. Проверка прохождения через узлы ---
            st.markdown("#### 4. Проверка прохождения многочлена через узловые точки")
            check_points_data = {"$X_i$": current_X_nodes, "$Y_i$ (исходные)": current_Y_nodes}

            # Используем значения, вычисленные численными методами, а не символьным полиномом (быстрее и менее подвержено ошибкам lambdify)
            P_values_at_Xi_L = np.array(
                [lagrange_polynomial_value(current_X_nodes, current_Y_nodes, xi) for xi in current_X_nodes])
            P_values_at_Xi_N = np.array(
                [newton_polynomial_value(current_X_nodes, divided_differences, xi) for xi in current_X_nodes])

            check_points_data["$L(X_i)$"] = P_values_at_Xi_L
            check_points_data["$N(X_i)$"] = P_values_at_Xi_N
            check_points_data["$|Y_i - L(X_i)|$"] = np.abs(current_Y_nodes - P_values_at_Xi_L)

            check_df = pd.DataFrame(check_points_data)
            # Форматирование для st.dataframe
            check_df_format_dict = {
                '$Y_i$ (исходные)': "{:.7f}",
                '$L(X_i)$': "{:.7f}",
                '$N(X_i)$': "{:.7f}",
                '$|Y_i - L(X_i)|$': "{:.2e}"
            }
            st.dataframe(check_df.style.format(check_df_format_dict))

            # Проверка np.allclose для точности
            if np.allclose(current_Y_nodes, P_values_at_Xi_L, atol=1e-9) and \
                    np.allclose(current_Y_nodes, P_values_at_Xi_N, atol=1e-9):
                st.success(
                    "✅ Проверка пройдена: Оба многочлена (Лагранжа и Ньютона) проходят через все заданные узловые точки $(X_i, Y_i)$ с высокой точностью.")
            else:
                st.error(
                    "❌ Ошибка проверки: Один или оба многочлена не проходят через все узловые точки с достаточной точностью. Проверьте таблицу разниц выше.")


            # --- 6. График ---
            st.markdown("#### 5. Графическая иллюстрация")
            plot_fig = go.Figure()

            # Определение диапазона для построения графика
            plot_x_min = min(min_X_node, current_X_star) - 0.5 * abs(
                max_X_node - min_X_node) if max_X_node != min_X_node else min_X_node - 1
            plot_x_max = max(max_X_node, current_X_star) + 0.5 * abs(
                max_X_node - min_X_node) if max_X_node != min_X_node else max_X_node + 1
            if plot_x_min == plot_x_max:  # Если все узлы и X* совпадают (маловероятно из-за проверки уникальности узлов)
                plot_x_min -= 1
                plot_x_max += 1

            x_dense_for_plot = np.linspace(plot_x_min, plot_x_max, 400)

            # Истинная функция (если Y вычислялись по ней)
            if actual_y_source == f"Вычислить из $y=f(x)={func_name_display}$":
                plot_fig.add_trace(go.Scatter(
                    x=x_dense_for_plot,
                    y=func_to_interpolate(x_dense_for_plot),
                    mode='lines',
                    name=f"Истинная функция $f(x)={func_name_display}$",
                    line=dict(dash='dash', color='green', width=2)
                ))

            # Интерполяционный многочлен (используем символьное представление L(x) для построения)
            try:
                # Преобразуем символьное выражение в вычисляемую функцию
                # Используем L_poly_symbolic_expr, так как L(x) и N(x) должны быть идентичны
                if isinstance(L_poly_symbolic_expr, (int, float, complex)):  # Если полином - константа
                    y_poly_dense_plot = np.full_like(x_dense_for_plot, float(L_poly_symbolic_expr))
                else:
                    # Добавляем 'numpy' и обработку Heaviside для большей совместимости lambdify
                    poly_callable_func = lambdify(x_symbol, L_poly_symbolic_expr,
                                                  modules=['numpy', {'Heaviside': lambda x_h: np.heaviside(x_h, 0.5)}])
                    y_poly_dense_plot = poly_callable_func(x_dense_for_plot)
            except Exception as e_lambdify:
                st.warning(
                    f"Не удалось создать функцию из символьного полинома для графика ({e_lambdify}). "
                    "Используется поточечное вычисление (может быть медленнее или менее точно для отображения сложных полиномов)."
                )
                # Откат к поточечному вычислению, если lambdify не сработало
                y_poly_dense_plot = np.array(
                    [lagrange_polynomial_value(current_X_nodes, current_Y_nodes, x_val) for x_val in x_dense_for_plot])

            plot_fig.add_trace(go.Scatter(
                x=x_dense_for_plot,
                y=y_poly_dense_plot,
                mode='lines',
                name="Интерполяционный многочлен $P(x)$",
                line=dict(color='rgba(255,0,0,0.9)', width=2.5)  # Ярко-красный
            ))

            # Узловые точки
            plot_fig.add_trace(go.Scatter(
                x=current_X_nodes,
                y=current_Y_nodes,
                mode='markers',
                name='Узловые точки $(X_i, Y_i)$',
                marker=dict(size=10, color='blue', symbol='circle', line=dict(width=1, color='DarkSlateGrey'))
            ))

            # Точка (X*, P(X*))
            plot_fig.add_trace(go.Scatter(
                x=[current_X_star],
                y=[L_value_at_X_star],  # Используем L(X*)
                mode='markers', name=f'$P(X^*={current_X_star:.2f})$',
                marker=dict(size=12, color='red', symbol='x-dot', line=dict(width=2, color='DarkSlateGrey'))
            ))

            # Точка (X*, f(X*)) истинная, если Y вычислялись
            if actual_y_source == f"Вычислить из $y=f(x)={func_name_display}$":
                plot_fig.add_trace(go.Scatter(
                    x=[current_X_star],
                    y=[f_true_at_X_star],
                    mode='markers', name=f'$f(X^*={current_X_star:.2f})$ (истина)',
                    marker=dict(size=12, color='green', symbol='cross-dot', line=dict(width=2, color='DarkSlateGrey'))
                ))

            plot_fig.update_layout(
                title=f"Интерполяция функции (для $f(x)={func_latex_display}$ или табличных данных)",
                xaxis_title=" ось X",
                yaxis_title=" ось Y",
                legend_title_text="<b>Обозначения на графике</b>",
                hovermode="x unified",  # Улучшает отображение подсказок
                margin=dict(l=20, r=20, t=50, b=20)  # Компактные отступы
            )
            st.plotly_chart(plot_fig, use_container_width=True)

        except ValueError as ve:  # Ошибки, связанные со значениями (например, текст вместо числа)
            st.error(f"🚫 Ошибка значения: {ve}")
            # st.error(traceback.format_exc()) # Раскомментировать для отладки
        except Exception as e:  # Другие непредвиденные ошибки
            st.error(f"💥 Произошла непредвиденная ошибка: {e}")
            st.error("Полная информация об ошибке для разработчика:")
            st.code(traceback.format_exc())


# --- Структура основного приложения ---

# Кастомный CSS для улучшения вида радиокнопок (опционально)
st.markdown(r"""
<style>
.stRadio[role=radiogroup] {
    flex-direction: row; /* Располагает кнопки в ряд */
    gap: 15px; /* Промежуток между кнопками */
}
.stRadio[role=radiogroup] > label {
    margin-right: 0; /* Убирает стандартный отступ справа от Streamlit */
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Лабораторная работа по численным методам")
st.markdown("---")
section_3_1()
