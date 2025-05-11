import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, expand, lambdify, latex, Piecewise, And, N
import pandas as pd
import traceback


# --- Математические функции для кубического сплайна ---

class CubicSpline:
    def __init__(self, x_nodes: np.ndarray, y_nodes: np.ndarray, natural_spline: bool = True):
        # --- 1. Валидация и подготовка входных данных ---
        if len(x_nodes) != len(y_nodes):
            raise ValueError("Количество узлов X и Y должно совпадать.")
        if len(x_nodes) < 2:  # Для одного сегмента сплайна (2 точки)
            raise ValueError("Для построения сплайна необходимо как минимум 2 узла.")

        # Сортировка узлов X и соответствующих Y по возрастанию X.
        # Это важно для корректного вычисления шагов h_i и определения сегментов.
        sorted_indices = np.argsort(x_nodes)
        self.x_nodes = np.array(x_nodes)[sorted_indices]
        self.y_nodes = np.array(y_nodes)[sorted_indices]

        if len(set(self.x_nodes)) != len(self.x_nodes):
            raise ValueError("Узлы X_i должны быть уникальными.")

        self.n = len(self.x_nodes) - 1  # Количество интервалов (сегментов сплайна). Например, для 5 узлов будет 4 интервала.
        self.h = np.diff(self.x_nodes)  # Вычисляет разности между соседними элементами x_nodes, т.е. шаги h_i = x_{i+1} - x_i.

        if np.any(self.h <= 0):  # Эта проверка покрывается уникальностью и сортировкой, но на всякий случай
            raise ValueError("Узлы x_nodes должны быть строго возрастающими.")


        # --- 2. Вычисление коэффициентов c_i (вторые производные в узлах) ---
        # S_i''(x_i) = c_i (в некоторых источниках c_i/2). Здесь мы следуем одной из распространенных нотаций,
        # где S_i(x) = a_i + b_i(x-x_i) + (c_i/2)(x-x_i)^2 + (d_i/6)(x-x_i)^3.
        # В этом случае c_i - это именно вторая производная.
        # Формула для системы уравнений относительно c_i для естественного сплайна:
        # h_{i-1}c_{i-1} + 2(h_{i-1}+h_i)c_i + h_i c_{i+1} = 6 * [ (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1} ]
        # Краевые условия для естественного сплайна: c_0 = 0, c_n = 0 (где n - индекс последнего узла, т.е. n = self.n)

        if self.n == 0:  # Случай двух точек x0, x1 (n=1 интервал)
            # Для естественного сплайна c0=0, c1=0. Полином становится линейным.
            # Если не естественный, нужны другие краевые условия.
            # Здесь мы всегда используем естественный сплайн
            self.c = np.zeros(2)  # c_0 = 0, c_1 = 0
        elif self.n == 1:  # Случай трех точек x0, x1, x2 (n=2 интервала)
            # Система для c1: 2(h0+h1)c1 = RHS, c0=0, c2=0
            self.c = np.zeros(self.n + 1)  # c_0, c_1, ..., c_n
            if natural_spline:  # c_0 = 0, c_n = 0
                # Уравнение для c_1 (когда n=2, т.е. 3 точки x0, x1, x2):
                # h_0 c_0 + 2(h_0+h_1)c_1 + h_1 c_2 = 6 * (...)
                # Если c_0=0, c_2=0: 2(h_0+h_1)c_1 = 6 * (...)
                if self.n + 1 > 2:  # Только если есть внутренние точки (3+ узла)
                    alpha = 6 * ((self.y_nodes[2] - self.y_nodes[1]) / self.h[1] - \
                                 (self.y_nodes[1] - self.y_nodes[0]) / self.h[0])
                    if not np.isclose(2 * (self.h[0] + self.h[1]), 0):
                        self.c[1] = alpha / (2 * (self.h[0] + self.h[1]))
                    else:  # h0 и h1 очень малы, но не ноль.
                        self.c[1] = 0  # Или другое поведение по умолчанию
                # c[0] и c[n] (c[2] в данном случае) уже 0
            else:
                raise NotImplementedError(
                    "Только естественный сплайн реализован для 3 точек в этом упрощенном конструкторе.")
        else:  # Общий случай для n >= 2 интервалов (3+ точек)
            # Формирование системы для c_i (вторые производные)
            # h_{i-1}c_{i-1} + 2(h_{i-1}+h_i)c_i + h_i c_{i+1} = rhs_i
            # для i = 1, ..., n-1. Краевые условия: c_0 = 0, c_n = 0.

            A = np.zeros((self.n - 1, self.n - 1))  # Система для c_1, ..., c_{n-1}
            b_rhs = np.zeros(self.n - 1)

            for i in range(self.n - 1):  # i от 0 до n-2, соответствует индексам c_1 ... c_{n-1}
                # Уравнение для c_{i+1}
                # h_i * c_i + 2*(h_i+h_{i+1})*c_{i+1} + h_{i+1}*c_{i+2} = 6*(...)
                # c_i в этом контексте это c_prev, c_{i+1} это c_curr, c_{i+2} это c_next

                # Главная диагональ
                A[i, i] = 2 * (self.h[i] + self.h[i + 1])

                # Правая часть
                term1 = (self.y_nodes[i + 2] - self.y_nodes[i + 1]) / self.h[i + 1]
                term2 = (self.y_nodes[i + 1] - self.y_nodes[i]) / self.h[i]
                b_rhs[i] = 6 * (term1 - term2)

                if i > 0:  # Нижняя диагональ
                    A[i, i - 1] = self.h[i]
                if i < self.n - 2:  # Верхняя диагональ
                    A[i, i + 1] = self.h[i + 1]

            # Решение системы A * c_internal = b_rhs
            try:
                c_internal = np.linalg.solve(A, b_rhs)
            except np.linalg.LinAlgError:
                raise ValueError("Не удалось решить систему для коэффициентов сплайна. "
                                 "Проверьте, что узлы X_i различны и их не слишком мало. "
                                 "Матрица может быть вырожденной или плохо обусловленной.")

            self.c = np.zeros(self.n + 1)
            self.c[1:-1] = c_internal  # c_1, ..., c_{n-1}
            # c_0 = 0, c_n = 0 (естественный сплайн) по умолчанию в np.zeros

        # --- 3. Вычисление остальных коэффициентов сплайна (a_i, b_i, d_i) ---
        # Для каждого i-го сегмента (от 0 до self.n-1):
        # S_i(x) = a_i + b_i(x-x_i) + (c_i/2)(x-x_i)^2 + (d_i/6)(x-x_i)^3
        # где c_i = S''(x_i) - это значения, которые мы нашли.
        # a_i = y_i
        # d_i = (c_{i+1} - c_i) / h_i  (здесь в коде d_i = (c_{i+1} - c_i) / h_i, а в формуле S_i(x) используется d_i/6)
        # b_i = (y_{i+1}-y_i)/h_i - h_i/6 * (2c_i + c_{i+1})
        #   (в коде h_i * (2 * self.c[i] + self.c[i+1]) / 6)

        self.a = self.y_nodes[:-1].copy()  # a_i = y_i

        self.d = np.zeros(self.n)
        self.b_coeffs = np.zeros(self.n)  # b_i в формуле для S_i(x)

        for i in range(self.n):  # Для каждого интервала i от 0 до n-1
            if np.isclose(self.h[i], 0):  # Должно быть отловлено раньше
                self.d[i] = 0
            else:
                self.d[i] = (self.c[i + 1] - self.c[i]) / self.h[i]

            term_y_h = (self.y_nodes[i + 1] - self.y_nodes[i]) / self.h[i]
            term_c_h = self.h[i] * (2 * self.c[i] + self.c[i + 1]) / 6
            self.b_coeffs[i] = term_y_h - term_c_h

        # Сохраняем все коэффициенты (a,b,c,d) для вывода
        self.coeffs_table = pd.DataFrame({
            'i (сегмент)': range(self.n),
            '$x_i$': self.x_nodes[:-1],
            '$x_{i+1}$': self.x_nodes[1:],
            '$a_i=y_i$': self.a,
            '$b_i$': self.b_coeffs,
            '$c_i=S\'\'(x_i)$': self.c[:-1],  # c_i для начала сегмента
            # '$c_{i+1}=S\'\'(x_{i+1})$': self.c[1:], # c_{i+1} для конца сегмента (можно добавить)
            '$d_i$': self.d
        })

    def evaluate(self, x_eval: float) -> float:
        """Вычисляет значение сплайна в точке x_eval."""

        # Находим сегмент, в который попадает x_eval
        # Если x_eval < x_nodes[0] или x_eval > x_nodes[-1], это экстраполяция

        if x_eval < self.x_nodes[0]:  # Экстраполяция влево
            segment_idx = 0
        elif x_eval > self.x_nodes[-1]:  # Экстраполяция вправо
            segment_idx = self.n - 1
        else:  # Интерполяция
            # np.searchsorted находит индекс, куда можно вставить x_eval, чтобы сохранить порядок
            # Если x_eval совпадает с x_node[j], вернет j.
            # Нам нужен индекс i такой, что x_nodes[i] <= x_eval <= x_nodes[i+1]
            segment_idx = np.searchsorted(self.x_nodes, x_eval, side='right') - 1
            # Если x_eval == x_nodes[0], segment_idx будет -1, исправим на 0.
            segment_idx = max(0, segment_idx)
            # Если x_eval == x_nodes[n] и side='right', то segment_idx будет n-1, что корректно для последнего сегмента.

        # Защита, если x_eval очень близок к x_nodes[0] и searchsorted вернул -1 (маловероятно с max(0, ...))
        if segment_idx < 0: segment_idx = 0
        # Защита, если x_eval очень близок к x_nodes[n] и searchsorted вернул индекс n
        if segment_idx >= self.n: segment_idx = self.n - 1

        # Коэффициенты для данного сегмента
        ai = self.a[segment_idx]
        bi = self.b_coeffs[segment_idx]
        ci = self.c[segment_idx]  # Это c_i, не c_{i+1}
        di = self.d[segment_idx]

        xi = self.x_nodes[segment_idx]
        delta_x = x_eval - xi

        # S_i(x) = a_i + b_i(x-x_i) + c_i/2 * (x-x_i)^2 + d_i/6 * (x-x_i)^3
        value = ai + bi * delta_x + (ci / 2.0) * delta_x ** 2 + (di / 6.0) * delta_x ** 3
        return value

    def get_symbolic_spline(self, x_sym: Symbol = None):
        """Возвращает символьное представление сплайна в виде Piecewise."""
        if x_sym is None:
            x_sym = Symbol('x')

        piecewise_args = []
        for i in range(self.n):
            ai = N(self.a[i], 10)  # N для округления и предотвращения очень длинных чисел в sympy
            bi = N(self.b_coeffs[i], 10)
            ci = N(self.c[i], 10)
            di = N(self.d[i], 10)
            xi = N(self.x_nodes[i], 10)
            xi_plus_1 = N(self.x_nodes[i + 1], 10)

            delta_x_sym = (x_sym - xi)

            # $S_i(x) = a_i + b_i(x-x_i) + c_i/2 * (x-x_i)^2 + d_i/6 * (x-x_i)^3
            poly_expr = ai + bi * delta_x_sym + (ci / 2) * delta_x_sym ** 2 + (di / 6) * delta_x_sym ** 3
            poly_expr_expanded = expand(poly_expr)

            if i < self.n - 1:
                condition = And(x_sym >= xi, x_sym < xi_plus_1)
            else:  # Последний сегмент, включаем правую границу
                condition = And(x_sym >= xi, x_sym <= xi_plus_1)

            piecewise_args.append((poly_expr_expanded, condition))

        # Для значений вне диапазона определения сплайна можно вернуть nan или экстраполировать
        # sympy.Piecewise по умолчанию возвращает 0 для x вне всех условий.
        # Мы хотим, чтобы он экстраполировал, используя крайние полиномы.

        # Создаем Piecewise. Если x_eval вне диапазона узлов, Piecewise вернет 0.
        # Чтобы обеспечить экстраполяцию, нужно добавить условия для x < x_0 и x > x_n
        # или просто использовать evaluate, которое уже это делает.
        # Для символьного представления ограничимся определением на [x_0, x_n].

        if not piecewise_args:  # Например, если всего 2 точки, n=1, то будет один полином.
            return "Не удалось построить символьное представление (мало сегментов)."

        # Добавим "else" условие для значений вне [x0, xn] (например, NaN или 0)
        # piecewise_args.append((sympy.nan, True)) # или (0, True)

        # Если нужно, чтобы Piecewise сам экстраполировал, нужно определить первый и последний полиномы
        # так, чтобы их условия покрывали (-oo, x1) и (xn-1, oo).
        # Однако, стандартно Piecewise определяет сплайн только на [x0, xn].

        spline_expression = Piecewise(*piecewise_args)
        return spline_expression


# --- Streamlit UI для пункта 3.2 ---
def section_3_2():
    st.header("3.2. Кубический сплайн")

    # Определяем функцию для генерации Y (если нужно) и ее имя
    default_func_to_interpolate = np.exp
    default_func_name_display = "e^x"

    st.sidebar.subheader("Настройки для пункта 3.2")

    # Предустановленные значения
    X_i_default = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    # Y_i_default = np.array([0.13534, 0.36788, 1.0, 2.7183, 7.3891]) # Соответствует e^x
    Y_i_default = default_func_to_interpolate(X_i_default)  # Более точно
    X_star_default = -0.5
    num_points_default = 5

    data_source_options_s32 = [
        f"Предустановка ({num_points_default} узлов, $f(x)={default_func_name_display}$)",
        "Пользовательский ввод"
    ]

    if 's32_data_source' not in st.session_state:
        st.session_state.s32_data_source = data_source_options_s32[0]

    st.session_state.s32_data_source = st.sidebar.radio(
        "Выберите источник данных для сплайна:",
        data_source_options_s32,
        index=data_source_options_s32.index(st.session_state.s32_data_source),
        key="s32_data_source_radio"
    )

    # Инициализация значений в session_state
    if 's32_x_nodes_str' not in st.session_state:
        st.session_state.s32_x_nodes_str = ", ".join(map(str, X_i_default))
    if 's32_y_nodes_str' not in st.session_state:
        st.session_state.s32_y_nodes_str = ", ".join(map(lambda x: f"{x:.5f}", Y_i_default))  # Форматируем для красоты
    if 's32_x_star_str' not in st.session_state:
        st.session_state.s32_x_star_str = str(X_star_default)
    if 's32_y_source_choice' not in st.session_state:  # Для пользовательского ввода Y
        st.session_state.s32_y_source_choice = f"Вычислить из $y=f(x)={default_func_name_display}$"

    # Установка значений и состояния редактируемости
    if st.session_state.s32_data_source == data_source_options_s32[0]:  # Предустановка
        st.session_state.s32_x_nodes_str = ", ".join(map(str, X_i_default))
        st.session_state.s32_y_nodes_str = ", ".join(
            map(lambda x: f"{x:.5f}", default_func_to_interpolate(X_i_default)))
        st.session_state.s32_x_star_str = str(X_star_default)
        is_editable_s32 = False
        # Для предустановки Y всегда вычисляются из e^x
        st.session_state.s32_y_source_choice = f"Вычислить из $y=f(x)={default_func_name_display}$"
    else:  # Пользовательский ввод
        is_editable_s32 = True
        # s32_y_source_choice остается как есть, чтобы пользователь мог выбрать

    st.subheader("Входные данные для сплайна")

    cols_s32_1 = st.columns(2)
    with cols_s32_1[0]:
        st.session_state.s32_x_nodes_str = st.text_input(
            "Узлы $X_i$ (через запятую, мин. 2 узла):",
            value=st.session_state.s32_x_nodes_str,
            disabled=not is_editable_s32,
            key="s32_x_nodes_input"
        )
    with cols_s32_1[1]:
        y_source_options_s32 = [f"Вычислить из $y=f(x)={default_func_name_display}$", "Ввести $Y_i$ вручную"]

        # Убедимся, что текущий выбор y_source_choice валиден
        if st.session_state.s32_y_source_choice not in y_source_options_s32:
            st.session_state.s32_y_source_choice = y_source_options_s32[0]

        current_y_source_idx_s32 = 0 if not is_editable_s32 else y_source_options_s32.index(
            st.session_state.s32_y_source_choice)

        st.session_state.s32_y_source_choice = st.radio(
            "Источник $Y_i$:",
            y_source_options_s32,
            index=current_y_source_idx_s32,
            key="s32_y_source_radio",
            horizontal=True,
            disabled=not is_editable_s32
        )

    if is_editable_s32 and st.session_state.s32_y_source_choice == "Ввести $Y_i$ вручную":
        st.session_state.s32_y_nodes_str = st.text_input(
            "Значения $Y_i$ (через запятую, соответствуют $X_i$):",
            value=st.session_state.s32_y_nodes_str,
            key="s32_y_nodes_manual_input"
        )

    st.session_state.s32_x_star_str = st.text_input(
        "Точка для вычисления значения сплайна $X^*$:",
        value=st.session_state.s32_x_star_str,
        key="s32_x_star_input"
    )

    if st.button("🛠️ Построить сплайн и вычислить значение", key="s32_run_button"):
        try:
            # --- 1. Парсинг и валидация входных данных ---
            x_nodes_str_s32 = st.session_state.s32_x_nodes_str
            try:
                current_X_nodes_s32 = np.array([float(x.strip()) for x in x_nodes_str_s32.split(',')])
            except ValueError:
                st.error(f"Ошибка парсинга узлов $X_i$: '{x_nodes_str_s32}'. Введите числа через запятую.")
                return

            if len(current_X_nodes_s32) < 2:
                st.error("Для построения сплайна необходимо как минимум 2 узла.")
                return

            # Получение Y_nodes
            actual_y_source_s32 = st.session_state.s32_y_source_choice
            if not is_editable_s32:  # Если выбрана предустановка
                actual_y_source_s32 = f"Вычислить из $y=f(x)={default_func_name_display}$"

            if actual_y_source_s32 == f"Вычислить из $y=f(x)={default_func_name_display}$":
                current_Y_nodes_s32 = default_func_to_interpolate(current_X_nodes_s32)
                y_data_source_info_s32 = f"вычислены по функции $y={default_func_name_display}$"
            else:  # Ручной ввод (и is_editable_s32 == True)
                y_nodes_str_s32 = st.session_state.s32_y_nodes_str
                if not y_nodes_str_s32.strip():
                    st.error("Введите значения $Y_i$ или выберите вычисление по функции.")
                    return
                try:
                    current_Y_nodes_s32 = np.array([float(y.strip()) for y in y_nodes_str_s32.split(',')])
                except ValueError:
                    st.error(f"Ошибка парсинга $Y_i$: '{y_nodes_str_s32}'. Введите числа через запятую.")
                    return
                y_data_source_info_s32 = "введены вручную"

            if len(current_X_nodes_s32) != len(current_Y_nodes_s32):
                st.error(f"Количество узлов $X_i$ ({len(current_X_nodes_s32)}) "
                         f"должно совпадать с количеством значений $Y_i$ ({len(current_Y_nodes_s32)}).")
                return

            try:
                current_X_star_s32 = float(st.session_state.s32_x_star_str)
            except ValueError:
                st.error(f"Ошибка парсинга $X^*$: '{st.session_state.s32_x_star_str}'. Введите число.")
                return

            st.markdown("---")
            st.subheader(" spline Результаты построения сплайна")

            # --- 2. Построение сплайна ---
            spline = CubicSpline(current_X_nodes_s32, current_Y_nodes_s32, natural_spline=True)

            # Обновляем current_X_nodes_s32 и current_Y_nodes_s32 отсортированными значениями из объекта сплайна
            current_X_nodes_s32_sorted = spline.x_nodes
            current_Y_nodes_s32_sorted = spline.y_nodes

            st.markdown("#### Входные данные (отсортированные):")
            df_input_s32 = pd.DataFrame({'i': range(len(current_X_nodes_s32_sorted)),
                                         'X_i': current_X_nodes_s32_sorted,
                                         'Y_i': current_Y_nodes_s32_sorted})
            st.dataframe(df_input_s32.style.format({'X_i': "{:.4f}", 'Y_i': "{:.7f}"}))
            st.write(f"Значения $Y_i$ {y_data_source_info_s32}.")
            st.write(f"Точка для вычисления значения сплайна $X^* = {current_X_star_s32:.4f}$")

            # Проверка X* на принадлежность отрезку интерполяции
            min_X_node_s32, max_X_node_s32 = spline.x_nodes[0], spline.x_nodes[-1]
            if not (min_X_node_s32 <= current_X_star_s32 <= max_X_node_s32):
                st.warning(
                    f"⚠️ **Предупреждение:** Точка $X^*={current_X_star_s32:.4f}$ находится вне основного отрезка интерполяции "
                    f"$[{min_X_node_s32:.4f}, {max_X_node_s32:.4f}]$. Выполняется экстраполяция сплайном, "
                    "результаты могут быть менее точными."
                )

            # --- 3. Вывод информации о сплайне ---
            st.markdown("#### Коэффициенты кубического сплайна $S_i(x)$")
            st.markdown("Для каждого сегмента $i \\in [0, n-1]$, где $n$ - число интервалов, "
                        "$S_i(x) = a_i + b_i(x-x_i) + \\frac{c_i}{2}(x-x_i)^2 + \\frac{d_i}{6}(x-x_i)^3$ "
                        "на отрезке $[x_i, x_{i+1}]$.")
            st.markdown("Здесь $c_i = S''(x_i)$ - вторая производная в узле $x_i$. "
                        "Для естественного сплайна $c_0 = 0$ и $c_n = 0$.")

            # Таблица коэффициентов
            coeffs_df = spline.coeffs_table.copy()
            formatters_coeffs = {col: "{:.5f}" for col in coeffs_df.columns if
                                 col not in ['i (сегмент)', '$x_i$', '$x_{i+1}$']}
            formatters_coeffs['$x_i$'] = "{:.4f}"
            formatters_coeffs['$x_{i+1}$'] = "{:.4f}"
            st.dataframe(coeffs_df.style.format(formatters_coeffs))

            st.markdown(f"Вторые производные в узлах $c_j=S''(x_j)$: "
                        f"{', '.join([f'c_{j}={val:.5f}' for j, val in enumerate(spline.c)])}")
            st.success(
                f"✅ Проверка краевых условий для естественного сплайна: $c_0 = {spline.c[0]:.5f}$ и $c_n = c_{{{spline.n}}} = {spline.c[-1]:.5f}$ (должны быть близки к 0).")

            with st.expander("Показать символьные выражения для сегментов сплайна $S_i(x)$"):
                x_sym_s32 = Symbol('x')
                spline_symbolic = spline.get_symbolic_spline(x_sym=x_sym_s32)
                if isinstance(spline_symbolic, str): # Ошибка или сообщение
                    st.write(spline_symbolic)
                else:
                    st.latex(f"S(x) = {latex(spline_symbolic, full_prec=False, inv_trig_style='power')}")
                    st.markdown("Обратите внимание: символьные выражения могут быть громоздкими. Округление применено для читаемости.")

            # --- 4. Вычисление значения в X* ---
            S_value_at_X_star = spline.evaluate(current_X_star_s32)
            st.markdown(f"#### Значение сплайна в точке $X^*={current_X_star_s32:.4f}$")
            st.markdown(f"**$S(X^*) = {S_value_at_X_star:.7f}$**")

            # Погрешность, если Y из функции
            if actual_y_source_s32 == f"Вычислить из $y=f(x)={default_func_name_display}$":
                f_true_at_X_star_s32 = default_func_to_interpolate(current_X_star_s32)
                st.write(
                    f"Истинное значение функции $f(X^*) = {default_func_name_display}({current_X_star_s32:.4f}) = {f_true_at_X_star_s32:.7f}$")

                abs_error_s32 = abs(f_true_at_X_star_s32 - S_value_at_X_star)
                rel_error_percent_s32 = (abs_error_s32 / abs(f_true_at_X_star_s32) * 100) if not np.isclose(
                    f_true_at_X_star_s32, 0) else (0 if np.isclose(abs_error_s32, 0) else float('inf'))

                st.success(f"Абсолютная погрешность интерполяции сплайном: $|f(X^*) - S(X^*)| = {abs_error_s32:.7f}$")
                st.success(f"Относительная погрешность: ${rel_error_percent_s32:.5f}\\%$ (если $f(X^*) \\neq 0$)")

                # Теоретическая оценка погрешности
                if hasattr(default_func_to_interpolate, '__name__') and default_func_to_interpolate.__name__ == 'exp':
                    # f(x) = e^x, f^(4)(x) = e^x
                    # M4 = max |e^x| на [x_0, x_n]
                    x_range_for_M4 = np.linspace(min_X_node_s32, max_X_node_s32, 200)
                    M4 = np.max(np.exp(x_range_for_M4))
                    H_max = np.max(spline.h)
                    theoretical_error_bound = (5.0 / 384.0) * M4 * H_max ** 4
                    st.info(
                        f"Теоретическая оценка максимальной абсолютной погрешности для естественного кубического сплайна "
                        f"на отрезке $[{min_X_node_s32:.2f}, {max_X_node_s32:.2f}]$: "
                        f"$|f(x) - S(x)| \\leq \\frac{{5}}{{384}} M_4 H^4 \\approx {theoretical_error_bound:.2e}$, "
                        f"где $M_4 = \\max |f^{{(4)}}(x)| \\approx {M4:.2e}$ и $H = \\max h_i \\approx {H_max:.2f}$."
                        f"\nЭта оценка показывает порядок возможной погрешности, реальная погрешность в $X^*$ может быть меньше."
                    )
            else:
                st.info(
                    "Поскольку значения $Y_i$ введены вручную, точное $f(X^*)$ и погрешность относительно него неизвестны.")

            # --- 5. График ---
            st.markdown("#### Графическая иллюстрация сплайна")
            plot_fig_s32 = go.Figure()

            # Диапазон для построения графика сплайна (немного шире узлов)
            plot_margin = 0.5 * (max_X_node_s32 - min_X_node_s32) if max_X_node_s32 > min_X_node_s32 else 1.0
            plot_x_min_s32 = min(min_X_node_s32, current_X_star_s32) - plot_margin
            plot_x_max_s32 = max(max_X_node_s32, current_X_star_s32) + plot_margin
            if plot_x_min_s32 == plot_x_max_s32:  # Редкий случай
                plot_x_min_s32 -= 1
                plot_x_max_s32 += 1

            x_dense_for_plot_s32 = np.linspace(plot_x_min_s32, plot_x_max_s32, 500)
            y_spline_dense_plot = np.array([spline.evaluate(x_val) for x_val in x_dense_for_plot_s32])

            # Истинная функция (если Y вычислялись по ней)
            if actual_y_source_s32 == f"Вычислить из $y=f(x)={default_func_name_display}$":
                plot_fig_s32.add_trace(go.Scatter(
                    x=x_dense_for_plot_s32,
                    y=default_func_to_interpolate(x_dense_for_plot_s32),
                    mode='lines',
                    name=f"Истинная функция $f(x)={default_func_name_display}$",
                    line=dict(dash='dash', color='green', width=2)
                ))

            # Кубический сплайн
            plot_fig_s32.add_trace(go.Scatter(
                x=x_dense_for_plot_s32,
                y=y_spline_dense_plot,
                mode='lines',
                name="Кубический сплайн $S(x)$",
                line=dict(color='rgba(255,100,0,0.9)', width=2.5)  # Оранжевый
            ))

            # Узловые точки
            plot_fig_s32.add_trace(go.Scatter(
                x=spline.x_nodes,
                y=spline.y_nodes,
                mode='markers',
                name='Узловые точки $(X_i, Y_i)$',
                marker=dict(size=10, color='blue', symbol='circle', line=dict(width=1, color='DarkSlateGrey'))
            ))

            # Точка (X*, S(X*))
            plot_fig_s32.add_trace(go.Scatter(
                x=[current_X_star_s32],
                y=[S_value_at_X_star],
                mode='markers', name=f'$S(X^*={current_X_star_s32:.2f})$',
                marker=dict(size=12, color='red', symbol='x-dot', line=dict(width=2, color='DarkSlateGrey'))
            ))

            # Точка (X*, f(X*)) истинная, если Y вычислялись
            if actual_y_source_s32 == f"Вычислить из $y=f(x)={default_func_name_display}$":
                plot_fig_s32.add_trace(go.Scatter(
                    x=[current_X_star_s32],
                    y=[f_true_at_X_star_s32],
                    mode='markers', name=f'$f(X^*={current_X_star_s32:.2f})$ (истина)',
                    marker=dict(size=12, color='green', symbol='cross-dot', line=dict(width=2, color='DarkSlateGrey'))
                ))

            plot_fig_s32.update_layout(
                title=f"Кубическая сплайн-интерполяция",
                xaxis_title=" ось X",
                yaxis_title=" ось Y",
                legend_title_text="<b>Обозначения на графике</b>",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(plot_fig_s32, use_container_width=True)


        except ValueError as ve:
            st.error(f"🚫 Ошибка значения при построении сплайна: {ve}")
            # st.error(traceback.format_exc()) # Для отладки
        except np.linalg.LinAlgError as lae:
            st.error(
                f"🚫 Ошибка линейной алгебры: {lae}. Возможно, узлы некорректны (например, дубликаты, которые не были отловлены, или слишком мало уникальных узлов для формирования невырожденной системы).")
            # st.error(traceback.format_exc()) # Для отладки
        except Exception as e:
            st.error(f"💥 Произошла непредвиденная ошибка при работе со сплайном: {e}")
            st.error("Полная информация об ошибке для разработчика:")
            st.code(traceback.format_exc())

# Кастомный CSS (если нужен, как в прошлом примере)
st.markdown(r"""
<style>
.stRadio[role=radiogroup] {
    flex-direction: row; 
    gap: 15px; 
}
.stRadio[role=radiogroup] > label {
    margin-right: 0; 
}
</style>
""", unsafe_allow_html=True)
st.title("🚀 Лабораторная работа по численным методам")
st.markdown("---")
section_3_2()
