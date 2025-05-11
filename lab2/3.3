import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sympy import Symbol, Matrix, latex, expand, N
import pandas as pd
import traceback


# --- Математические функции для МНК (остаются без изменений) ---

def solve_lsm_polynomial(x_nodes: np.ndarray, y_nodes: np.ndarray, degree: int):
    """
    Находит коэффициенты аппроксимирующего многочлена P_degree(x) методом наименьших квадратов.
    Многочлен ищется в виде P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_degree*x^degree.

    Возвращает:
        coeffs (np.ndarray): Массив коэффициентов [a_0, a_1, ..., a_degree].
        G (np.ndarray): Матрица Грама (левая часть) нормальной системы уравнений.
        b_vec (np.ndarray): Вектор правой части нормальной системы уравнений.
    """
    m = len(x_nodes)  # Количество точек
    if m < degree + 1:
        raise ValueError(
            f"Для аппроксимации многочленом степени {degree} необходимо как минимум {degree + 1} точек. "
            f"Получено: {m}."
        )

    # --- Формирование нормальной системы МНК: G * a = b_vec ---
    # G - матрица Грама, (degree+1) x (degree+1)
    # Элементы G[l, j] = sum(x_i^(l+j)) for i from 0 to m-1
    # l, j - индексы от 0 до 'degree'
    G = np.zeros((degree + 1, degree + 1))
    for l_idx in range(degree + 1): # Индекс строки матрицы G (соответствует k в формуле (3.17) из текста)
        for j_idx in range(degree + 1): # Индекс столбца матрицы G (соответствует i в формуле (3.17))
            # G[l_idx, j_idx] = sum_{i=0}^{m-1} (x_nodes[i] ^ (l_idx + j_idx))
            G[l_idx, j_idx] = np.sum(x_nodes ** (l_idx + j_idx))

    # b_vec - вектор правой части, (degree+1) x 1
    # Элементы b_vec[l] = sum(y_i * x_i^l) for i from 0 to m-1
    b_vec = np.zeros(degree + 1)
    for l_idx in range(degree + 1): # Индекс элемента вектора b_vec (соответствует k в формуле (3.17))
        # b_vec[l_idx] = sum_{i=0}^{m-1} (y_nodes[i] * (x_nodes[i] ^ l_idx))
        b_vec[l_idx] = np.sum(y_nodes * (x_nodes ** l_idx))

    # --- Решение нормальной системы ---
    try:
        # Пытаемся решить систему G * coeffs = b_vec стандартным методом
        coeffs = np.linalg.solve(G, b_vec)
        # Закомментирована попытка регуляризации - добавление малой диагональной компоненты к G
        # для улучшения обусловленности, если матрица G близка к вырожденной.
        # G_reg = G + np.eye(degree + 1) * 1e-9
        # coeffs = np.linalg.solve(G_reg, b_vec)
    except np.linalg.LinAlgError: # Если матрица G вырождена (определитель равен 0) или плохо обусловлена
        st.warning("Матрица Грама (G) вырождена или плохо обусловлена. "
                   "Это может произойти, если узлы X_i не достаточно разнообразны, "
                   "их количество близко к степени полинома, или степень полинома слишком высока для данных. "
                   "Попытка решения с использованием псевдообратной матрицы (менее точный метод)...")
        try:
            # np.linalg.lstsq решает систему Ax=b методом наименьших квадратов,
            # что эквивалентно использованию псевдообратной матрицы для вырожденных систем.
            # Он возвращает кортеж, первый элемент которого - решение.
            coeffs = np.linalg.lstsq(G, b_vec, rcond=None)[0]
        except np.linalg.LinAlgError: # Если и псевдообратная матрица не помогла
            raise ValueError("Не удалось решить нормальную систему МНК даже с помощью псевдообратной матрицы. "
                             "Проверьте входные данные (узлы X_i) и выбранную степень полинома.")
    return coeffs, G, b_vec # Возвращаем найденные коэффициенты и компоненты системы


def polynomial_value(coeffs: np.ndarray, x_eval: float) -> float:
    """
    Вычисляет значение многочлена P(x_eval) = a0 + a1*x + a2*x^2 + ...
    по его коэффициентам (coeffs = [a0, a1, a2, ...]).
    """
    val = 0.0
    # Итерируем по коэффициентам и соответствующим степеням x_eval
    for k, ak in enumerate(coeffs):  # k - степень, ak - коэффициент при x^k
        val += ak * (x_eval ** k)
    # Альтернатива: np.polyval ожидает коэффициенты в порядке убывания степеней.
    # Наши 'coeffs' в порядке возрастания, поэтому нужно coeffs[::-1].
    # val = np.polyval(coeffs[::-1], x_eval)
    return val


def polynomial_symbolic(coeffs: np.ndarray, x_sym: Symbol = None, precision: int = 5):
    """
    Возвращает символьное представление многочлена P(x) с использованием Sympy.
    Коэффициенты округляются до 'precision' знаков для лучшей читаемости.
    """
    if x_sym is None:
        x_sym = Symbol('x') # Создаем символьную переменную 'x', если не передана

    poly_expr = 0 # Инициализируем символьное выражение нулем
    # Итерируем по коэффициентам для построения суммы a_k * x^k
    for k, ak_val in enumerate(coeffs):
        # sympy.N(value, n) округляет 'value' до 'n' значащих цифр (в данном случае используется как точность после запятой при выводе).
        # Это помогает избежать очень длинных чисел в символьном выражении.
        poly_expr += N(ak_val, precision) * (x_sym ** k)
    return expand(poly_expr) # expand() раскрывает скобки и приводит к стандартному виду


def sum_squared_errors(x_nodes: np.ndarray, y_nodes: np.ndarray, coeffs: np.ndarray) -> float:
    """
    Вычисляет сумму квадратов ошибок (SSE или RSS): E = sum_{i=0}^{m-1} (P(x_i) - y_i)^2.
    Это значение, которое минимизируется методом наименьших квадратов.
    """
    # Вычисляем предсказанные значения y_pred = P(x_i) для всех узловых x_i
    y_pred = np.array([polynomial_value(coeffs, xi) for xi in x_nodes])
    # Вычисляем сумму квадратов разностей между предсказанными и фактическими значениями y
    error = np.sum((y_pred - y_nodes) ** 2)
    return error


# --- Streamlit UI для пункта 3.3 ---
def section_3_3():
    st.header("3.3. Метод наименьших квадратов (МНК)")

    st.sidebar.subheader("Настройки для пункта 3.3")

    # Предустановленные значения
    X_i_default_s33 = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    Y_i_default_s33 = np.array([0.04979, 0.13534, 0.36788, 1.0, 2.7183, 7.3891])

    data_source_options_s33 = [
        f"Предустановка ({len(X_i_default_s33)} узлов, данные похожи на $e^x$)",
        "Пользовательский ввод"
    ]

    if 's33_data_source' not in st.session_state:
        st.session_state.s33_data_source = data_source_options_s33[0]

    st.session_state.s33_data_source = st.sidebar.radio(
        "Выберите источник табличных данных:",
        data_source_options_s33,
        index=data_source_options_s33.index(st.session_state.s33_data_source),
        key="s33_data_source_radio"
    )

    # Инициализация значений в session_state
    if 's33_x_nodes_str' not in st.session_state:
        st.session_state.s33_x_nodes_str = ", ".join(map(str, X_i_default_s33))
    if 's33_y_nodes_str' not in st.session_state:
        st.session_state.s33_y_nodes_str = ", ".join(map(lambda x: f"{x:.5f}", Y_i_default_s33))
    if 's33_poly_degree' not in st.session_state:
        st.session_state.s33_poly_degree = 1  # Начальная степень по умолчанию

    # Установка значений и состояния редактируемости
    if st.session_state.s33_data_source == data_source_options_s33[0]:
        st.session_state.s33_x_nodes_str = ", ".join(map(str, X_i_default_s33))
        st.session_state.s33_y_nodes_str = ", ".join(map(lambda x: f"{x:.5f}", Y_i_default_s33))
        is_editable_s33 = False
    else:
        is_editable_s33 = True

    st.subheader("Входные табличные данные и параметры")

    cols_s33_input1 = st.columns(2)
    with cols_s33_input1[0]:
        st.session_state.s33_x_nodes_str = st.text_input(
            "Узлы $X_i$ (через запятую):",
            value=st.session_state.s33_x_nodes_str,
            disabled=not is_editable_s33,
            key="s33_x_nodes_input"
        )
    with cols_s33_input1[1]:
        st.session_state.s33_y_nodes_str = st.text_input(
            "Значения $Y_i$ (через запятую, соответствуют $X_i$):",
            value=st.session_state.s33_y_nodes_str,
            disabled=not is_editable_s33,
            key="s33_y_nodes_input"
        )

    # Ввод степени полинома
    st.session_state.s33_poly_degree = st.number_input(
        "Степень аппроксимирующего многочлена $k$:",
        min_value=0,
        max_value=10,  # Ограничение сверху, чтобы избежать слишком сложных вычислений / плохой обусловленности
        value=st.session_state.s33_poly_degree,
        step=1,
        key="s33_poly_degree_input",
        help="Выберите степень полинома $P_k(x)$ для аппроксимации. $k=0$ - константа, $k=1$ - прямая, $k=2$ - парабола и т.д."
    )

    if st.button("📊 Выполнить аппроксимацию МНК", key="s33_run_button"):
        try:
            # --- 1. Парсинг и валидация входных данных ---
            x_nodes_str_s33 = st.session_state.s33_x_nodes_str
            y_nodes_str_s33 = st.session_state.s33_y_nodes_str

            try:
                current_X_nodes_s33 = np.array([float(x.strip()) for x in x_nodes_str_s33.split(',')])
                current_Y_nodes_s33 = np.array([float(y.strip()) for y in y_nodes_str_s33.split(',')])
            except ValueError:
                st.error("Ошибка парсинга $X_i$ или $Y_i$. Введите числа через запятую.")
                return

            if len(current_X_nodes_s33) != len(current_Y_nodes_s33):
                st.error(f"Количество узлов $X_i$ ({len(current_X_nodes_s33)}) "
                         f"должно совпадать с $Y_i$ ({len(current_Y_nodes_s33)}).")
                return

            if len(current_X_nodes_s33) == 0:
                st.error("Необходимо ввести данные (узлы $X_i$ и $Y_i$).")
                return

            poly_degree_to_fit = st.session_state.s33_poly_degree

            # Проверка на достаточное количество точек
            if len(current_X_nodes_s33) < poly_degree_to_fit + 1:
                st.error(f"Для аппроксимации многочленом степени {poly_degree_to_fit} необходимо как минимум "
                         f"{poly_degree_to_fit + 1} точек. Получено: {len(current_X_nodes_s33)}.")
                return

            # Сортировка узлов (не обязательна для МНК, но удобна для графиков)
            sorted_indices_s33 = np.argsort(current_X_nodes_s33)
            current_X_nodes_s33_sorted = current_X_nodes_s33[sorted_indices_s33]
            current_Y_nodes_s33_sorted = current_Y_nodes_s33[sorted_indices_s33]

            st.markdown("---")
            st.subheader(f"🔍 Результаты аппроксимации МНК (многочлен степени {poly_degree_to_fit})")

            st.markdown("#### Входные данные для МНК (отсортированные по $X_i$):")
            df_input_s33 = pd.DataFrame({'i': range(len(current_X_nodes_s33_sorted)),
                                         'X_i': current_X_nodes_s33_sorted,
                                         'Y_i': current_Y_nodes_s33_sorted})
            st.dataframe(df_input_s33.style.format({'X_i': "{:.4f}", 'Y_i': "{:.5f}"}))

            x_symbol_s33 = Symbol('x')

            # --- Аппроксимация многочленом выбранной степени ---
            coeffs_pk, G_pk_np, b_pk_np = solve_lsm_polynomial(
                current_X_nodes_s33_sorted,
                current_Y_nodes_s33_sorted,
                poly_degree_to_fit
            )
            pk_symbolic = polynomial_symbolic(coeffs_pk, x_symbol_s33, precision=5)  # Используем 5 знаков для sympy
            sse_pk = sum_squared_errors(current_X_nodes_s33_sorted, current_Y_nodes_s33_sorted, coeffs_pk)

            st.markdown(f"#### Аппроксимация многочленом $P_{{{poly_degree_to_fit}}}(x)$")
            st.markdown("Нормальная система $G \\cdot \\mathbf{a} = \\mathbf{b}$:")

            G_pk_sympy = Matrix(np.round(G_pk_np, 4))  # Округление для вывода
            b_pk_sympy = Matrix(np.round(b_pk_np.reshape(-1, 1), 4))

            # Символы для вектора коэффициентов a_0, a_1, ..., a_k
            coeff_symbols = [Symbol(f'a_{j}') for j in range(poly_degree_to_fit + 1)]
            coeffs_vec_sympy_pk = Matrix(coeff_symbols)

            sys_eq_md_pk = f"$$ {latex(G_pk_sympy)} {latex(coeffs_vec_sympy_pk)} = {latex(b_pk_sympy)} $$"
            st.markdown(sys_eq_md_pk)

            coeffs_str = ", ".join([f"$a_{j} = {coeffs_pk[j]:.5f}$" for j in range(len(coeffs_pk))])
            st.markdown(f"Коэффициенты: {coeffs_str}")
            st.latex(f"P_{{{poly_degree_to_fit}}}(x) = {latex(pk_symbolic)}")
            st.success(
                f"Сумма квадратов ошибок $E_{{{poly_degree_to_fit}}} = \\sum (P_{{{poly_degree_to_fit}}}(x_i) - y_i)^2 = {sse_pk:.7f}$")

            # --- График ---
            st.markdown("---")
            st.markdown("#### Графическая иллюстрация аппроксимации")
            plot_fig_s33 = go.Figure()

            # Исходные точки
            plot_fig_s33.add_trace(go.Scatter(
                x=current_X_nodes_s33_sorted,
                y=current_Y_nodes_s33_sorted,
                mode='markers',
                name='Исходные точки $(X_i, Y_i)$',
                marker=dict(size=10, color='blue', symbol='circle')
            ))

            # Диапазон для построения графиков полиномов
            plot_x_min_s33 = np.min(current_X_nodes_s33_sorted)
            plot_x_max_s33 = np.max(current_X_nodes_s33_sorted)
            plot_margin_s33 = 0.1 * (plot_x_max_s33 - plot_x_min_s33) if plot_x_max_s33 > plot_x_min_s33 else 0.5

            x_dense_for_plot_s33 = np.linspace(plot_x_min_s33 - plot_margin_s33,
                                               plot_x_max_s33 + plot_margin_s33,
                                               300)

            # Аппроксимирующий полином
            y_pk_dense_plot = np.array([polynomial_value(coeffs_pk, x_val) for x_val in x_dense_for_plot_s33])
            plot_fig_s33.add_trace(go.Scatter(
                x=x_dense_for_plot_s33,
                y=y_pk_dense_plot,
                mode='lines',
                name=f"$P_{{{poly_degree_to_fit}}}(x)$ ($E={sse_pk:.3e}$)",
                line=dict(color='rgba(255,0,0,0.8)', width=2)
            ))

            plot_fig_s33.update_layout(
                title=f"Аппроксимация МНК многочленом степени {poly_degree_to_fit}",
                xaxis_title=" ось X",
                yaxis_title=" ось Y",
                legend_title_text="<b>Обозначения</b>",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(plot_fig_s33, use_container_width=True)

        except ValueError as ve:
            st.error(f"🚫 Ошибка значения при МНК: {ve}")
            # st.error(traceback.format_exc())
        except Exception as e:
            st.error(f"💥 Произошла непредвиденная ошибка при МНК: {e}")
            st.error("Полная информация об ошибке для разработчика:")
            st.code(traceback.format_exc())


# --- Запуск (как в предыдущих примерах) ---
st.markdown(r"""
<style>
.stRadio[role=radiogroup] { flex-direction: row; gap: 15px; }
.stRadio[role=radiogroup] > label { margin-right: 0; }
</style>
""", unsafe_allow_html=True)
st.title("🚀 Лабораторная работа по численным методам")
st.markdown("---")
section_3_3()
