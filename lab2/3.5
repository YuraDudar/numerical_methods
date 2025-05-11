import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import traceback
import sympy
from sympy import lambdify, Symbol, integrate as sympy_integrate, sympify, SympifyError

# --- Функция для интегрирования ---
def user_function_wrapper(func_str, x_sym_for_func):
    """
    Преобразует строку func_str в вызываемую Python-функцию.
    Использует sympy для парсинга строки и lambdify для создания функции.
    x_sym_for_func - это символьная переменная (например, Symbol('x')),
    которая будет использоваться в строке функции.
    Возвращает кортеж (вызываемая_функция, sympy_выражение).
    """
    try:
        expr = sympify(func_str)
        # Проверка на наличие x в выражении, если нет - это константа
        if not expr.has(x_sym_for_func) and not expr.is_Number:
            # Если это не число и не содержит x, возможно, ошибка в строке
            # Но sympify может вернуть просто символ, если строка "y"
            pass  # Пока пропустим, lambdify может выдать ошибку

        # Проверка знаменателя на 0 (если это дробь)
        # Это сложнее сделать универсально для sympy выражения без явного анализа структуры
        # Проще ловить ZeroDivisionError или ValueError при вычислении.

        func = lambdify(x_sym_for_func, expr, modules=['numpy', {'Heaviside': lambda x_h: np.heaviside(x_h, 0.5)}])

        # Тестовый вызов для проверки на ранние ошибки
        try:
            _ = func(0.5)  # Примерное значение внутри обычного диапазона
            if isinstance(_, (int, float, complex)) and np.isnan(_):  # Если результат NaN без ошибки
                # Это может быть из-за log(-1) и т.п. Lambdify может не кинуть исключение
                pass  # Оставим обработку ошибок на момент вычисления интеграла
        except (TypeError, NameError) as e_lambdify_test:  # Ошибка в выражении или неизвестные символы
            raise ValueError(
                f"Ошибка в определении функции '{func_str}': {e_lambdify_test}. Убедитесь, что используется переменная 'x'.")
        except Exception:  # Другие ошибки при тестовом вызове
            pass  # Ловим при вычислении интеграла

        return func, expr
    except SympifyError:
        raise ValueError(f"Не удалось распознать математическое выражение: '{func_str}'.")
    except Exception as e:
        raise ValueError(f"Неизвестная ошибка при создании функции из строки '{func_str}': {e}")


# --- Квадратурные формулы ---

def get_integration_grid(x0, xk, N):
    """
    Генерирует равномерную сетку узлов для интегрирования от x0 до xk с N интервалами.
    Возвращает:
        xs (np.ndarray): Массив узлов, включая x0 и xk (N+1 точек).
        h_actual (float): Фактический шаг сетки.
    """
    if N <= 0:
        raise ValueError("Количество интервалов N должно быть положительным.")
    h_actual = (xk - x0) / N
    xs = np.linspace(x0, xk, N + 1)
    return xs, h_actual


def method_rectangles_midpoint(func, xs, h_actual):
    """Метод средних прямоугольников."""
    integral = 0.0
    for i in range(len(xs) - 1):
        midpoint = (xs[i] + xs[i + 1]) / 2
        try:
            val_f = func(midpoint)
            if np.isnan(val_f) or np.isinf(val_f):
                raise ValueError(f"Функция вернула NaN или Inf в точке x={midpoint:.4f}")
            integral += val_f
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise ValueError(f"Ошибка вычисления f({midpoint:.4f}) в методе прямоугольников: {e}")
    return integral * h_actual


def method_trapezoids(func, xs, h_actual):
    """Метод трапеций."""
    integral = 0.0
    ys = np.zeros(len(xs))
    for i, xi in enumerate(xs):
        try:
            val_f = func(xi)
            if np.isnan(val_f) or np.isinf(val_f):
                raise ValueError(f"Функция вернула NaN или Inf в точке x={xi:.4f}")
            ys[i] = val_f
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise ValueError(f"Ошибка вычисления f({xi:.4f}) в методе трапеций: {e}")

    integral = (ys[0] + ys[-1]) / 2.0
    integral += np.sum(ys[1:-1])
    return integral * h_actual


def method_simpson(func, xs, h_actual):
    """Метод Симпсона."""
    N = len(xs) - 1 # Количество интервалов
    # Метод Симпсона требует четного числа интервалов N.
    if N % 2 != 0:
        raise ValueError("Для метода Симпсона требуется четное число интервалов (N).")

    ys = np.zeros(len(xs)) # Массив для значений функции в узлах
    # Вычисляем значения функции во всех узлах
    for i, xi in enumerate(xs):
        try:
            val_f = func(xi)
            if np.isnan(val_f) or np.isinf(val_f):
                raise ValueError(f"Функция вернула NaN или Inf в точке x={xi:.4f}")
            ys[i] = val_f
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            raise ValueError(f"Ошибка вычисления f({xi:.4f}) в методе Симпсона: {e}")

    # Формула Симпсона: (h/3) * [y0 + yN + 4*(y1+y3+...) + 2*(y2+y4+...)]
    integral = ys[0] + ys[-1] # y0 + yN
    # Сумма значений с нечетными индексами (y1, y3, ..., y_{N-1}) умножается на 4
    integral += 4 * np.sum(ys[1:-1:2])  # ys[1:-1:2] выбирает элементы с шагом 2, начиная с ys[1] до предпоследнего
    # Сумма значений с четными индексами (y2, y4, ..., y_{N-2}) умножается на 2
    integral += 2 * np.sum(ys[2:-2:2])  # ys[2:-2:2] выбирает элементы с шагом 2, начиная с ys[2] до пред-предпоследнего
    return integral * (h_actual / 3.0) # Умножаем на h/3


# --- Метод Рунге-Ромберга ---
def runge_romberg_error_estimation(I_h1, I_h2, h1, h2, p):
    """
    Оценка погрешности и уточнение результата по методу Рунге-Ромберга.
    I_h1 - значение интеграла, полученное с шагом h1.
    I_h2 - значение интеграла, полученное с шагом h2.
    p - порядок точности основного метода численного интегрирования.
    Предполагается, что h2 < h1 (I_h2 более точное).
    """
    if h1 <= 0 or h2 <= 0 or p <= 0:
        raise ValueError("Шаги h1, h2 и порядок точности p должны быть положительными.")
    if np.isclose(h1, h2):  # Шаги не должны быть слишком близки, иначе знаменатель будет 0
        raise ValueError("Шаги h1 и h2 слишком близки для корректной оценки по Рунге-Ромбергу.")

    k = h1 / h2  # Отношение шагов, k > 1
    if k < 1:  # Если h1 < h2, меняем местами для k > 1
        k = 1 / k
        # Погрешность оценивается для более грубого шага, т.е. I_h1
        # Error(I_h1) ~ (I_h1 - I_h2) / (1 - (h2/h1)^p) = (I_h1 - I_h2) / (1 - (1/k)^p)
        # Если хотим погрешность для I_h2: Error(I_h2) ~ (I_h2 - I_h1) / ( (h1/h2)^p - 1)
        # Формула из лекций: R_h2 = (I_h2 - I_h1) / ( (h1/h2)^p - 1 )
        # I_h1 - значение с большим шагом, I_h2 - с меньшим
        # Здесь I_h1 и I_h2 передаются как значения, а не как обозначения шагов
        # Пусть I_h1 - значение с шагом h1, I_h2 - значение с шагом h2
        # Если h1 > h2, то k = h1/h2 > 1. Погрешность для I_h2: (I_h2 - I_h1) / (k^p - 1)
        # Если h2 > h1, то k = h2/h1 > 1. Погрешность для I_h1: (I_h1 - I_h2) / (k^p - 1)

    # Стандартная формула: Погрешность более точного значения (с шагом h2)
    # R_h2 = (I_h2 - I_h1) / ( (h1/h2)^p - 1 )
    # Убедимся, что h1 - больший шаг, h2 - меньший шаг.
    # Если это не так, поменяем их местами вместе с интегралами.
    current_I_h_coarse, current_I_h_fine = I_h1, I_h2
    current_h_coarse, current_h_fine = h1, h2

    if h1 < h2:  # h1 должен быть большим шагом
        current_I_h_coarse, current_I_h_fine = I_h2, I_h1
        current_h_coarse, current_h_fine = h2, h1

    ratio_k = current_h_coarse / current_h_fine

    denominator = ratio_k ** p - 1
    if abs(denominator) < 1e-12:  # Избегаем деления на ноль
        raise ValueError(f"Знаменатель в методе Рунге-Ромберга ({denominator:.2e}) близок к нулю (k^p - 1). "
                         f"k={ratio_k:.3f}, p={p}. Проверьте шаги и порядок точности.")

    # Погрешность для значения с МЕНЬШИМ шагом (I_h_fine)
    error_estimate_fine = (current_I_h_fine - current_I_h_coarse) / denominator
    # Уточненное значение
    I_уточненное = current_I_h_fine + error_estimate_fine

    return error_estimate_fine, I_уточненное


# --- Streamlit UI для пункта 3.5 ---
def section_3_5():
    st.header("3.5. Численное интегрирование и оценка погрешности")

    st.sidebar.subheader("Настройки для пункта 3.5")

    # Предустановленные значения
    default_func_str = "x / ((2*x + 7) * (3*x + 4))"
    default_x0 = -1.0
    default_xk = 1.0
    default_h1 = 0.5
    default_h2 = 0.25

    # Ввод данных
    st.session_state.s35_func_str = st.sidebar.text_input(
        "Функция y=f(x) (используйте 'x' как переменную):",
        value=st.session_state.get("s35_func_str", default_func_str),
        key="s35_func_str_input"
    )
    cols_limits = st.sidebar.columns(2)
    st.session_state.s35_x0 = cols_limits[0].number_input(
        "$X_0$ (начало):", value=st.session_state.get("s35_x0", default_x0), format="%.4f", step=0.1, key="s35_x0_input"
    )
    st.session_state.s35_xk = cols_limits[1].number_input(
        "$X_k$ (конец):", value=st.session_state.get("s35_xk", default_xk), format="%.4f", step=0.1, key="s35_xk_input"
    )

    cols_steps = st.sidebar.columns(2)
    st.session_state.s35_h1 = cols_steps[0].number_input(
        "Шаг $h_1$:", value=st.session_state.get("s35_h1", default_h1), format="%.4f", step=0.01, min_value=1e-6,
        key="s35_h1_input"
    )
    st.session_state.s35_h2 = cols_steps[1].number_input(
        "Шаг $h_2$ (меньше $h_1$):", value=st.session_state.get("s35_h2", default_h2), format="%.4f", step=0.01,
        min_value=1e-6, key="s35_h2_input"
    )

    # Точное значение интеграла (если возможно)
    x_s = Symbol('x')
    analytical_integral_val = None
    analytical_integral_expr = None
    try:
        func_to_integrate_sympy, expr_sympy = user_function_wrapper(st.session_state.s35_func_str, x_s)
        # Попытка найти первообразную
        antiderivative = sympy_integrate(expr_sympy, x_s)
        if not antiderivative.has(sympy_integrate):  # Если интегрирование удалось (нет знака интеграла)
            analytical_integral_val = (antiderivative.subs(x_s, st.session_state.s35_xk) -
                                       antiderivative.subs(x_s, st.session_state.s35_x0)).evalf()
            analytical_integral_expr = antiderivative
    except Exception:  # Ошибка в функции или не удалось проинтегрировать
        analytical_integral_val = None  # Не можем найти точное значение
        analytical_integral_expr = None

    if st.button("🧮 Вычислить интеграл и оценить погрешность", key="s35_run_button"):
        try:
            current_x0 = float(st.session_state.s35_x0)
            current_xk = float(st.session_state.s35_xk)
            current_h1 = float(st.session_state.s35_h1)
            current_h2 = float(st.session_state.s35_h2)
            current_func_str = st.session_state.s35_func_str

            if current_x0 >= current_xk:
                st.error("$X_0$ должен быть меньше $X_k$.")
                return
            if current_h1 <= 0 or current_h2 <= 0:
                st.error("Шаги $h_1$ и $h_2$ должны быть положительными.")
                return
            if np.isclose(current_h1, current_h2):
                st.warning("Шаги $h_1$ и $h_2$ очень близки. Оценка по Рунге-Ромбергу может быть неточной.")
            # Гарантируем, что h1 - больший шаг для удобства далее (хотя функция Рунге это обрабатывает)
            # h_large = max(current_h1, current_h2)
            # h_small = min(current_h1, current_h2)

            x_sym = Symbol('x')  # Символ для lambdify
            func_callable, sympy_expr = user_function_wrapper(current_func_str, x_sym)

            st.markdown("---")
            st.subheader(
                f"Результаты интегрирования $F = \\int_{{{current_x0}}}^{{{current_xk}}} ({sympy.latex(sympy_expr)}) \\, dx$")

            results_data = []
            methods_info = {
                "Прямоугольники (средние)": {"func": method_rectangles_midpoint, "p": 2},
                "Трапеции": {"func": method_trapezoids, "p": 2},
                "Симпсона": {"func": method_simpson, "p": 4},
            }

            for method_name, info in methods_info.items():
                integrator_func = info["func"]
                p_order = info["p"]

                st.markdown(f"#### Метод: {method_name} (порядок $p={p_order}$)")

                # Расчет для h1
                N1 = int(round((current_xk - current_x0) / current_h1))
                if N1 == 0: N1 = 1  # Минимум 1 интервал
                if method_name == "Симпсона" and N1 % 2 != 0:
                    N1 += 1  # Гарантируем четное число для Симпсона
                xs1, h_actual1 = get_integration_grid(current_x0, current_xk, N1)
                try:
                    I_h1 = integrator_func(func_callable, xs1, h_actual1)
                except ValueError as e_int:
                    st.error(
                        f"Ошибка при вычислении интеграла ({method_name}, $h_1={current_h1:.4f}$ -> $h_{{act1}}={h_actual1:.4f}$, $N_1={N1}$): {e_int}")
                    continue  # Пропускаем этот метод, если ошибка

                # Расчет для h2
                N2 = int(round((current_xk - current_x0) / current_h2))
                if N2 == 0: N2 = 1
                if method_name == "Симпсона" and N2 % 2 != 0:
                    N2 += 1
                xs2, h_actual2 = get_integration_grid(current_x0, current_xk, N2)
                try:
                    I_h2 = integrator_func(func_callable, xs2, h_actual2)
                except ValueError as e_int:
                    st.error(
                        f"Ошибка при вычислении интеграла ({method_name}, $h_2={current_h2:.4f}$ -> $h_{{act2}}={h_actual2:.4f}$, $N_2={N2}$): {e_int}")
                    continue

                # Оценка по Рунге-Ромбергу
                try:
                    # Передаем ФАКТИЧЕСКИЕ шаги, использованные при расчете
                    error_R, I_уточненное_R = runge_romberg_error_estimation(I_h1, I_h2, h_actual1, h_actual2, p_order)
                    runge_info = f"Погрешность (Рунге): {error_R:.3e}, Уточненное значение: {I_уточненное_R:.7f}"
                except ValueError as e_runge:
                    error_R, I_уточненное_R = None, None
                    runge_info = f"Ошибка Рунге: {e_runge}"

                results_data.append({
                    "Метод": method_name,
                    "$I(h_1)$ ($h_1 \approx {}$) ".format(f"{h_actual1:.4f}, N_1={N1}"): f"{I_h1:.7f}",
                    "$I(h_2)$ ($h_2 \approx {}$) ".format(f"{h_actual2:.4f}, N_2={N2}"): f"{I_h2:.7f}",
                    "Рунге-Ромберг": runge_info
                })

            if results_data:
                df_results = pd.DataFrame(results_data)
                st.dataframe(df_results)

            if analytical_integral_val is not None:
                st.success(
                    f"**Аналитическое (точное) значение интеграла: $F_{{точно}} \\approx {float(analytical_integral_val):.7f}$**")
                st.markdown(f"Первообразная $F(x) = {sympy.latex(analytical_integral_expr)}$ (без константы $C$) ")
                # Можно добавить сравнение с точным значением, если оно есть
                if I_уточненное_R is not None and results_data:  # Если есть хоть одно уточненное значение
                    last_method_name = results_data[-1]["Метод"]
                    abs_err_runge_vs_true = abs(float(analytical_integral_val) - I_уточненное_R)
                    st.info(
                        f"Абсолютная погрешность уточненного значения (для {last_method_name}) по сравнению с точным: {abs_err_runge_vs_true:.3e}")
            else:
                st.info(
                    "Точное аналитическое значение интеграла для данной функции не было найдено или не вычислялось.")

            # --- Визуализация (опционально) ---
            # Можно нарисовать график функции и выделить область интегрирования
            st.markdown("---")
            st.markdown("#### График функции и область интегрирования")

            plot_xs_dense = np.linspace(current_x0, current_xk, 300)
            try:
                plot_ys_dense = np.array([func_callable(x_val) for x_val in plot_xs_dense])
            except Exception as e_plot:
                st.warning(f"Не удалось построить график функции: {e_plot}")
                plot_ys_dense = np.full_like(plot_xs_dense, np.nan)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_xs_dense, y=plot_ys_dense, mode='lines', name='$f(x)$'))
            fig.add_vrect(x0=current_x0, x1=current_xk,
                          fillcolor="rgba(0,100,80,0.2)", layer="below", line_width=0,
                          name="Область интегрирования")
            fig.update_layout(title=f"График функции $y = {sympy.latex(sympy_expr)}$",
                              xaxis_title="x", yaxis_title="y", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- Теоретические остаточные члены ---
            st.markdown("---")
            st.markdown("#### Теоретические оценки погрешности (остаточные члены $R_N$)")
            st.markdown(
                "Для равномерной сетки с шагом $h$ и $N$ интервалами на отрезке $[a,b]$:"
                "<ul>"
                "<li><b>Метод прямоугольников (средних):</b> $R_N = -\\frac{(b-a)h^2}{24} f''(\\xi)$, $\\xi \\in [a,b]$. Порядок $O(h^2)$.</li>"
                "<li><b>Метод трапеций:</b> $R_N = -\\frac{(b-a)h^2}{12} f''(\\xi)$, $\\xi \\in [a,b]$. Порядок $O(h^2)$.</li>"
                "<li><b>Метод Симпсона:</b> $R_N = -\\frac{(b-a)h^4}{180} f^{(4)}(\\xi)$, $\\xi \\in [a,b]$. Порядок $O(h^4)$.</li>"
                "</ul>"
                "Нахождение $\\max|f''(x)|$ и $\\max|f^{(4)}(x)|$ на отрезке $[a,b]$ (мажорантная оценка) "
                "может быть сложной аналитической задачей.", unsafe_allow_html=True
            )


        except ValueError as ve:
            st.error(f"🚫 Ошибка значения: {ve}")
        except TypeError as te:  # Например, если sympy не может обработать выражение
            st.error(f"🚫 Ошибка типа при обработке функции или данных: {te}")
        except Exception as e:
            st.error(f"💥 Произошла непредвиденная ошибка: {e}")
            st.code(traceback.format_exc())


# --- Запуск ---
st.markdown(r"""
<style>
.stRadio[role=radiogroup] { flex-direction: row; gap: 15px; }
.stRadio[role=radiogroup] > label { margin-right: 0; }
</style>
""", unsafe_allow_html=True)
st.title("🚀 Лабораторная работа по численным методам")
st.markdown("---")
section_3_5()
