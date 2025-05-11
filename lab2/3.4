import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import traceback
import collections


# --- Функции для численного дифференцирования ---

def check_constant_step(xs, tolerance=1e-9):
    """
    Проверяет, является ли сетка узлов xs равномерной.
    Возвращает:
        float: Значение постоянного шага, если сетка равномерная.
        None: Если сетка неравномерная или точек меньше 2.
    """
    if not isinstance(xs, (list, np.ndarray)): xs = np.array(xs) # Преобразуем в numpy массив, если это список
    if len(xs) < 2: # Для шага нужно как минимум 2 точки
        return None
    steps = np.diff(xs) # Вычисляет разности между соседними элементами (шаги)
    if len(steps) == 0: # Если всего одна точка после diff (т.е. исходно 2 точки)
        return None

    first_step = steps[0] # Берем первый шаг как эталон

    # Проверка, не являются ли все шаги нулевыми (т.е. все точки совпадают)
    if np.all(np.abs(steps) < tolerance):
        # Эта ситуация должна быть отловлена ранее проверкой уникальности узлов.
        raise ValueError("Все узлы совпадают или шаги между ними нулевые. Проверьте уникальность и возрастание узлов X.")

    # np.allclose проверяет, что все элементы массива steps близки к first_step с заданной абсолютной погрешностью tolerance.
    if np.allclose(steps, first_step, atol=tolerance):
        if abs(first_step) < tolerance: # Проверяем, не является ли сам постоянный шаг нулевым
            # Эта ситуация также маловероятна, если предыдущая проверка на нулевые шаги сработала.
            raise ValueError("Шаг сетки является нулевым, что указывает на совпадающие точки.")
        return first_step  # Возвращаем значение постоянного шага
    return None  # Сетка неравномерная


def calculate_derivatives_at_point(X_star_val, xs_nodes, ys_nodes, tolerance=1e-9):
    """
    Вычисляет первую и вторую производную функции, заданной таблично (xs_nodes, ys_nodes),
    в точке X_star_val. Использует различные формулы в зависимости от того,
    является ли X_star узлом, и является ли сетка равномерной.
    """
    n = len(xs_nodes) # Количество узлов
    # --- 0. Предварительные проверки и подготовка данных ---
    if n != len(ys_nodes):
        raise ValueError("Длины массивов X и Y должны совпадать.")
    if n < 2: # Для вычисления производных нужно как минимум 2 точки
        raise ValueError("Для вычисления производных необходимо как минимум 2 точки.")

    # Гарантируем, что работаем с numpy массивами типа float
    xs = np.array(xs_nodes, dtype=float)
    ys = np.array(ys_nodes, dtype=float)
    X_star = float(X_star_val) # Убедимся, что X_star это float

    # Сортировка узлов X по возрастанию и соответствующая перестановка Y.
    # Это критически важно для правильной работы формул и поиска интервалов.
    # np.all(np.diff(xs) > -tolerance) проверяет, что массив уже почти отсортирован (не убывает)
    # Если он не отсортирован или содержит почти равные элементы, которые нарушают порядок, сортируем.
    if not np.all(np.diff(xs) > -tolerance):
        sorted_indices = np.argsort(xs)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]
        # st.warning("Данные были отсортированы по X для корректной работы.") # Это сообщение лучше выводить в UI

    # После сортировки проверяем на строгое возрастание и уникальность узлов X.
    # np.diff(xs) > tolerance означает, что x[i+1] - x[i] должно быть больше tolerance.
    if not np.all(np.diff(xs) > tolerance):
        # Находим проблемные пары (слишком близкие или неупорядоченные узлы)
        problem_indices = [i for i in range(n - 1) if not (xs[i+1] - xs[i] > tolerance)]
        problem_pairs = [(xs[i], xs[i+1]) for i in problem_indices]
        raise ValueError(f"Узлы X должны быть строго возрастающими и различными. "
                         f"Обнаружены проблемы в парах: {problem_pairs}")

    # Проверка, находится ли X_star в допустимом диапазоне (между первым и последним узлом).
    # Экстраполяция не поддерживается для простоты.
    if X_star < xs[0] - tolerance or X_star > xs[-1] + tolerance:
        raise ValueError(f"Точка X*={X_star:.4f} вне диапазона данных [{xs[0]:.4f}, {xs[-1]:.4f}]. "
                         "Вычисление производной вне диапазона не поддерживается.")

    # Проверяем, является ли сетка узлов X равномерной
    h_const = check_constant_step(xs, tolerance)

    # Инициализация словаря для хранения результатов
    results = {
        "X_star_calc": X_star, "is_node": False, "node_index": -1, "h_const": h_const,
        "f_prime": None, "f_prime_method": "N/A", "f_prime_order": "N/A", "f_prime_nodes_indices": [],
        "f_double_prime": None, "f_double_prime_method": "N/A", "f_double_prime_order": "N/A", "f_double_prime_nodes_indices": []
    }

    # --- 2. Проверяем, совпадает ли X_star с одним из узлов сетки ---
    # np.isclose сравнивает X_star с каждым элементом xs с учетом погрешности tolerance.
    # np.where возвращает индексы элементов, для которых условие истинно.
    match_indices = np.where(np.isclose(xs, X_star, atol=tolerance))[0]
    is_node_calc = len(match_indices) > 0 # True, если найдено совпадение

    if is_node_calc: # --- Случай 1: X_star является узлом сетки ---
        k = match_indices[0] # Индекс узла, с которым совпал X_star
        results["is_node"] = True
        results["node_index"] = k
        results["X_star_calc"] = xs[k]  # Уточняем X_star до значения узла для точности

        # === Первая производная f'(x_k) в узле x_k ===
        if h_const is not None:  # Сетка РАВНОМЕРНАЯ с шагом h_const
            h = h_const
            if k == 0:  # X_star - левый крайний узел (x_0)
                if n >= 3:  # Можно использовать трехточечную формулу O(h^2)
                    results["f_prime"] = (-3*ys[0] + 4*ys[1] - ys[2]) / (2*h) # Правая разность 2-го порядка
                    results["f_prime_method"] = "Правая трехточечная (равн.)"
                    results["f_prime_order"] = "O(h^2)"
                    results["f_prime_nodes_indices"] = [0, 1, 2]
                elif n == 2: # Всего 2 точки, используем двухточечную O(h)
                    results["f_prime"] = (ys[1] - ys[0]) / h # Правая разность 1-го порядка
                    results["f_prime_method"] = "Правая двухточечная (равн.)"
                    results["f_prime_order"] = "O(h)"
                    results["f_prime_nodes_indices"] = [0, 1]
            elif k == n - 1:  # X_star - правый крайний узел (x_{n-1})
                if n >= 3: # Трехточечная O(h^2)
                    results["f_prime"] = (3*ys[n-1] - 4*ys[n-2] + ys[n-3]) / (2*h) # Левая разность 2-го порядка
                    results["f_prime_method"] = "Левая трехточечная (равн.)"
                    results["f_prime_order"] = "O(h^2)"
                    results["f_prime_nodes_indices"] = [n-3, n-2, n-1]
                elif n == 2: # Двухточечная O(h)
                    results["f_prime"] = (ys[n-1] - ys[n-2]) / h # Левая разность 1-го порядка
                    results["f_prime_method"] = "Левая двухточечная (равн.)"
                    results["f_prime_order"] = "O(h)"
                    results["f_prime_nodes_indices"] = [n-2, n-1]
            else:  # X_star - внутренний узел (0 < k < n-1)
                results["f_prime"] = (ys[k+1] - ys[k-1]) / (2*h) # Центральная разность O(h^2)
                results["f_prime_method"] = "Центральная двухточечная (равн.)"
                results["f_prime_order"] = "O(h^2)"
                results["f_prime_nodes_indices"] = [k-1, k, k+1] # Используются y[k-1], y[k+1]
        else:  # Сетка НЕРАВНОМЕРНАЯ, X_star = xs[k] (узел)
            if k == 0 and n >= 2:  # Левый край
                h0 = xs[1] - xs[0]
                results["f_prime"] = (ys[1] - ys[0]) / h0 # Правая разность
                results["f_prime_method"] = "Правая двухточечная (неравн.)"
                results["f_prime_order"] = f"O({h0:.2f})" # Порядок зависит от конкретного шага
                results["f_prime_nodes_indices"] = [0, 1]
            elif k == n - 1 and n >= 2:  # Правый край
                h_prev = xs[n-1] - xs[n-2]
                results["f_prime"] = (ys[n-1] - ys[n-2]) / h_prev # Левая разность
                results["f_prime_method"] = "Левая двухточечная (неравн.)"
                results["f_prime_order"] = f"O({h_prev:.2f})"
                results["f_prime_nodes_indices"] = [n-2, n-1]
            elif 0 < k < n - 1 and n >= 3:  # Внутренний узел, используем трехточечную формулу для неравномерной сетки
                h_L = xs[k] - xs[k-1] # Шаг слева
                h_R = xs[k+1] - xs[k] # Шаг справа
                # Формула получается из дифференцирования интерполяционного полинома Лагранжа 2-й степени, построенного по точкам (k-1, k, k+1), и вычисления производной в точке x_k.
                term_k_minus_1 = ys[k-1] * (-h_R) / (h_L * (h_L + h_R))
                term_k         = ys[k]   * (h_R - h_L) / (h_L * h_R)
                term_k_plus_1  = ys[k+1] * h_L / (h_R * (h_L + h_R))
                results["f_prime"] = term_k_minus_1 + term_k + term_k_plus_1
                results["f_prime_method"] = "Центральная (3 точки, неравн.)"
                results["f_prime_order"] = "O(h_avg)"  # Порядок точности сложнее оценить, зависит от соотношения шагов
                results["f_prime_nodes_indices"] = [k-1, k, k+1]
            else:  # Недостаточно точек для выбранного метода
                results["f_prime_method"] = "Недостаточно точек для производной в узле (неравн.)"

        # === Вторая производная f''(x_k) в узле x_k ===
        # Обычно требует как минимум 3 точки.
        if h_const is not None and 0 < k < n - 1 and n >= 3:  # Равномерная сетка, внутренний узел
            h = h_const
            results["f_double_prime"] = (ys[k+1] - 2*ys[k] + ys[k-1]) / (h**2) # Центральная разность для второй производной
            results["f_double_prime_method"] = "Центральная трехточечная (равн.)"
            results["f_double_prime_order"] = "O(h^2)"
            results["f_double_prime_nodes_indices"] = [k-1, k, k+1]
        elif not h_const and 0 < k < n - 1 and n >= 3:  # Неравномерная сетка, внутренний узел
            h_L = xs[k] - xs[k-1]
            h_R = xs[k+1] - xs[k]
            # Формула из дифференцирования интерполяционного полинома Лагранжа 2-й степени P2''(xk)
            term_R_div = (ys[k+1] - ys[k]) / h_R
            term_L_div = (ys[k] - ys[k-1]) / h_L
            results["f_double_prime"] = 2 * (term_R_div - term_L_div) / (h_R + h_L)
            results["f_double_prime_method"] = "Центральная (3 точки, неравн.)"
            results["f_double_prime_order"] = "O(h_avg)"
            results["f_double_prime_nodes_indices"] = [k-1, k, k+1]
        else:  # Крайние точки или менее 3 точек - вторая производная в узле не вычисляется этими методами.
            results["f_double_prime_method"] = "Не вычисляется (край узла / <3 точек / неравн. край)"

    else:  # --- Случай 2: X_star НЕ является узлом сетки (лежит между узлами) ---
        # Используем интерполяционный полином 2-й степени (по трем точкам) и дифференцируем его.
        results["f_prime_method"] = "Интерполяционная формула (3 точки)"
        results["f_double_prime_method"] = "Интерполяционная формула (3 точки)"
        # Порядок точности этих формул зависит от того, насколько хорошо квадратичная интерполяция приближает функцию.
        results["f_prime_order"] = "O(h_avg^2)" if h_const else "O(h_avg)" # Примерная оценка
        results["f_double_prime_order"] = "O(h_avg)" if h_const else "O(1)" # Примерная оценка

        # Находим индекс левого узла 'idx_left' такой, что xs[idx_left] <= X_star < xs[idx_left+1].
        # np.searchsorted(xs, X_star, side='right') вернет индекс, куда X_star можно вставить.
        # Вычитание 1 дает левый индекс.
        idx_left = np.searchsorted(xs, X_star, side='right') - 1
        # np.clip ограничивает значение idx_left диапазоном [0, n-2], чтобы избежать выхода за пределы массива
        # при выборе шаблона из 3-х точек. n-2 - это максимально возможный левый индекс для шаблона из 3 точек.
        idx_left = np.clip(idx_left, 0, n - 2)

        # Логика выбора шаблона из 3-х точек [x_j, x_{j+1}, x_{j+2}] для построения интерполяционного полинома.
        # Цель - выбрать шаблон так, чтобы X_star находился "внутри" этого шаблона.
        j = -1 # Индекс первой точки шаблона
        if n == 2:  # Если всего 2 точки (xs[0], xs[1]), X_star между ними
            h_seg = xs[1] - xs[0]
            results["f_prime"] = (ys[1] - ys[0]) / h_seg # Производная линейной функции
            results["f_double_prime"] = 0.0  # Вторая производная линейной функции равна 0
            results["f_prime_method"] = "Линейная интерполяция (2 точки)"
            results["f_prime_order"] = f"O({h_seg:.2f})"
            results["f_double_prime_method"] = "Линейная интерполяция (вторая пр-я = 0)"
            results["f_double_prime_order"] = "N/A"
            results["f_prime_nodes_indices"] = [0, 1]
            results["f_double_prime_nodes_indices"] = [0, 1]
            return results  # Завершаем для случая 2-х точек

        # Если точек >= 3
        # idx_left - это индекс i такой, что x_i <= X_star < x_{i+1}
        if idx_left == 0:  # X_star между x0 и x1. Шаблон: x0, x1, x2. j=0.
            j = 0
        elif idx_left == n - 2:  # X_star между x_{n-2} и x_{n-1} (предпоследний и последний узлы).
                                # Шаблон: x_{n-3}, x_{n-2}, x_{n-1}. j=n-3.
            j = n - 3
        elif 0 < idx_left < n - 2:  # X_star где-то в середине, есть выбор из двух шаблонов.
            # Шаблон 1: xs[idx_left-1], xs[idx_left], xs[idx_left+1]  (j = idx_left-1)
            # Шаблон 2: xs[idx_left],   xs[idx_left+1], xs[idx_left+2] (j = idx_left)
            # Выбираем тот шаблон, для которого X_star ближе к средней точке интервала, образованного крайними точками шаблона.
            mid_tpl1 = (xs[idx_left - 1] + xs[idx_left + 1]) / 2 # Середина первого возможного шаблона
            mid_tpl2 = (xs[idx_left] + xs[idx_left + 2]) / 2     # Середина второго возможного шаблона
            if abs(X_star - mid_tpl1) <= abs(X_star - mid_tpl2): # Если X_star ближе к середине первого шаблона
                j = idx_left - 1
            else: # Иначе выбираем второй шаблон
                j = idx_left
        # Если n=3, то idx_left может быть только 0. Тогда j=0.
        # Это покрывается первым if.

        if j != -1:  # Если подходящий 3-точечный шаблон был выбран
            x_tpl = xs[j : j+3] # Узлы X шаблона
            y_tpl = ys[j : j+3] # Значения Y шаблона
            results["f_prime_nodes_indices"] = list(range(j, j+3))
            results["f_double_prime_nodes_indices"] = list(range(j, j+3))

            # Вычисляем разделенные разности для интерполяционного полинома Ньютона 2-й степени
            # P2(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1)
            # где x0, x1, x2 - это x_tpl[0], x_tpl[1], x_tpl[2]
            try:
                f01 = (y_tpl[1] - y_tpl[0]) / (x_tpl[1] - x_tpl[0]) # f[x0, x1]
                f12 = (y_tpl[2] - y_tpl[1]) / (x_tpl[2] - x_tpl[1]) # f[x1, x2]
                if np.isclose(x_tpl[2] - x_tpl[0], 0): raise ZeroDivisionError("Знаменатель f012 близок к нулю")
                f012 = (f12 - f01) / (x_tpl[2] - x_tpl[0])      # f[x0, x1, x2]
            except ZeroDivisionError: # Если узлы шаблона совпадают (не должно быть из-за проверок выше)
                raise ValueError("Деление на ноль при вычислении разделенных разностей. Проверьте узлы шаблона.")

            # Производные от P2(x):
            # P'_2(x) = f[x0,x1] + f[x0,x1,x2] * ( (x-x_tpl[0]) + (x-x_tpl[1]) )
            results["f_prime"] = f01 + f012 * ( (X_star - x_tpl[0]) + (X_star - x_tpl[1]) )
            # P''_2(x) = 2 * f[x0,x1,x2]
            results["f_double_prime"] = 2 * f012
        else: # Если не удалось выбрать 3-точечный шаблон (например, если n < 3, но этот случай обрабатывается ранее)
            results["f_prime_method"] = "Не удалось выбрать 3-точечный шаблон для интерполяции."
            results["f_double_prime_method"] = "Не удалось выбрать 3-точечный шаблон для интерполяции."
    return results


# --- Streamlit UI для пункта 3.4 ---
# (Комментарии к UI-части аналогичны предыдущим разборам, фокусируясь на специфике дифференцирования)
def section_3_4():
    st.header("3.4. Численное дифференцирование")
    st.sidebar.subheader("Настройки для пункта 3.4")

    # Предустановленные значения
    X_i_default_s34 = np.array([-0.2, 0.0, 0.2, 0.4, 0.6])
    Y_i_default_s34 = np.array([-0.20136, 0.0, 0.20136, 0.41152, 0.64350]) # Данные из примера
    X_star_default_s34 = 0.2

    # Опции выбора источника данных
    data_source_options_s34 = ["Предустановка (5 узлов, X* = 0.2)", "Пользовательский ввод"]
    if 's34_data_source' not in st.session_state: st.session_state.s34_data_source = data_source_options_s34[0]
    st.session_state.s34_data_source = st.sidebar.radio("Источник табличных данных:", data_source_options_s34,
        index=data_source_options_s34.index(st.session_state.s34_data_source), key="s34_data_source_radio")

    # Инициализация и установка значений для полей ввода
    if 's34_x_nodes_str' not in st.session_state: st.session_state.s34_x_nodes_str = ", ".join(map(str, X_i_default_s34))
    if 's34_y_nodes_str' not in st.session_state: st.session_state.s34_y_nodes_str = ", ".join(map(lambda x: f"{x:.5f}", Y_i_default_s34))
    if 's34_x_star_str' not in st.session_state: st.session_state.s34_x_star_str = str(X_star_default_s34)

    if st.session_state.s34_data_source == data_source_options_s34[0]: # Предустановка
        st.session_state.s34_x_nodes_str = ", ".join(map(str, X_i_default_s34))
        st.session_state.s34_y_nodes_str = ", ".join(map(lambda x: f"{x:.5f}", Y_i_default_s34))
        st.session_state.s34_x_star_str = str(X_star_default_s34)
        is_editable_s34 = False
    else: # Пользовательский ввод
        is_editable_s34 = True

    st.subheader("Входные данные и точка вычисления")
    # Поля для ввода X_i, Y_i, X_star
    cols_s34_1 = st.columns(2)
    with cols_s34_1[0]:
        st.session_state.s34_x_nodes_str = st.text_input("Узлы $X_i$ (через запятую, мин. 2 точки):",
            value=st.session_state.s34_x_nodes_str, disabled=not is_editable_s34, key="s34_x_nodes_input")
    with cols_s34_1[1]:
        st.session_state.s34_y_nodes_str = st.text_input("Значения $Y_i$ (соответствуют $X_i$):",
            value=st.session_state.s34_y_nodes_str, disabled=not is_editable_s34, key="s34_y_nodes_input")
    st.session_state.s34_x_star_str = st.text_input("Точка $X^*$ для вычисления производных:",
        value=st.session_state.s34_x_star_str, key="s34_x_star_input")

    # Кнопка для запуска вычислений
    if st.button("📈 Вычислить производные", key="s34_run_button"):
        try:
            # Парсинг входных данных
            x_nodes_str = st.session_state.s34_x_nodes_str
            y_nodes_str = st.session_state.s34_y_nodes_str
            x_star_str = st.session_state.s34_x_star_str
            current_X_nodes = np.array([float(x.strip()) for x in x_nodes_str.split(',')])
            current_Y_nodes = np.array([float(y.strip()) for y in y_nodes_str.split(',')])
            current_X_star = float(x_star_str)

            # Для отображения отсортируем данные (функция calculate_derivatives_at_point сделает это внутри себя)
            sorted_indices_display = np.argsort(current_X_nodes)
            display_X_nodes = current_X_nodes[sorted_indices_display]
            display_Y_nodes = current_Y_nodes[sorted_indices_display]

            st.markdown("---")
            st.subheader("📊 Результаты численного дифференцирования")
            st.markdown(f"Для точки $X^* = {current_X_star:.4f}$")
            # Отображение входных (отсортированных для вывода) данных
            st.markdown("#### Входные данные (отсортированные по $X_i$):")
            df_input_s34 = pd.DataFrame({'i': range(len(display_X_nodes)), 'X_i': display_X_nodes, 'Y_i': display_Y_nodes})
            st.dataframe(df_input_s34.style.format({'X_i': "{:.4f}", 'Y_i': "{:.5f}"}))

            # Вызов основной функции для вычисления производных
            derivative_results = calculate_derivatives_at_point(current_X_star, current_X_nodes, current_Y_nodes)

            # Обработка возможной ошибки из функции (хотя она бросает ValueError)
            if derivative_results.get("error"): st.error(f"Ошибка: {derivative_results['error']}"); return

            # Вывод результатов для первой производной
            st.markdown("#### Первая производная $f'(X^*)$")
            if derivative_results["f_prime"] is not None:
                st.success(f"**$f'(X^*) \\approx {derivative_results['f_prime']:.7f}$**")
                st.markdown(f"*Метод: {derivative_results['f_prime_method']}*")
                st.markdown(f"*Порядок точности: {derivative_results['f_prime_order']}*")
                if derivative_results['f_prime_nodes_indices']: # Узлы, использованные для вычисления
                    st.markdown(f"*Использованы узлы с индексами: {derivative_results['f_prime_nodes_indices']} "
                                f"($X = [{', '.join([f'{display_X_nodes[i]:.2f}' for i in derivative_results['f_prime_nodes_indices']])}]$)*")
            else:
                st.warning(f"Не удалось вычислить первую производную. Причина: {derivative_results['f_prime_method']}")

            # Вывод результатов для второй производной
            st.markdown("#### Вторая производная $f''(X^*)$")
            if derivative_results["f_double_prime"] is not None:
                st.success(f"**$f''(X^*) \\approx {derivative_results['f_double_prime']:.7f}$**")
                st.markdown(f"*Метод: {derivative_results['f_double_prime_method']}*")
                st.markdown(f"*Порядок точности: {derivative_results['f_double_prime_order']}*")
                if derivative_results['f_double_prime_nodes_indices']: # Узлы, использованные для вычисления
                    st.markdown(f"*Использованы узлы с индексами: {derivative_results['f_double_prime_nodes_indices']} "
                                f"($X = [{', '.join([f'{display_X_nodes[i]:.2f}' for i in derivative_results['f_double_prime_nodes_indices']])}]$)*")
            else:
                st.warning(f"Не удалось вычислить вторую производную. Причина: {derivative_results['f_double_prime_method']}")

            st.markdown("---")
            st.markdown("##### Дополнительная информация о расчетах:")
            if derivative_results["is_node"]: st.info(f"Точка $X^*={derivative_results['X_star_calc']:.4f}$ совпадает с узлом $x_{{{derivative_results['node_index']}}}$.")
            else: st.info(f"Точка $X^*={derivative_results['X_star_calc']:.4f}$ находится между узлами.")
            if derivative_results["h_const"] is not None: st.info(f"Обнаружена равномерная сетка с шагом $h = {derivative_results['h_const']:.4f}$.")
            else: st.info("Обнаружена неравномерная сетка.")

            # --- График ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=display_X_nodes, y=display_Y_nodes, mode='lines+markers', name='Данные $f(x_i)$'))
            # Отмечаем точку X*
            fig.add_trace(go.Scatter(x=[derivative_results["X_star_calc"]],
                y=[np.interp(derivative_results["X_star_calc"], display_X_nodes, display_Y_nodes)], # Интерполируем y для X* для графика
                mode='markers', name='$X^*$', marker=dict(color='red', size=12, symbol='x')))
            # Отмечаем узлы, использованные для первой производной
            if derivative_results.get("f_prime_nodes_indices"):
                prime_nodes_x = [display_X_nodes[i] for i in derivative_results["f_prime_nodes_indices"]]
                prime_nodes_y = [display_Y_nodes[i] for i in derivative_results["f_prime_nodes_indices"]]
                fig.add_trace(go.Scatter(x=prime_nodes_x, y=prime_nodes_y, mode='markers', name='Узлы для $f\'(X^*)$',
                    marker=dict(color='rgba(255,165,0,0.7)', size=10, symbol='circle-open', line=dict(width=2))))
            # Отмечаем узлы, использованные для второй производной (если они другие)
            if derivative_results.get("f_double_prime_nodes_indices") and derivative_results["f_double_prime_nodes_indices"] != derivative_results.get("f_prime_nodes_indices"):
                double_prime_nodes_x = [display_X_nodes[i] for i in derivative_results["f_double_prime_nodes_indices"]]
                double_prime_nodes_y = [display_Y_nodes[i] for i in derivative_results["f_double_prime_nodes_indices"]]
                fig.add_trace(go.Scatter(x=double_prime_nodes_x, y=double_prime_nodes_y, mode='markers', name='Узлы для $f\'\'(X^*)$',
                    marker=dict(color='rgba(0,128,0,0.7)', size=10, symbol='diamond-open', line=dict(width=2))))
            fig.update_layout(title="Данные и точка вычисления производной", xaxis_title="X", yaxis_title="Y", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        # Блоки except для обработки ошибок
        except ValueError as ve: st.error(f"🚫 Ошибка значения: {ve}")
        except Exception as e: st.error(f"💥 Произошла непредвиденная ошибка: {e}"); st.code(traceback.format_exc())

# --- Запуск приложения ---
# (CSS и заголовок аналогичны предыдущим примерам)
st.markdown(r"""<style>.stRadio[role=radiogroup]{flex-direction:row;gap:15px;}.stRadio[role=radiogroup]>label{margin-right:0;}</style>""", unsafe_allow_html=True)
st.title("🚀 Лабораторная работа по численным методам")
st.markdown("---")
section_3_4()
