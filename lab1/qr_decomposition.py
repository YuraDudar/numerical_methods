import numpy as np
import pandas as pd
import streamlit as st
import time
import cmath # For complex number phase calculation and square roots

# --- FUNCTION IMPLEMENTATIONS ---
def householder_qr(A):
    """
    Performs QR decomposition of a matrix A using Householder reflections.
    Handles complex matrices.

    Args:
        A (np.ndarray): Input matrix (m x n).

    Returns:
        tuple: (Q, R) where Q is an (m x m) unitary matrix and R is an (m x n)
               upper triangular matrix. Returns Q and R as complex128 type.

    Raises:
        np.linalg.LinAlgError: Can potentially be raised if numerical issues occur,
                               although Householder is generally robust.
    """
    m, n = A.shape
    R = A.astype(np.complex128, copy=True) # Work on a copy with complex type
    Q = np.eye(m, dtype=np.complex128)
    tiny = np.finfo(np.complex128).eps # Small threshold for checks

    # Loop over columns to introduce zeros below the diagonal
    for j in range(min(m - 1, n)): # No need to process the last column for R, or last row for Q
        # Extract the vector to be reflected (column j from row j downwards)
        # Ensure x is treated as a column vector (k x 1)
        x = R[j:, j].reshape(-1, 1)
        k = x.shape[0] # Size of the sub-vector

        if k <= 1: # Nothing to reflect if sub-vector has 0 or 1 element
            continue

        # Calculate the norm of the vector x
        norm_x = np.linalg.norm(x)

        # If the norm is very small, the column is already essentially zero below R[j,j]
        if norm_x < tiny:
             continue

        # Calculate the Householder vector v
        x0 = x[0, 0]

        # Determine the phase/sign factor 's'
        if abs(x0) < tiny:
            s = 1.0
        else:
            s = x0 / abs(x0)

        alpha = -s * norm_x
        u = x.copy()
        u[0, 0] -= alpha

        u_norm_sq = np.vdot(u, u).real

        if u_norm_sq < tiny:
            continue

        beta = 2.0 / u_norm_sq

        # Apply reflection P = I - beta * u * u^H efficiently
        # 1. Apply to R: R[j:, j:] = R[j:, j:] - beta * u @ (u^H @ R[j:, j:])
        sub_R = R[j:, j:]
        w = np.dot(u.conj().T, sub_R)
        R[j:, j:] -= beta * np.dot(u, w)

        # 2. Apply to Q: Q[:, j:] = Q[:, j:] - beta * (Q[:, j:] @ u) @ u^H
        sub_Q = Q[:, j:]
        z = np.dot(sub_Q, u)
        Q[:, j:] -= beta * np.dot(z, u.conj().T)

    return Q, R


def qr_eigenvalue_algorithm(A, epsilon, max_iterations):
    """
    Finds eigenvalues of a square matrix A using the basic QR algorithm
    with Householder QR decomposition. Handles real matrices with complex
    eigenvalues by extracting eigenvalues from the final quasi-triangular matrix.

    Args:
        A (np.ndarray): Input square matrix.
        epsilon (float): Convergence tolerance (sub-diagonal elements relative magnitude).
        max_iterations (int): Maximum number of iterations.

    Returns:
        tuple:
            - eigenvalues (np.ndarray): Array of computed eigenvalues (complex128),
                                        sorted by real part, then imaginary part.
            - iterations (int): Number of iterations performed.
            - converged (bool): True if the algorithm converged within max_iterations.
            - history (dict): Dictionary containing iteration numbers ('iter')
                              and sum of abs sub-diagonal norms ('sub_diag_norm').
            - elapsed_time (float): Execution time in seconds.

    Raises:
        ValueError: If input matrix is not square.
        np.linalg.LinAlgError: If QR decomposition fails during iterations.
    """
    start_time = time.time()
    tiny = np.finfo(np.complex128).eps

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")

    Ak = A.astype(np.complex128, copy=True)
    n = Ak.shape[0]
    converged = False
    history = {'iter': [], 'sub_diag_norm': []}
    iterations_done = 0

    # --- QR Algorithm Loop ---
    for k in range(max_iterations):
        iterations_done = k + 1
        A_to_decompose = Ak

        try:
            Qk, Rk = householder_qr(A_to_decompose)
        except np.linalg.LinAlgError as e:
             raise np.linalg.LinAlgError(f"QR decomposition failed at iteration {iterations_done}: {e}")

        Ak = Rk @ Qk

        # Check convergence based on sub-diagonal elements *relevant* to splitting
        # Use a relative tolerance to avoid issues with scaling
        sub_diag_norm_sum = 0.0 # Sum for history plot
        is_converging = True
        for i in range(n - 1):
            # Check if element Ak[i+1, i] is negligible compared to its diagonal neighbors
            sub_diag_val = abs(Ak[i+1, i])
            diag_sum = abs(Ak[i, i]) + abs(Ak[i+1, i+1])
            sub_diag_norm_sum += sub_diag_val # Track the sum for history plot

            # If diag_sum is very small, use absolute tolerance epsilon
            # Otherwise use relative tolerance
            # Convergence requires *all* relevant sub-diagonals to be small
            if sub_diag_val > epsilon * (diag_sum if diag_sum > tiny else 1.0):
                 is_converging = False
                 # No need to break, calculate full norm sum for history

        history['iter'].append(iterations_done)
        history['sub_diag_norm'].append(sub_diag_norm_sum) # Store the sum of *all* sub-diagonals

        # Declare convergence if *all* relevant sub-diagonals are small enough
        if is_converging:
            converged = True
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    # --- Extract Eigenvalues from the final matrix Ak (potentially quasi-triangular) ---
    eigenvalues_list = [] # Use a list to append eigenvalues
    i = 0
    while i < n:
        if i == n - 1:
            # Last element is always a 1x1 block (no element below it)
            eigenvalues_list.append(Ak[i, i])
            i += 1
        else:
            # Check the sub-diagonal element Ak[i+1, i] using relative tolerance
            sub_diag_val = abs(Ak[i+1, i])
            diag_sum = abs(Ak[i, i]) + abs(Ak[i+1, i+1])
            # Use a similar tolerance as in convergence check, perhaps slightly relaxed
            # A very small epsilon might sometimes prematurely split a 2x2 block
            relative_epsilon = max(epsilon * 10, tiny*100) # Heuristic adjustment
            if sub_diag_val < relative_epsilon * (diag_sum if diag_sum > tiny else 1.0):
                 # Treat Ak[i,i] as a 1x1 block (real eigenvalue)
                 eigenvalues_list.append(Ak[i, i])
                 i += 1
            else:
                # Sub-diagonal is not small -> treat as 2x2 block
                # Extract the 2x2 submatrix
                a = Ak[i, i]
                b = Ak[i, i+1]
                c = Ak[i+1, i]
                d = Ak[i+1, i+1]

                # Calculate eigenvalues of the 2x2 block using quadratic formula
                # λ^2 - tr*λ + det = 0
                trace = a + d
                determinant = a * d - b * c
                # Use cmath.sqrt to handle complex roots correctly from discriminant
                discriminant_sqrt = cmath.sqrt(trace**2 - 4 * determinant)

                eig1 = (trace + discriminant_sqrt) / 2.0
                eig2 = (trace - discriminant_sqrt) / 2.0

                eigenvalues_list.append(eig1)
                eigenvalues_list.append(eig2)
                i += 2 # Skip the next diagonal element as it's part of the block

    eigenvalues = np.array(eigenvalues_list, dtype=np.complex128)

    # Sort eigenvalues for consistent output and comparison
    sort_indices = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues_sorted = eigenvalues[sort_indices]

    # Sanity check: Ensure we found the correct number of eigenvalues
    if len(eigenvalues_sorted) != n:
        # This case should ideally not happen if logic is correct, but good to have a warning
        print(f"Warning: Found {len(eigenvalues_sorted)} eigenvalues, expected {n}. Final matrix form might be unexpected.")
        # You might want to pad with NaN or raise an error depending on requirements
        # For now, return what was found. Comparison with NumPy will show the discrepancy.

    return eigenvalues_sorted, iterations_done, converged, history, elapsed_time


# --- GUI --- (Provided by user, unchanged - except default matrix and description)
st.title("5. QR-алгоритм для собственных значений")
st.markdown("""
Нахождение собственных значений (**λ**) произвольной квадратной матрицы **A** с помощью QR-алгоритма. Метод итеративно применяет QR-разложение:
1.  **A₀ = A**
2.  **Aₖ = QₖRₖ** (QR-разложение)
3.  **Aₖ₊₁ = RₖQₖ**

В пределе матрица **Aₖ** стремится к **верхней квази-треугольной** форме (форма Шура для комплексных матриц, вещественная форма Шура для действительных). Диагональные элементы (для действительных СЗ) или собственные значения блоков 2x2 на диагонали (для комплексно-сопряженных пар СЗ) предельной матрицы являются собственными значениями **A**.

Алгоритм реализован с использованием **QR-разложения на основе отражений Хаусхолдера**, поддерживает **комплексные числа** и извлекает собственные значения из **квази-треугольной формы**.
""") # Updated description
st.divider()

# --- Функция ввода матрицы 3x3 и параметров ---
def input_matrix_3x3_qr():
    st.subheader("Ввод матрицы 3×3 и параметров")
    if 'A_qr_input' not in st.session_state:
        # Default changed to the problematic matrix for easy testing
        st.session_state['A_qr_input'] = np.array([
            [1.0, 3.0, 1.0],
            [1.0, 1.0, 4.0],
            [4.0, 3.0, 1.0]
        ])
    if 'epsilon_qr' not in st.session_state:
        st.session_state['epsilon_qr'] = 1e-7 # Tolerance for convergence check
    if 'max_iter_qr' not in st.session_state:
        st.session_state['max_iter_qr'] = 500
    if 'tiny' not in st.session_state:
        st.session_state['tiny'] = np.finfo(np.complex128).eps

    A = np.zeros((3, 3), dtype=float)

    cols_header = st.columns(3)
    headers = [f"Столбец {j+1}" for j in range(3)]
    for col, header in zip(cols_header, headers):
        col.markdown(f"**{header}**")

    input_container = st.container()
    with input_container:
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                A[i, j] = cols[j].number_input(
                    f"A[{i},{j}]", value=float(st.session_state['A_qr_input'][i, j]),
                    label_visibility="collapsed", key=f"A_qr_{i}_{j}", format="%f"
                )
    st.session_state['A_qr_input'] = A.copy()


    col_param1, col_param2 = st.columns(2)
    with col_param1:
        epsilon_input = st.number_input(
            "Требуемая точность (ε):",
            min_value=1e-15, max_value=1.0, value=st.session_state['epsilon_qr'], format="%.1e", step=1e-8,
            help="Критерий остановки: поддиагональные элементы малы относительно диагональных", # Updated help text
            key="epsilon_qr_input"
        )
        st.session_state['epsilon_qr'] = epsilon_input

    with col_param2:
        max_iter_input = st.number_input(
            "Максимальное число итераций:",
            min_value=1, max_value=10000, value=st.session_state['max_iter_qr'], step=50,
            help="Предотвращение бесконечного цикла.",
            key="max_iter_qr_input"
        )
        st.session_state['max_iter_qr'] = int(max_iter_input)


    return st.session_state['A_qr_input'], st.session_state['epsilon_qr'], st.session_state['max_iter_qr'], st.session_state['tiny']

# --- Ввод данных ---
A_input, epsilon, max_iterations, tiny_val = input_matrix_3x3_qr()

st.divider()

# --- Кнопка Решить ---
if st.button("Найти собственные значения (QR-алгоритм)", type="primary"):

    st.subheader("Исходная матрица и параметры:")
    col_mat, col_param = st.columns([2,1])
    with col_mat:
        st.markdown("**Матрица A:**")
        st.dataframe(pd.DataFrame(A_input))
    with col_param:
         st.metric("Точность ε", f"{epsilon:.1e}")
         st.metric("Макс. итераций", max_iterations)

    st.divider()
    st.subheader("Ход решения:")

    # Запуск QR-алгоритма
    try:
        # Используем обновленную функцию
        eigenvalues, iterations, converged, history, elapsed_time = qr_eigenvalue_algorithm(
            A_input, epsilon, max_iterations
        )

        st.markdown(f"**Результаты после {iterations} итераций (время: {elapsed_time:.4f} сек):**")

        if converged:
            st.success(f"Алгоритм сошелся (поддиагональные элементы достаточно малы для разделения блоков) за {iterations} итераций.")
        else:
            last_norm_str = f"{history['sub_diag_norm'][-1]:.2e}" if history['sub_diag_norm'] else "N/A"
            st.warning(f"Алгоритм НЕ сошелся за {max_iterations} итераций до требуемой точности ε={epsilon:.1e}. "
                       f"Последняя сумма модулей поддиаг. элементов: {last_norm_str}")

        # Отображение результатов
        st.markdown("**Найденные собственные значения (λ):**")
        formatted_eigenvalues = []
        # Используем разумный порог для отображения мнимой части
        display_threshold = epsilon # Можно использовать epsilon или tiny*100
        for val in eigenvalues: # eigenvalues теперь содержит правильные комплексные значения
            if abs(val.imag) < display_threshold:
                formatted_eigenvalues.append(f"{val.real:.6f}")
            else:
                formatted_eigenvalues.append(
                    f"{val.real:.6f} {'+' if val.imag >= 0 else '-'} {abs(val.imag):.6f}j"
                )
        st.dataframe(pd.DataFrame(formatted_eigenvalues, columns=['λ (найденные)']))


        # Сравнение с numpy.linalg.eigvals
        try:
            np_eigenvals = np.linalg.eigvals(A_input.astype(np.complex128))
            np_sort_indices = np.lexsort((np_eigenvals.imag, np_eigenvals.real))
            np_eigenvals_sorted = np_eigenvals[np_sort_indices]

            formatted_np_eigenvals = []
            for val in np_eigenvals_sorted:
                    # Используем tiny_val для сравнения с нулем мнимой части у NumPy
                    if abs(val.imag) < tiny_val:
                        formatted_np_eigenvals.append(f"{val.real:.6f}")
                    else:
                        formatted_np_eigenvals.append(
                            f"{val.real:.6f} {'+' if val.imag >= 0 else '-'} {abs(val.imag):.6f}j"
                        )

            with st.expander("Сравнение с NumPy (np.linalg.eigvals)"):
                    len_our = len(eigenvalues) # Сравниваем длину массива до форматирования
                    len_np = len(np_eigenvals_sorted)
                    if len_our != len_np:
                        st.warning(f"Количество найденных СЗ не совпадает: Наш={len_our}, NumPy={len_np}. Проверьте логику извлечения СЗ.")
                        # Вывод для отладки
                        max_len = max(len_our, len_np)
                        formatted_eigenvalues_padded = formatted_eigenvalues + ['-'] * (max_len - len_our)
                        formatted_np_eigenvals_padded = formatted_np_eigenvals + ['-'] * (max_len - len_np)
                        comparison_df = pd.DataFrame({
                            "QR-алгоритм (наш)": formatted_eigenvalues_padded,
                            "NumPy": formatted_np_eigenvals_padded
                         })
                        st.dataframe(comparison_df)
                    else:
                        comparison_df = pd.DataFrame({
                            "QR-алгоритм (наш)": formatted_eigenvalues,
                            "NumPy": formatted_np_eigenvals
                        })
                        # eigenvalues уже отсортированы нашим алгоритмом
                        diff = np.abs(eigenvalues - np_eigenvals_sorted) # Считаем разницу между комплексными массивами
                        comparison_df['|Разница|'] = [f"{d:.2e}" for d in diff]
                        st.dataframe(comparison_df)


        except np.linalg.LinAlgError:
                st.warning("NumPy не смог вычислить собственные значения для сравнения.")


        # Анализ сходимости
        st.divider()
        with st.expander("Анализ сходимости (Сумма модулей поддиагональных элементов)"):
            if history['iter']:
                    hist_df = pd.DataFrame({
                    'Итерация': history['iter'],
                    'Σ |Aₖ[i,j]| (i>j)': history['sub_diag_norm']
                })
                    st.dataframe(hist_df.style.format({'Σ |Aₖ[i,j]| (i>j)': '{:.4e}'}))

                    chart_df = hist_df.rename(columns={'Σ |Aₖ[i,j]| (i>j)':'Sub-diagonal Norm Sum'}).set_index('Итерация')
                    chart_df['Sub-diagonal Norm Sum'] = chart_df['Sub-diagonal Norm Sum'].replace(0, np.nan)
                    chart_df.dropna(subset=['Sub-diagonal Norm Sum'], inplace=True)
                    if not chart_df.empty:
                        st.line_chart(chart_df)
                        st.caption("График показывает уменьшение суммы модулей всех поддиагональных элементов (линейная шкала Y).")
                    else:
                         st.info("Нет данных для построения графика сходимости.")
            else:
                    st.info("История итераций пуста.")

    except ValueError as e:
        st.error(f"Ошибка входных данных или в алгоритме: {e}")
    except np.linalg.LinAlgError as e:
            st.error(f"Ошибка линейной алгебры (например, в QR разложении на одной из итераций): {e}")
    except Exception as e:
            st.error(f"Непредвиденная ошибка: {e}")
            # import traceback
            # st.error(traceback.format_exc())
