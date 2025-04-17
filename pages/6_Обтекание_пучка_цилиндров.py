import streamlit as st
from PIL import Image
import base64
import subprocess
import os
import math
import gmsh
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

def run_gmsh(file_path):
    try:
        env = os.environ.copy()
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"  # Используем программный рендеринг
        subprocess.run(["gmsh", file_path], check=True, env=env)
        st.success("Gmsh успешно запущен в программном режиме!")
    except FileNotFoundError:
        st.error("Gmsh не найден. Убедитесь, что он установлен и доступен в PATH.")
    except subprocess.CalledProcessError:
        st.error("Ошибка при запуске Gmsh.")

def show_code(code, language="python"):
    st.code(code, language)


menu = st.sidebar.radio('***',
    ("Схема течения", 
    ""
    "Симметричное обтекание",       
    "Несимметричное обтекание",
    "Программная реализация",
    )
)

if menu == "Схема течения":
    st.markdown(r"""
    **Схема течения**
    """)

    # Загрузка изображения
    st.image("6.png", use_container_width=True)
    r"""
    -----------------------------------
    ###### Параметры

     - $N = 9$ -- количество ячеек периодичности
     - $H = 3$ -- высота канала
     - $L = 3$ -- ширина канала
    """
    with st.expander("Параметры ячейки периодичности"):
        code_1 = """
        // Размеры области
        L = 3.0; // Ширина области
        H = 3.0; // Высота области

        // Размеры эллипсов
        r = 0.125;
        R = 0.175;

        // Сетка и шаг
        N = 30;
        d = H / N;
        dd = 0.05 * d;

        // Количество эллипсов
        nx = 3;
        ny = 3;

        // Размер ячейки
        dx = L / nx;
        dy = H / ny;
        """
        show_code(code_1, "python")
    
    with st.expander("Построение геометрических примитивов"):
        code_2 = """
        // Углы прямоугольника
        Point(1) = {0, 0, 0, d};
        Point(2) = {L, 0, 0, d};
        Point(3) = {L, H, 0, d};
        Point(4) = {0, H, 0, d};

        // Границы
        Line(1) = {1, 2}; // нижняя
        Line(2) = {2, 3}; // правая
        Line(3) = {3, 4}; // верхняя
        Line(4) = {4, 1}; // левая

        Line Loop(1) = {1, 2, 3, 4};

        // Общие списки
        ellipses_loops[] = {};
        ellipses_all_curves[] = {};

        line_group_id = 10;

        For j In {1:ny}
            ellipse_lines_row[] = {};
            For i In {1:nx}
                n = (j - 1) * nx + i;

                xc = (i - 0.5) * dx;
                yc = (j - 0.5) * dy;

                p = 10 * n;
                c = 1000 + 4 * n;

                Point(p + 0) = {xc, yc, 0, dd};
                Point(p + 1) = {xc + R, yc, 0, dd};
                Point(p + 2) = {xc, yc + r, 0, dd};
                Point(p + 3) = {xc - R, yc, 0, dd};
                Point(p + 4) = {xc, yc - r, 0, dd};

                Ellipse(c + 0) = {p + 1, p + 0, p + 2};
                Ellipse(c + 1) = {p + 2, p + 0, p + 3};
                Ellipse(c + 2) = {p + 3, p + 0, p + 4};
                Ellipse(c + 3) = {p + 4, p + 0, p + 1};

                Rotate {{0, 0, 1}, {xc, yc, 0}, Pi/4} {
                    Curve{c + 0 : c + 3};
                }

                Line Loop(n + 1) = {c + 0, c + 1, c + 2, c + 3};
                ellipses_loops[] += {n + 1};
                ellipses_all_curves[] += {c + 0, c + 1, c + 2, c + 3};
                ellipse_lines_row[] += {c + 0, c + 1, c + 2, c + 3};
            EndFor

            // Физическая группа строк эллипсов с уникальным числовым идентификатором
            Physical Line(line_group_id) = {ellipse_lines_row[]};
            line_group_id += 1;
        EndFor
        """
        show_code(code_2, "python")

    with st.expander("Построение плоскости"):
        code_3 = """
        // Поверхность с отверстиями
        Plane Surface(1) = {1, ellipses_loops[]};
        """
        show_code(code_3, "python")

    with st.expander("Определение физических ГУ"):
        code_4 = """
        // Границы
        Physical Line(101) = {1}; // bottom
        Physical Line(102) = {2}; // right
        Physical Line(103) = {3}; // top
        Physical Line(104) = {4}; // left

        // Основная поверхность
        Physical Surface(201) = {1};
        """
        show_code(code_4, "python")

    with st.expander("Построение сетки"):
        code_5 = """
        // Построение сетки
        Mesh 2;
        """
        show_code(code_5, "python")

    result = ''
    for i in range(1, 6):
        result += globals()[f'code_{i}']


    def save_example_file():
        example_file_path = './group_ellips.geo'
        with open(example_file_path, 'w') as f:
                f.write(result)
        return example_file_path

    # Кнопка для загрузки и запуска примера
    if st.button("Построение расчетной области 🔧"):
            example_file_path = save_example_file()
            run_gmsh(example_file_path)

elif menu == "Симметричное обтекание":
    r"""
    ##### Сетка
    """
    st.image("group_ellips_с.png", caption="",use_container_width=True)
    
    r"""
        ------------------------------------------------
        * Число ячеек сетки: 335690

        * Число узлов сетки: 168677

        ------------------------------------------------
        ##### Решение методом конечных элементов (FEniCS)
    """
    st.image("ellips_array_result_с.png", caption="",use_container_width=True)

    r"""
        ------------------------------------------------
        * Суммарная циркулляция: $5.146854*10^{-8}$
        """

elif menu == "Несимметричное обтекание":

    subtopic = st.selectbox(
        "Большая полуось эллипса",  
        ["R = 0.175", "R = 0.225", "R = 0.275"]  
    )

    if subtopic == "R = 0.175":

        
        r"""
            ------------------------------------------------
            * Число ячеек сетки: 337634

            * Число узлов сетки: 169793

            ------------------------------------------------
            ##### Решение методом конечных элементов (FEniCS)
        """
        st.image("ellips_array_result_1.png", caption="",use_container_width=True)

        r"""
            ------------------------------------------------
            * Суммарная циркулляция: $2.536574*10^{-7}$
            """


    elif subtopic == "R = 0.225":

        
        r"""
            ------------------------------------------------

            * Число ячеек сетки: 355588

            * Число узлов сетки: 179094

            ------------------------------------------------
            ##### Решение методом конечных элементов (FEniCS)
        """
        st.image("ellips_array_result_2.png", caption="",use_container_width=True)

        r"""
            ------------------------------------------------
            * Суммарная циркулляция: $2.536574*10^{-7}$
            """

    elif subtopic == "R = 0.275":
        
        r"""
            ------------------------------------------------

            * Число ячеек сетки: 356954

            * Число узлов сетки: 179615

            ------------------------------------------------
            ##### Решение методом конечных элементов (FEniCS)
        """
        st.image("ellips_array_result_3.png", caption="",use_container_width=True)

        r"""
            ------------------------------------------------
            * Суммарная циркулляция: $7.114579*10^{-6}$
            """

elif menu == "Программная реализация":

    r"""
    ##### Программная реализация
    """

    with st.expander("Определение функционального пространства"):
        code = """
            # Функциональное пространство
            V = FunctionSpace(mesh, "CG", 1)
        """

        st.code(code, language="python")

    with st.expander("Условие на входе в канал"):
        code = """
            # Граничные условия
            u_infinity = Expression("x[1]", degree=2)
            H = 3
            psi_top = u_infinity * H
            """

        st.code(code, language="python")

    with st.expander("Граничные условия"):
        code = """
        #Граничные условия
        bcs = [
            DirichletBC(V, Constant(0.0), boundaries, 101),     # top
            DirichletBC(V, psi_top, boundaries, 102),           # bottom
            DirichletBC(V, psi_top, boundaries, 103),           # inlet
            DirichletBC(V, psi_top, boundaries, 104),           # outlet
            DirichletBC(V, Constant(1.5), boundaries, 10),      # cylinder row 1
            DirichletBC(V, Constant(4.5), boundaries, 11),      # cylinder row 2
            DirichletBC(V, Constant(7.5), boundaries, 12)       # cylinder row 3
        ]
        """

        st.code(code, language="python")

    with st.expander("Вариационная задача"):
        code = """
            u = TrialFunction(V)
            v = TestFunction(V)
            f = Constant(0.0)
            a = dot(grad(u), grad(v)) * dx
            L = f * v * dx
        """

        st.code(code, language="python")

    with st.expander("Решение задачи"):
        code = """
            u = Function(V)
            solve(a == L, u, bcs)
        """

        st.code(code, language="python")

    with st.expander("Вычисление скорости"):
        code = """
            V_vector = VectorFunctionSpace(mesh, "CG", 1)
            velocity = project(grad(u), V_vector)
            velocity_magnitude = project(sqrt(dot(velocity, velocity)), V)
        """

        st.code(code, language="python")

    with st.expander("Поиск критических точек"):
        code = """
            # Поиск критических точек (где скорость почти нулевая)
            threshold = 1e-1
            critical_points = (velocity_values < threshold)
            critical_x = x[critical_points]
            critical_y = y[critical_points]

            fig, ax = plt.subplots(figsize=(10, 10))
            cbar = plt.colorbar(
                plt.tricontourf(x, y, triangles, velocity_values, levels=100, cmap='viridis')
            )

        """

        st.code(code, language="python")

    with st.expander("Построение эллипса"):
        code = """
            # Автоматическая отрисовка эллипсов
            additional_centers = [
                (0.5, 0.5), (1.5, 0.5), (0.5, 2.5),
                (0.5, 1.5), (1.5, 1.5), (1.5, 2.5),
                (2.5, 0.5), (2.5, 1.5), (2.5, 2.5)
            ]
            a, b = 0.275, 0.125
            alpha = np.pi / 4
            theta = np.linspace(0, 2 * np.pi, 200)
            # Построение дополнительных эллипсов
            for cx, cy in additional_centers:
                x_ellipse = a * np.cos(theta)
                y_ellipse = b * np.sin(theta)
                x_rot = cx + x_ellipse * np.cos(alpha) - y_ellipse * np.sin(alpha)
                y_rot = cy + x_ellipse * np.sin(alpha) + y_ellipse * np.cos(alpha)
                ax.fill(x_rot, y_rot, 'w', zorder=3)
                ax.plot(x_rot, y_rot, 'r', linewidth=1, zorder=4)
        """

        st.code(code, language="python")


        

