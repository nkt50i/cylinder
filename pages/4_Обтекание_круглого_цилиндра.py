import streamlit as st

menu = st.sidebar.radio('***',
    ("Комплексный потенциал течения", 
    "Критические точки",       
    "Аналитическое решение",
    "Решение методом конечных элементов (FEM)",
    "Циркулляция",
    "Программная реализация",
    )
)

if menu == "Комплексный потенциал течения":
    r"""
    ##### Комплексный потенциал течения
    
    $\begin{aligned}
    w = u_{\infty} \left(z + \frac{a^2}{z}\right) + \frac{\Gamma}{2 \pi i} \ln{z}, \quad z \in  \mathbb{C}
    \end{aligned}$
     * $a$ - ридиус круглого цилиндра

     * $\begin{aligned}
       u_{\infty} = u_{x_1} + i u_{x_2}
       \end{aligned}$

     * $\begin{aligned}
       z = x_1 + i x_2
       \end{aligned}$

    Второе слагаемое представляет собой комплексный потенциал вихря с циркуляцией $\Gamma$
    
    $\begin{aligned}
    w_{vortex} =  \frac{\Gamma}{2 \pi i} \ln{z}
    \end{aligned}$

    ##### Комплексно-сопряженная скорость

    $\begin{aligned}
    \frac{d w}{d z} = u_{x_1} - i u_{x_2} = u_{\infty} \left(1 - \frac{a^2}{z^2}\right) +  \frac{\Gamma}{2 \pi i} \frac{1}{z}
    \end{aligned}$

    $\begin{aligned}    
    \text{Re } \frac{d w}{ d z} = u_{x_1}
    \end{aligned}$

    $\begin{aligned}    
    \text{Im } \frac{d w}{ d z} = - u_{x_2}
    \end{aligned}$


    Так как все линии тока от вихря – это окружности с центром вначале координат, то ясно, что добавление этого потенциала не нарушит условий непротекания цилиндра.

    ##### Бесциркулляционное течение
    
    $\Gamma = 0$

    $\begin{aligned}
    w_{vortex} = 0
    \end{aligned}$

    $\begin{aligned}
    w = u_{\infty} \left(z + \frac{a^2}{z}\right), \quad |z| > a
    \end{aligned}$

    $\begin{aligned}
    \frac{d w}{d z} = u_{\infty} \left(1 - \frac{a^2}{z^2}\right)
    \end{aligned}$
    """

elif menu == "Критические точки":
    r"""
    ##### Критические точки

    Определим положение критических точек, решив уравнение
    
    $\begin{aligned}
    u_{\infty} \left(1 - \frac{a^2}{z^2}\right) + \frac{\Gamma}{2 \pi i} \frac{1}{z} = 0
    \end{aligned}$

    Оно сводится к квадратному

    $\begin{aligned}
    z^2 - \frac{\Gamma i}{2 \pi u_{\infty}} z - a^2 = 0
    \end{aligned}$

    $\begin{aligned}
    z_{1,2} = \frac{\Gamma i}{4 \pi u_{\infty}} \pm \sqrt{a^2 - \frac{\Gamma^2}{16 \pi^2 u^2_{\infty}}}
    \end{aligned}$

    """

    subtopic = st.selectbox(
        "Выберите случай",  
        ["1. Оба корня чисто мнимые и лежат на мнимой оси", "2. Кратные корни", "3. Комплексные корни", "4. Бесциркулляционное течение"]  
    )

    if subtopic == "1. Оба корня чисто мнимые и лежат на мнимой оси":
        r"""
        $\begin{aligned}
        \Gamma > 4 \pi u_{\infty} a
        \end{aligned}$

        $\begin{aligned}
        z_1 = i \left[\frac{\Gamma i}{4 \pi u_{\infty}} + \sqrt{\frac{\Gamma^2}{16 \pi^2 vu^2_{\infty}} - a^2}\right]
        \end{aligned}$

        $\begin{aligned}
        \text{Im } z_1 > \frac{\Gamma}{4 \pi u_{\infty}} > a
        \end{aligned}$

        $\begin{aligned}
        z_2 = i \left[\frac{\Gamma i}{4 \pi u_{\infty}} - \sqrt{\frac{\Gamma^2}{16 \pi^2 u^2_{\infty}} - a^2}\right]
        \end{aligned}$

        $\begin{aligned}
        \text{Im } z_2 = \frac{a^2}{\frac{\Gamma}{4 \pi u_{\infty}} + \sqrt{\frac{\Gamma^2}{16 \pi^2 u_{\infty}^2} - a^2}} < \frac{a^2}{\Gamma^2 / (4 \pi u_{\infty})} < \frac{a^2}{a} < a
        \end{aligned}$ 

        Второй корень находится внутри круга и не представляет для нас никакого смысла.
        """

    elif subtopic == "2. Кратные корни":
        r"""
        $\begin{aligned}
        \Gamma = 4 \pi u_{\infty} a
        \end{aligned}$

        $\begin{aligned}
        z_1 = z_2 = i a
        \end{aligned}$

        Корни сливаются между собой и расположены в наивысшей точке цилиндра.
        """

    elif subtopic == "3. Комплексные корни":
        r"""
        $\begin{aligned}
        \Gamma < 4 \pi u_{\infty} a
        \end{aligned}$

        $\begin{aligned}
        z_1 = \frac{\Gamma i}{4 \pi u_{\infty}} + \sqrt{a^2 - \frac{\Gamma^2}{16 \pi^2 u^2_{\infty}}}
        \end{aligned}$

        $\begin{aligned}
        z_2 = \frac{\Gamma i}{4 \pi u_{\infty}} - \sqrt{a^2 - \frac{\Gamma^2}{16 \pi^2 u^2_{\infty}}}
        \end{aligned}$

        Корни расположены симметрично относительно оси $x_2$. Оба корня лежат на окружности, так как

        $\begin{aligned}
        |z_1| = |z_2| = a
        \end{aligned}$
        """

    elif subtopic == "4. Бесциркулляционное течение":
        r"""
        $\begin{aligned}
        u_{\infty} \left(1 - \frac{a^2}{z^2}\right) = 0
        \end{aligned}$

        $\begin{aligned}
        z_1 = a
        \end{aligned}$

        $\begin{aligned}
        z_2 = - a
        \end{aligned}$

        """

elif menu == "Аналитическое решение":
    r"""
    ##### Аналитическое решение
    """

    st.image("circulation_cylinder.png", caption="",use_container_width=True)

    r"""
     * Радиус цилиндра: $a = 0.125$
     * Скорость на входе: $u_{\infty} = 1$

     ------------------------------------------------
    ##### Критические точки
    $x_1 = 0.375, \quad x_2 = 0.5$
    
    $x_1 = 0.625, \quad x_2 = 0.5$
    ##### Циркулляция
    $\Gamma = 0$
    """

    with st.expander("Функция комплексного потенциала и комплексно-сопряженной скорости"):
        code = """
        def velocity_field(x, y):
            z = (x - cx) + 1j * (y - cy)
            if np.abs(z) < c:  # Если точка внутри цилиндра, возвращаем NaN
                return np.nan, np.nan
            w_prime = vinf * (np.exp(-1j * alpha_rad) - c**2 / z**2 * np.exp(1j * alpha_rad))
            return np.real(w_prime), -np.imag(w_prime)
        """

        st.code(code, language="python")

    with st.expander("Поиск критических точек"):
        code = """
        def stagnation_eq(x):
            x = x[0]  # Берём первый элемент массива
            z = complex(x - cx, cy - cy)  # Только x-координата меняется, y=cy
            w_prime = vinf * (np.exp(-1j * alpha_rad) - c**2 / z**2 * np.exp(1j * alpha_rad))
            return np.real(w_prime)  # Должно быть равно 0

            # Ищем два корня (левый и правый стагнационные точки)
            stagnation_x1 = fsolve(stagnation_eq, cx + c)[0]
            stagnation_x2 = fsolve(stagnation_eq, cx - c)[0]
            stagnation_points = np.array([[stagnation_x1, cy], [stagnation_x2, cy]])
            """

        st.code(code, language="python")

    with st.expander("Вычисление поля скоростей"):
        code = """
        
        U, V = np.vectorize(velocity_field)(X, Y)

        speed = np.sqrt(np.abs(U)**2 + np.abs(V)**2)
        """

        st.code(code, language="python")

elif menu == "Решение методом конечных элементов (FEM)":

    r"""

    ##### Решение методом конечных элементов (FEM)

    """

    subtopic = st.selectbox(
        "Размер сетки",  
        ["712", "2958", "13746"]  
    )

    if subtopic == "712":

        st.image("ellips_1.png", caption="",use_container_width=True)

        r"""
        * Число ячеек сетки: 712

        * Число узлов сетки: 396

        ------------------------------------------------
        """
        st.image("ellips_solve_1.png", caption="",use_container_width=True)

    elif subtopic == "2958":

        st.image("ellips_2.png", caption="",use_container_width=True)

        r"""
        * Число ячеек сетки: 2968

        * Число узлов сетки: 1636

        ------------------------------------------------
        """
        st.image("ellips_solve_2.png", caption="",use_container_width=True)

    elif subtopic == "13746":

        st.image("ellips_3.png", caption="",use_container_width=True)

        r"""
        * Число ячеек сетки: 13746

        * Число узлов сетки: 7307

        ------------------------------------------------
        """
        st.image("ellips_solve_3.png", caption="",use_container_width=True)

elif menu == "Циркулляция":

    r"""
    ##### Циркулляция

    """
    r"""

        | Размер сетки                       |      $p=1$     |      $p=2$     |      $p=3$     |
        |------------------------------------|----------------|----------------|----------------|
        | $712$                              | $-1.29200e-04$ | $-7.53011e-05$ | $-3.96871e-05$ |
        | $2968$                             |  $8.61544e-07$ |  $1.19674e-05$ |  $7.68601e-06$ |
        | $13746$                            |  $2.88749e-06$ |  $1.19933e-06$ |  $8.54737e-07$ |

    """

elif menu == "Программная реализация":

    r"""
    ##### Программная реализация
    """

    with st.expander("Определение функционального пространства"):
        code = """
            V = FunctionSpace(mesh, "CG", 1)
        """

        st.code(code, language="python")

    with st.expander("Условие на входе в канал"):
        code = """
            u_infinity = Expression("x[1]", degree=2)
            H = 1
            psi_0 = u_infinity * H
            """

        st.code(code, language="python")

    with st.expander("Граничные условия"):
        code = """
        bcs = [DirichletBC(V, Constant(0.0), boundaries, 1), 
                DirichletBC(V, u_infinity * H, boundaries, 2),
                DirichletBC(V, Constant(0.5), boundaries, 5),
                DirichletBC(V, u_infinity * H, boundaries, 3),
                DirichletBC(V, u_infinity * H, boundaries, 4)]
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
            V_vector = VectorFunctionSpace(mesh, "CG", 3)
            velocity = project(grad(u), V_vector)
            velocity_magnitude = project(sqrt(dot(velocity, velocity)), V)
        """

        st.code(code, language="python")

    with st.expander("Вычисление циркуляции"):
        code = """
            n = FacetNormal(mesh)
            u_n = dot(grad(u), n)
            Gamma = assemble(u_n * ds(subdomain_data=boundaries, subdomain_id=5))
        """

        st.code(code, language="python")

    with st.expander("Поиск критических точек"):
        code = """
            threshold = 1e-15
            critical_points = (velocity_values < threshold)
            critical_x = x[critical_points]
            critical_y = y[critical_points]

        """

        st.code(code, language="python")

    with st.expander("Построение эллипса"):
        code = """
            # Параметры эллипса
            cx, cy = 0.5, 0.5  # Центр эллипса
            a, b = 0.125, 0.125   # Полуоси эллипса
            alpha = np.radians(45)  # Угол поворота

            # Генерация эллипса
            theta = np.linspace(0, 2 * np.pi, 300)
            x_ellipse = a * np.cos(theta)
            y_ellipse = b * np.sin(theta)

            # Поворот эллипса
            ellipse_x = cx + x_ellipse * np.cos(alpha) - y_ellipse * np.sin(alpha)
            ellipse_y = cy + x_ellipse * np.sin(alpha) + y_ellipse * np.cos(alpha)

        """

        st.code(code, language="python")


    

    





    
