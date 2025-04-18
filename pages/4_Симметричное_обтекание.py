import streamlit as st

menu = st.sidebar.radio('***',
    ("Комплексный потенциал течения", 
    "Критические точки",       
    "Аналитическое решение",
    "Решение методом конечных элементов (FEniCS)",
    "Циркулляция по контуру цилиндра",
    "Программная реализация",
    )
)

if menu == "Комплексный потенциал течения":
    r"""
    ##### Комплексный потенциал течения
    
    $\begin{aligned}
    w = u_{\infty} \left(z + \frac{r^2}{z}\right) + \frac{\Gamma}{2 \pi i} \ln{z}, \quad z \in  \mathbb{C}
    \end{aligned}$
     * $r$ -- ридиус круглого цилиндра

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
    \frac{d w}{d z} = u_{x_1} - i u_{x_2} = u_{\infty} \left(1 - \frac{r^2}{z^2}\right) +  \frac{\Gamma}{2 \pi i} \frac{1}{z}
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
    w = u_{\infty} \left(z + \frac{r^2}{z}\right), \quad |z| > r
    \end{aligned}$

    $\begin{aligned}
    \frac{d w}{d z} = u_{\infty} \left(1 - \frac{r^2}{z^2}\right)
    \end{aligned}$
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

    with st.expander("Вычисление поля скоростей"):
        code = """
        
        U, V = np.vectorize(velocity_field)(X, Y)

        speed = np.sqrt(np.abs(U)**2 + np.abs(V)**2)
        """

        st.code(code, language="python")


elif menu == "Критические точки":
    r"""
    ##### Критические точки

    Определим положение критических точек, решив уравнение
    
    $\begin{aligned}
    u_{\infty} \left(1 - \frac{r^2}{z^2}\right) + \frac{\Gamma}{2 \pi i} \frac{1}{z} = 0
    \end{aligned}$

    Оно сводится к квадратному

    $\begin{aligned}
    z^2 - \frac{\Gamma i}{2 \pi u_{\infty}} z - r^2 = 0
    \end{aligned}$

    $\begin{aligned}
    z_{1,2} = \frac{\Gamma i}{4 \pi u_{\infty}} \pm \sqrt{r^2 - \frac{\Gamma^2}{16 \pi^2 u^2_{\infty}}}
    \end{aligned}$

    """

    with st.expander("Критические точки"):
        code = """
        # Поиск критических точек
        def stagnation_eq(x):
            #Уравнение для нахождения критической точки (по горизонтальной оси)
            x = x[0]
            z = complex(x - cx, cy - cy)  # Только x-координата меняется, y=cy
            w_prime = vinf * (np.exp(-1j * alpha_rad) - c**2 / z**2 * np.exp(1j * alpha_rad))
            return np.real(w_prime)  # Должно быть равно 0

        # Ищем два корня (левая и правая критические точки)
        stagnation_x1 = fsolve(stagnation_eq, cx + c)[0]
        stagnation_x2 = fsolve(stagnation_eq, cx - c)[0]
        stagnation_points = np.array([[stagnation_x1, cy], [stagnation_x2, cy]])
        """

        st.code(code, language="python")

    subtopic = st.selectbox(
        "Выберите случай",  
        ["1. Оба корня чисто мнимые и лежат на мнимой оси", "2. Кратные корни", "3. Комплексные корни", "4. Бесциркулляционное течение"]  
    )

    if subtopic == "1. Оба корня чисто мнимые и лежат на мнимой оси":
        r"""
        $\begin{aligned}
        \Gamma > 4 \pi u_{\infty} r
        \end{aligned}$

        $\begin{aligned}
        z_1 = i \left[\frac{\Gamma i}{4 \pi u_{\infty}} + \sqrt{\frac{\Gamma^2}{16 \pi^2 u^2_{\infty}} - r^2}\right]
        \end{aligned}$

        $\begin{aligned}
        \text{Im } z_1 > \frac{\Gamma}{4 \pi u_{\infty}} > r
        \end{aligned}$

        $\begin{aligned}
        z_2 = i \left[\frac{\Gamma i}{4 \pi u_{\infty}} - \sqrt{\frac{\Gamma^2}{16 \pi^2 u^2_{\infty}} - r^2}\right]
        \end{aligned}$

        $\begin{aligned}
        \text{Im } z_2 = \frac{a^2}{\frac{\Gamma}{4 \pi u_{\infty}} + \sqrt{\frac{\Gamma^2}{16 \pi^2 u_{\infty}^2} - r^2}} < \frac{r^2}{\Gamma^2 / (4 \pi u_{\infty})} < \frac{r^2}{r} < r
        \end{aligned}$ 

        Второй корень находится внутри круга и не представляет для нас никакого смысла.
        """
        st.image("teor_4.png", caption="",use_container_width=True)

    elif subtopic == "2. Кратные корни":
        r"""
        $\begin{aligned}
        \Gamma = 4 \pi u_{\infty} r
        \end{aligned}$

        $\begin{aligned}
        z_1 = z_2 = i r
        \end{aligned}$

        Корни сливаются между собой и расположены в наивысшей точке цилиндра.
        """
        st.image("teor_3.png", caption="",use_container_width=True)

    elif subtopic == "3. Комплексные корни":
        r"""
        $\begin{aligned}
        \Gamma < 4 \pi u_{\infty} r
        \end{aligned}$

        $\begin{aligned}
        z_1 = \frac{\Gamma i}{4 \pi u_{\infty}} + \sqrt{r^2 - \frac{\Gamma^2}{16 \pi^2 u^2_{\infty}}}
        \end{aligned}$

        $\begin{aligned}
        z_2 = \frac{\Gamma i}{4 \pi u_{\infty}} - \sqrt{r^2 - \frac{\Gamma^2}{16 \pi^2 u^2_{\infty}}}
        \end{aligned}$

        Корни расположены симметрично относительно оси $x_2$. Оба корня лежат на окружности, так как

        $\begin{aligned}
        |z_1| = |z_2| = r
        \end{aligned}$
        """
        st.image("teor_5.png", caption="",use_container_width=True)

    elif subtopic == "4. Бесциркулляционное течение":
        r"""
        $\begin{aligned}
        u_{\infty} \left(1 - \frac{r^2}{z^2}\right) = 0
        \end{aligned}$

        $\begin{aligned}
        z_1 = r
        \end{aligned}$

        $\begin{aligned}
        z_2 = - r
        \end{aligned}$

        """
        st.image("teor_2.png", caption="",use_container_width=True)
    

elif menu == "Аналитическое решение":
    r"""
    ##### Аналитическое решение
    """

    st.image("circulation_cylinder.png", caption="",use_container_width=True)

    r"""
     -----------------------------------------------

     * Радиус цилиндра: $r = 0.125$
     * Скорость на входе: $u_{\infty} = 1$

     ------------------------------------------------
    ##### Критические точки
     * $x_1 = 0.375, \quad x_2 = 0.5$
    
     * $x_1 = 0.625, \quad x_2 = 0.5$
    -------------------------------------------------
    ##### Циркулляция по контуру цилиндра
     * $\Gamma = 0$
    """


elif menu == "Решение методом конечных элементов (FEniCS)":

    r"""

    ##### Решение методом конечных элементов (FEniCS)

    """

    subtopic = st.selectbox(
        "Размер сетки",  
        ["712", "2958", "13746"]  
    )

    if subtopic == "712":

        st.image("ellips_1.png", caption="",use_container_width=True)

        r"""
        -----------------------------------------------

        * Число ячеек сетки: 712

        * Число узлов сетки: 396

        ------------------------------------------------
        """
        st.image("ellips_solve_1.png", caption="",use_container_width=True)

    elif subtopic == "2958":

        st.image("ellips_2.png", caption="",use_container_width=True)

        r"""
        -----------------------------------------------

        * Число ячеек сетки: 2968

        * Число узлов сетки: 1636

        ------------------------------------------------
        """
        st.image("ellips_solve_2.png", caption="",use_container_width=True)

    elif subtopic == "13746":

        st.image("ellips_3.png", caption="",use_container_width=True)

        r"""
        -----------------------------------------------

        * Число ячеек сетки: 13746

        * Число узлов сетки: 7307

        ------------------------------------------------
        """
        st.image("ellips_solve_3.png", caption="",use_container_width=True)

elif menu == "Циркулляция по контуру цилиндра":

    r"""
    ##### Циркулляция по контуру цилиндра

    """
    r"""

        | Размер сетки                       |      $p=1$         |        $p=2$       |        $p=3$       |
        |------------------------------------|--------------------|--------------------|--------------------|
        | $396$                              | $-1.29200*10^{-4}$ | $-7.53011*10^{-5}$ | $-3.96871*10^{-5}$ |
        | $1636$                             |  $8.61544*10^{-7}$ |  $1.19674*10^{-5}$ |  $7.68601*10^{-6}$ |
        | $7307$                             |  $2.88749*10^{-6}$ |  $1.19933*10^{-6}$ |  $8.54737*10^{-7}$ |

    """
    r"""
    ##### Число искомых дискретных значений
    """
    """


        | Размер сетки                       |        $p=1$       |        $p=2$       |    $p=3$    |
        |------------------------------------|--------------------|--------------------|-------------|
        | $396$                              |        $396$       |      $1504$        |    $3324$   |
        | $1636$                             |        $1636$      |      $6240$        |   $13812$   |
        | $7307$                             |        $7307$      |      $28360$       |   $63159$   |

    """



elif menu == "Программная реализация":

    r"""
    ##### Программная реализация
    """

    with st.expander("Определение функционального пространства"):
        code = """
            V = FunctionSpace(mesh, "CG", 3)
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
        bcs = [DirichletBC(V, Constant(0.0), boundaries, 1), #bottom
                DirichletBC(V, u_infinity * H, boundaries, 2), #top
                DirichletBC(V, Constant(0.5), boundaries, 5), #cylinder
                DirichletBC(V, u_infinity * H, boundaries, 3), #inlet
                DirichletBC(V, u_infinity * H, boundaries, 4)] #outlet
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


    

    





    
