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
    
    Пусть эллипс обтекается идеальной жидкостью со скоростью $u_{\infty} e^{i \alpha}$

    Комплексный потенциал в параметрическом виде

    $\begin{aligned}
    w = \frac{u_{\infty}}{2} \left( e^{-i \alpha} t + \frac{e^{i \alpha} r^2}{t} \right) + \frac{\Gamma}{2 \pi i} \ln{t}, \quad r = a + b
    \end{aligned}$

    $\begin{aligned}
    z = \frac{1}{2} \left(t + \frac{c^2}{t}\right), \quad c = \sqrt{a^2 - b^2}, \quad a \ge b
    \end{aligned}$
     * $r$ - ридиус параметрического круга

     * $a, b$ - полуоси эллипса

     * $\alpha$ - угол атаки

     * $\begin{aligned}
       u_{\infty} = u_{x_1} + i u_{x_2}
       \end{aligned}$

     * $\begin{aligned}
       z = x_1 + i x_2
       \end{aligned}$

    Исключим параметр $t$

    $\begin{aligned}
    t^2 - 2zt + c^2 = 0, \quad t = z + \sqrt{z^2 - c^2}
    \end{aligned}$

    Если взять корень со знаком минус, то внешность эллипса перейдет во внутренность параметрического круга.

    $\begin{aligned}
    w = \frac{u_{\infty}}{2} \left( e^{-i \alpha} \left(z + \sqrt{z^2 - c^2}\right)  + \frac{e^{i \alpha} r^2}{\left(z + \sqrt{z^2 - c^2}\right)} \right) + \frac{\Gamma}{2 \pi i} \ln{\left(z + \sqrt{z^2 - c^2}\right)}
    \end{aligned}$


    ##### Комплексно-сопряженная скорость

    $\begin{aligned}
    \frac{d w}{d z} = \frac{u_{\infty}}{2} \left( e^{-i \alpha} \left(1 + \frac{z}{\sqrt{z^2 - c^2}} \right)  + \frac{e^{i \alpha} r^2 \left(1 + \frac{z}{\sqrt{z^2 - c^2}}\right)}{\left(z + \sqrt{z^2 - c^2}\right)^2} \right) + \frac{\Gamma}{2 \pi i} \frac{\left(1 + \frac{z}{\sqrt{z^2 - c^2}} \right)}{\left(z + \sqrt{z^2 - c^2}\right)}
    \end{aligned}$

    $\begin{aligned}
    \frac{d w}{d z} = u_{x_1} - i u_{x_2} 
    \end{aligned}$

    $\begin{aligned}    
    \text{Re } \frac{d w}{ d z} = u_{x_1}
    \end{aligned}$

    $\begin{aligned}    
    \text{Im } \frac{d w}{ d z} = - u_{x_2}
    \end{aligned}$


    ##### Бесциркулляционное течение
    
    $\Gamma = 0$

    $\begin{aligned}
    w = \frac{u_{\infty}}{2} \left( e^{-i \alpha} \left(z + \sqrt{z^2 - c^2}\right)  + \frac{e^{i \alpha} r^2}{\left(z + \sqrt{z^2 - c^2}\right)} \right)
    \end{aligned}$

    $\begin{aligned}
    \frac{d w}{d z} = \frac{u_{\infty}}{2} \left( e^{-i \alpha} \left(1 + \frac{z}{\sqrt{z^2 - c^2}} \right)  + \frac{e^{i \alpha} r^2 \left(1 + \frac{z}{\sqrt{z^2 - c^2}}\right)}{\left(z + \sqrt{z^2 - c^2}\right)^2} \right)
    \end{aligned}$
    
    """

elif menu == "Критические точки":
    r"""
    ##### Критические точки

    Определим положение критических точек, решив уравнение
    
    $\begin{aligned}
    \frac{d w}{d z} = \frac{u_{\infty}}{2} \left( e^{-i \alpha} \left(1 + \frac{z}{\sqrt{z^2 - c^2}} \right)  + \frac{e^{i \alpha} r^2 \left(1 + \frac{z}{\sqrt{z^2 - c^2}}\right)}{\left(z + \sqrt{z^2 - c^2}\right)^2} \right) + \frac{\Gamma}{2 \pi i} \frac{\left(1 + \frac{z}{\sqrt{z^2 - c^2}} \right)}{\left(z + \sqrt{z^2 - c^2}\right)} = 0
    \end{aligned}$

    При $\Gamma = 0$

    $\begin{aligned}
    \frac{d w}{d z} = \frac{u_{\infty}}{2} \left( e^{-i \alpha} \left(1 + \frac{z}{\sqrt{z^2 - c^2}} \right)  + \frac{e^{i \alpha} r^2 \left(1 + \frac{z}{\sqrt{z^2 - c^2}}\right)}{\left(z + \sqrt{z^2 - c^2}\right)^2} \right) = 0
    \end{aligned}$

    """
    
elif menu == "Аналитическое решение":
    r"""
    ##### Аналитическое решение
    """

    subtopic = st.selectbox(
        "Большая полуось эллипса",  
        ["a = 0.175", "a = 0.225", "a = 0.275"]  
    )

    if subtopic == "a = 0.175":

        st.image("Ellips_1.1.png", caption="",use_container_width=True)

        r"""
        * Полуоси эллипса: $a = 0.175, \quad b = 0.125$
        * Скорость на входе: $u_{\infty} = 1$
        * Угол поворота цилиндра: $\alpha = \frac{\pi}{4}$

        ------------------------------------------------
        ##### Критические точки
        $x_1 = 0.35, \quad x_2 = 0.475$
        
        $x_1 = 0.65, \quad x_2 = 0.52500$
        ##### Циркулляция
        $\Gamma = 0$
        """

    elif subtopic == "a = 0.225":

        st.image("Ellips_1.2.png", caption="",use_container_width=True)

        r"""
        * Полуоси эллипса: $a = 0.225, \quad b = 0.125$
        * Скорость на входе: $u_{\infty} = 1$

        ------------------------------------------------
        ##### Критические точки
        $x_1 = 0.325, \quad x_2 = 0.45$
        
        $x_1 = 0.675, \quad x_2 = 0.55$
        ##### Циркулляция
        $\Gamma = 0$
        """

    elif subtopic == "a = 0.275":

        st.image("Ellips_1.3.png", caption="",use_container_width=True)

        r"""
        * Полуоси эллипса: $a = 0.275, \quad b = 0.125$
        * Скорость на входе: $u_{\infty} = 1$

        ------------------------------------------------
        ##### Критические точки
        $x_1 = 0.3, \quad x_2 = 0.425$
        
        $x_1 = 0.7, \quad x_2 = 0.575$
        ##### Циркулляция
        $\Gamma = 0$
        """

    with st.expander("Функция комплексного потенциала и комплексно-сопряженной скорости"):
        code = """
            def t(z):
                return z + np.sqrt(z - c) * np.sqrt(z + c)
    
            def w(z, Gamma):
                return (vinff / 2 * (np.exp(-1j * alpha1) * t(z) + (np.exp(1j * alpha1) * r**2) / t(z)) +
                        (Gamma * np.log(t(z))) / (2 * np.pi * 1j))
    
            def w_prime(z, Gamma):
                h = 1e-8
                return (w(z + h, Gamma) - w(z - h, Gamma)) / (2 * h)
        """

        st.code(code, language="python")

    with st.expander("Поиск критических точек"):
        code = """
        def find_stagnation_point(initial_guess):
            def func(z_real_imag):
                z_complex = z_real_imag[0] + 1j * z_real_imag[1]
                w_p = w_prime(z_complex, Gamma)
                return [np.real(w_p), np.imag(w_p)]
            
            sol = root(func, [np.real(initial_guess), np.imag(initial_guess)], method='hybr')
            
            z_stag = sol.x[0] + 1j * sol.x[1]
            if (np.real(z_stag) ** 2 / a ** 2 + np.imag(z_stag) ** 2 / b1 ** 2) < 1:
                return None 
            return z_stag
            """

        st.code(code, language="python")

    with st.expander("Вычисление поля скоростей"):
        code = """
        
        Ux = np.ma.masked_invalid(Ux)
        Uy = np.ma.masked_invalid(Uy)

        speed = np.sqrt(np.abs(Ux)**2 + np.abs(Uy)**2)
        """

        st.code(code, language="python")

    with st.expander("Вычисление циркулляции по контуру"):
        code = """
        
        dz = (x_ellipse[1:] - x_ellipse[:-1]) + 1j * (y_ellipse[1:] - y_ellipse[:-1])
        velocity = w_prime(x_ellipse[:-1] + 1j * y_ellipse[:-1], Gamma)
        circulation = np.sum(velocity * dz)
        """

        st.code(code, language="python")

elif menu == "Решение методом конечных элементов (FEM)":

    r"""

    ##### Решение методом конечных элементов (FEM)

    """

    subtopic = st.selectbox(
        "Большая полуось ",  
        ["0.175", "0.225", "0.275"]  
    )

    if subtopic == "0.175":

        st.image("ellips_0175.png", caption="",use_container_width=True)

        r"""
        * Число ячеек сетки: 27298

        * Число узлов сетки: 14594

        ------------------------------------------------
        """
        st.image("ellips_solve_0175.png", caption="",use_container_width=True)
        r"""
        * Циркулляция: 9.88850e-06
        """

    elif subtopic == "0.225":

        st.image("ellips_0225.png", caption="",use_container_width=True)

        r"""
        * Число ячеек сетки: 31802

        * Число узлов сетки: 16635

        ------------------------------------------------
        """
        st.image("ellips_solve_0225.png", caption="",use_container_width=True)

        r"""
        * Циркулляция: 1.07842e-05
        """

    elif subtopic == "0.275":

        st.image("ellips_0275.png", caption="",use_container_width=True)

        r"""
        * Число ячеек сетки: 36174

        * Число узлов сетки: 18929

        ------------------------------------------------
        """
        st.image("ellips_solve_0275.png", caption="",use_container_width=True)

        r"""
        * Циркулляция: 2.25365e-05
        """

elif menu == "Циркулляция":

    r"""
    ##### Циркулляция

    """
    r"""

        | Полуоси эллипса                  |     $p=1$      |      $p=2$     |      $p=3$     |
        |----------------------------------|----------------|----------------|----------------|
        | $a = 0.175, b = 0.125$           |  $9.88850e-06$ | $-2.98645e-06$ | $-2.23483e-06$ |
        | $a = 0.225, b = 0.125$           |  $1.07842e-05$ | $-1.31345e-06$ | $-8.00204e-07$ |
        | $a = 0.275, b = 0.125$           |  $2.25365e-05$ |  $2.09678e-06$ |  $1.39354e-06$ |

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


    

    





    
