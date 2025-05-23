import streamlit as st

st.set_page_config(page_title="👋", layout="wide")

st.markdown("""
    <h1 style="text-align:center; font-size: 50px;">Филиал Московского государственного университета имени М. В. Ломоносова в городе Сарове</h1>
""", unsafe_allow_html=True)
st.markdown("""
    <h1 style="text-align:center; font-size: 40px;">Кафедра математики</h1>
""", unsafe_allow_html=True)
st.markdown("""
    <h1 style="text-align:center; font-size: 35px;">Группа ВМ - 124</h1>
""", unsafe_allow_html=True)

# Дополнительное изображение по центру
st.image("logo.jpg", width=300, use_container_width=True)

st.markdown("""
    <h1 style="text-align:center; font-size: 40px;">Численное исследование потенциального обтекания пучка цилиндров эллиптического сечения</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style="text-align:left; font-size: 35px;">Презентацию подготовили:</h1>
""", unsafe_allow_html=True)

# Данные участников
participants = [
    {"name": "Головня Никита", "photo": "0.jpg", "role": "Построение сеток в Gmsh, генерация геометрии, аналитическое решение для симметричного и несимметричного обтекания, численное решение (FEniCS) для симметричного и несимметричного обтекания, построение численного решения для симметричного и несимметричного обтекания пучка цилиндров (FEniCS),  программная реализация, визуализация, структура презентации."},
    {"name": "Александр Романенко", "photo": "1.jpg", "role": "Верификация аналитического и численного решения симметричного обтекания, оформление презентации."},
    {"name": "Гашигуллин Камиль", "photo": "2.jpg", "role": "Верификация аналитического и численного решения несимметричного обтекания, оформление презентации."},
    {"name": "Коврижных Анастасия", "photo": "3.jpg", "role": "Постановка задачи, геометрическая модель, математическая модель, конечно-элементная аппроксимация задачи, оформление презентации."},
    {"name": "Сержантов Артемий", "photo": "4.jpg", "role": "Переключение слайдов."},
]

# Вывод участников в две строки
row1 = participants[:3]
row2 = participants[3:]

cols1 = st.columns(3)
for i, participant in enumerate(row1):
    with cols1[i]:
        st.image(participant["photo"], width=200)
        st.markdown(f"""
            <h3 style="margin: 0; text-align: left;">{participant['name']}</h3>
            <p style="font-size: 16px; margin: 0; text-align: left;"><i>{participant['role']}</i></p>
        """, unsafe_allow_html=True)

cols2 = st.columns(3)
for i, participant in enumerate(row2):
    with cols2[i]:
        st.image(participant["photo"], width=200)
        st.markdown(f"""
            <h3 style="margin: 0; text-align: left;">{participant['name']}</h3>
            <p style="font-size: 16px; margin: 0; text-align: left;"><i>{participant['role']}</i></p>
        """, unsafe_allow_html=True)

st.markdown("""
    <h2 style="text-align:left;">О презентации</h2>
    <p style="text-align:left; font-size: 18px;">
        Исследование потенциального обтекания пучков цилиндров эллиптического сечения представляет собой
        важную задачу теоретической и прикладной аэрогидродинамики. Такие течения встречаются в
        широком спектре инженерных приложений, включая конструкции теплообменников, элементы
        гидродинамических систем и аэродинамические структуры. 
    </p>
    <p style="text-align:left; font-size: 18px;">
        Данный проект можно посмотреть и скачать на 
        <a href="https://github.com/nkt50i/cylinder" target="_blank" style="font-weight: bold;">
        GitHub</a>.
    </p>
    <p style="text-align:left; font-size: 18px;">
        Ознакомиться с основным истоником информации можно по ссылке 
        <a href="https://kpfu.ru//staff_files/F916745654/Maklakov_AM_D_Part1.pdf" target="_blank" style="font-weight: bold;">
        источник</a>.
    </p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔗 GitHub-репозиторий")
    st.image("qr_github.png", caption="cylinder", width=250)

with col2:
    st.subheader("📄 Источник")
    st.image("qr_paper.png", caption="Маклаков Д. В. Аналитические методы гидродинамики.", width=250)
