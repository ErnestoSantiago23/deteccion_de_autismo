import streamlit as st

st.set_page_config(
    page_title="SpectrumInsight",
    page_icon="https://as1.ftcdn.net/v2/jpg/06/73/98/70/1000_F_673987016_XJuf04WTeSXl8zWRQgEsDEIs5lScsG5D.jpg",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Aplicar el CSS
local_css("style.css")

st.title("Recommendations for Suspected Autism")
st.write("""
    If you suspect your child might be on the autism spectrum, it's important to take certain steps to ensure you get the necessary support and resources. Here are some recommendations:

    1. **Consult a Professional**: The most important step is to consult with a pediatrician or a child development specialist. They can guide you on the next steps to take.

    2. **Observe Your Child's Behavior**: Take note of any unusual behavior or characteristics. This can be useful during medical consultations.

    3. **Seek Support and Resources**: There are many organizations and support groups for families with children on the autism spectrum. These resources can be very helpful.

    4. **Education and Awareness**: Learning about autism will help you better understand what your child might be experiencing and how you can support them.

    5. **Plan for the Future**: Consider your child’s long-term needs, including education, therapy, and social support.

    Remember, every child is unique, and the autism spectrum is very broad. What works for one child may not be suitable for another.
    """)
