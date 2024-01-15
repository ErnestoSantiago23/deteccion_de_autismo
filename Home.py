import streamlit as st
import pandas as pd
import numpy as np
from deteccion_de_autismo.interface.main_local import load_model
from deteccion_de_autismo.interface.main_local import predict

st.set_page_config(
    page_title="SpectrumInsight",
    page_icon="https://as1.ftcdn.net/v2/jpg/06/73/98/70/1000_F_673987016_XJuf04WTeSXl8zWRQgEsDEIs5lScsG5D.jpg",
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Aplicar el CSS
local_css("style.css")



st.markdown('<style>h1{color: whith;}</style>', unsafe_allow_html=True)
st.markdown('<h1>Welcome to SpectrumInsight! 👋</h1>', unsafe_allow_html=True)

st.markdown(
    """
    We are delighted that you are joining us on this journey of discovery and understanding.
    SpectrumInsight is designed to be an ally on the road to a better understanding of the autism spectrum.
    Here, you will find carefully crafted tools to help you better assess and understand each child's unique characteristics.

    Our mission is to provide a safe, informative and easy-to-use environment to support you in this important process.

"""
)

st.image('https://media.istockphoto.com/id/1468195688/es/foto/s%C3%ADmbolo-de-rompecabezas-de-color-de-la-conciencia-p%C3%BAblica-para-el-trastorno-del-espectro.jpg?s=2048x2048&w=is&k=20&c=zoM5HITgbAUD_kqctuKaSsEBm3LW81zKtNynB6d4jSE=',
         caption='Autism')
