import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
from deteccion_de_autismo.interface.main_local import load_model
from deteccion_de_autismo.interface.main_local import predict

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Aplicar el CSS
local_css("style.css")

model = load_model()

st.title("SpectrumInsight")

preguntas = ["Question 1: Choose the child's age range in months",
             "Question 2: Choose the child's sex",
             "Question 3: Choose the child's ethnicity",
             "Question 4: Does the child have a family member with ASD?",
             "Question 5: Does your child look at you when you call his/her name?",
             "Question 6: How easy is it for you to get eye contact with your child?",
             "Question 7: Does your child point to indicate that s/he wants something? (e.g. a toy that is out of reach)",
             "Question 8: Does your child point to share interest with you? (e.g. pointing at an interesting sight)",
             "Question 9: Does your child pretend? (e.g. care for dolls, talk on a toy phone)",
             "Question 10: Does your child follow where you're looking?",
             "Question 11: If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them? (e.g. stroking hair, hugging them)",
             "Question 12: Would you describe your child's first words as:",
             "Question 13: Does your child use simple gestures? (e.g. wave goodbye)",
             "Question 14: Does your child stare at nothing with no apparent purpose?"]


answer_5_and_11 = ["--","Always", "Usually", "Sometimes", "Rarely", "Never"]
answer_7_questions = ["--","Many times a day", "A few times a day", "A few times a week", "Less than once a week", "Never"]
answer_6 = ["--","Very easy", "Quite easy", "Quite difficult", "Very difficult", "Impossible"]
answer_12 = ["--","Very typical", "Quite typical", "Slightly unusual", "Very unusual", "My child doesn't speak"]
answer_1 = ["--","12-18", "18-24", "24-30", "30 or more"]
answer_2 = ["--","Male", "Female"]
answer_3 = ["--",'Middle eastern', 'White european', 'Hispanic', 'Black', 'Asian', 'South asian', 'Native indian', 'Pacifica', 'Latino', 'Mixed', 'Others']
answer_4 = ["--","Yes", "No"]


mapeo_respuestas = {"Always": 0, "Sometimes": 1, "Never": 1, "Rarely": 1, "Usually": 0,
                    "Many times a day": 0, "A few times a day": 0, "A few times a week": 1, "Less than once a week": 1,
                    "Very easy": 0, "Quite easy": 0, "Quite difficult": 1, "Very difficult": 1, "Impossible": 1,
                    "Very typical": 0, "Quite typical": 0, "Slightly unusual": 1, "Very unusual": 1,
                    "My child doesn't speak": 0, "12-18": 0, "18-24": 1, "24-30": 2, "30 or more": 3,
                    "Male": 1, "Female": 0, "Yes": 1, "No": 0, "Hispanic": 0,
                    "White european": 5, "Middle eastern": 8, "Black": 7, "Asian": 6, "South asian": 10,
                    "Native indian": 2, "Mixed": 9, "Latino": 1, "Pacifica": 4, "Others": 3}

# Lista para almacenar las respuestas del usuario
respuestas_usuario = []

# Iterar a través de las preguntas y obtener las respuestas del usuario
for i, pregunta in enumerate(preguntas):
    if "Question 1:" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_1)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))
    elif "Question 2" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_2)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))
    elif "Question 3" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_3)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))
    elif "Question 4" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_4)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))
    elif "Question 5" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_5_and_11)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))
    elif "Question 6" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_6)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))
    elif "Question 11" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_5_and_11)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))
    elif "Question 12" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_12)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))
    elif "Question 14" in pregunta:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_7_questions)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        if mapeo_respuestas.get(respuesta, respuesta) == 0:
            respuestas_usuario.append(1)
        else:
            respuestas_usuario.append(0)
    else:
        st.write(f"**{pregunta}**")
        respuesta = st.selectbox(f"Select an option for question {i+1}", answer_7_questions)
        st.write(f"*Selected Option for Question {i+1}:* {respuesta}")
        respuestas_usuario.append(mapeo_respuestas.get(respuesta, respuesta))


# Botón de predicción
if st.button("Predict"):
    # Verificar si todas las preguntas fueron respondidas
    if "--" in respuestas_usuario:
        # Mostrar mensaje de error si alguna pregunta no fue respondida
        st.error("Please answer all the questions before proceeding.")
    else:
        # Realizar la predicción usando tu modelo
        prediction =predict(model, np.array(respuestas_usuario).reshape(1,-1))
        proba = model.predict_proba(np.array(respuestas_usuario).reshape(1,-1))

        # Mostrar el resultado
        if prediction == 1:
            st.write("There is a {:.2%} chance that the child has autism".format(proba[0, 1]))
        else:
            st.write("There is a {:.2%} chance that the child does not have autism".format(proba[0, 0]))

        st.write("**Please do not take these results as infallible. Go to the doctor, if you have not already done so, and verify these results with a professional. In the graph below you can see the possibilities for each case.**")

        proba_flattened = proba.flatten() * 100

        classes = model.classes_
        fig, ax = plt.subplots()
        bars = ax.bar([0, 1], proba_flattened, tick_label=['No', 'Yes'], color=['lightcoral', 'lightgreen'], edgecolor='black')
        ax.set_xlabel('Classes')
        ax.set_ylabel('Probabilities')
        ax.set_title('Probabilities for Each Class')
        ax.set_facecolor('#04B2D9')
        fig.set_facecolor('#04B2D9')

        # Añadir etiquetas de porcentaje en las barras
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom')

        st.pyplot(fig)
