FROM python:3.10
COPY requirements.txt requirements.txt
COPY model.pkl model.pkl
COPY data data
COPY deteccion_de_autismo deteccion_de_autismo
RUN pip install -r requirements.txt
CMD ["uvicorn", "deteccion_de_autismo.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
