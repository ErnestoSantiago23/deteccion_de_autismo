FROM python:3.10
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["uvicorn", "deteccion_de_autismo.api.app:app", "--host", "0.0.0.0", "--port", "80"]
