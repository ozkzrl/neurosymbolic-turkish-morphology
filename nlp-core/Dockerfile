FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# İndirme krizlerini engellemek için pip zaman aşımı süresini uzatıyoruz
ENV PIP_DEFAULT_TIMEOUT=1000

# Ekstra güvenli indirme flagleri ile kurulumu tetikliyoruz
RUN pip install --no-cache-dir --retries 10 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
