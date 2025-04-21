FROM python:3.10-slim-buster

# ensure stdout/stderr are sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install Python dependencies first for better cache utilization
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# expose the application port
EXPOSE 8080

CMD ["python3", "app.py"]