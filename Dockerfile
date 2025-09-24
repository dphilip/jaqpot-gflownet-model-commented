FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html

EXPOSE 8000

CMD ["python", "-m", "main", "--port", "8000"]
