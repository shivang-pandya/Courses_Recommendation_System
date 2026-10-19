FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py preprocess.py rl_environment.py dqn_agent.py ./
COPY templates ./templates
COPY static ./static
COPY models ./models
COPY data ./data

EXPOSE 5000

USER nobody

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]