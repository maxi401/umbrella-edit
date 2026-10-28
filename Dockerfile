FROM caddy:2 AS caddy
FROM python:3.12-slim
COPY --from=caddy /usr/bin/caddy /usr/bin/caddy
WORKDIR /app
COPY requirements-hosted.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.12.1' && \
    pip install --no-cache-dir -r requirements-hosted.txt
COPY app.py segment_archive.py Caddyfile start-hosted.sh ./
COPY static ./static
ENV FASHION_HOSTED=1 FASHION_DATA_DIR=/data FASHION_SOURCE_DIR=/data/source HF_HOME=/data/model HF_HUB_OFFLINE=0
EXPOSE 10000
CMD ["sh", "/app/start-hosted.sh"]
