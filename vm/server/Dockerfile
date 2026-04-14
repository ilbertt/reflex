FROM alpine:latest

RUN apk add --no-cache nginx python3 curl wrk

# nginx config that we can modify at runtime
COPY nginx.conf /etc/nginx/nginx.conf
COPY default.conf /etc/nginx/http.d/default.conf

# Metrics endpoint (tiny python script)
COPY metrics.py /metrics.py

# Simple app that does fake "work"
COPY app.py /app.py

# Control script that applies config changes
COPY apply_config.py /apply_config.py

RUN mkdir -p /run/nginx /var/log/nginx /www
COPY index.html /www/index.html

EXPOSE 80 9100

CMD ["sh", "-c", "nginx && python3 /metrics.py & python3 /app.py"]
