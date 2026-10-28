#!/bin/sh
set -eu
: "${FASHION_USERNAME:?Set FASHION_USERNAME}"
: "${FASHION_PASSWORD_HASH:?Set FASHION_PASSWORD_HASH to a Caddy bcrypt hash}"
: "${RENDER_EXTERNAL_HOSTNAME:?Render must provide the external hostname}"
[ -d /data ] && grep -qs ' /data ' /proc/mounts || { echo 'A persistent disk must be mounted at /data' >&2; exit 1; }
caddy validate --config /app/Caddyfile --adapter caddyfile
caddy run --config /app/Caddyfile --adapter caddyfile &
exec python /app/app.py --host 127.0.0.1 --port 8765
