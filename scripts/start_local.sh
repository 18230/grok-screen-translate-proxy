#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/.grok2api.pid"
LOG_FILE="$ROOT_DIR/logs/local_server.log"
HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${SERVER_PORT:-18000}"
SESSION_NAME="${SCREEN_SESSION_NAME:-grok2api-local}"

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/data"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "端口 $PORT 已被占用，请先释放或设置 SERVER_PORT。"
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN
  exit 1
fi

if command -v screen >/dev/null 2>&1; then
  if screen -list | grep -q "[.]$SESSION_NAME[[:space:]]"; then
    echo "screen 会话 $SESSION_NAME 已存在，请先执行 scripts/stop_local.sh。"
    exit 1
  fi
else
  echo "未找到 screen，无法创建稳定后台会话。"
  exit 1
fi

if [[ ! -x "$ROOT_DIR/.venv/bin/granian" ]]; then
  echo "未找到 .venv/bin/granian，请先安装项目依赖。"
  exit 1
fi

if [[ ! -f "$ROOT_DIR/config.json" ]]; then
  echo "未找到 config.json，请先配置 Grok Cookie 和指纹。"
  exit 1
fi

screen -dmS "$SESSION_NAME" bash -lc "
  cd '$ROOT_DIR' || exit 1
  export SERVER_HOST='$HOST'
  export SERVER_PORT='$PORT'
  export SERVER_WORKERS='${SERVER_WORKERS:-1}'
  export DATA_DIR='${DATA_DIR:-./data}'
  export LOG_DIR='${LOG_DIR:-./logs}'
  export ACCOUNT_STORAGE='${ACCOUNT_STORAGE:-local}'
  export LOG_FILE_ENABLED='${LOG_FILE_ENABLED:-true}'
  export GROK_APP_API_KEY='${GROK_APP_API_KEY:-sk-local-test}'
  exec '$ROOT_DIR/.venv/bin/granian' \
    --interface asgi \
    --host '$HOST' \
    --port '$PORT' \
    --workers '${SERVER_WORKERS:-1}' \
    app.main:app \
    >> '$LOG_FILE' 2>&1
"

sleep 2

pid="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
if [[ -z "$pid" ]]; then
  echo "启动失败，最近日志："
  tail -n 80 "$LOG_FILE" || true
  screen -S "$SESSION_NAME" -X quit >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
  exit 1
fi
echo "$pid" > "$PID_FILE"

echo "grok2api 已启动：pid=$pid url=http://$HOST:$PORT"
echo "screen 会话：$SESSION_NAME"
echo "日志：$LOG_FILE"
