#!/bin/sh
set -eu

STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/srv/backups/gpu-task-dispatcher-$STAMP"
OLD_DISPATCHER="/srv/apps/gpu-dispatcher"
OLD_MONITOR="/srv/apps/ai-monitor-mvp"
NEW_APP="/srv/apps/gpu-task-dispatcher"
NEW_DATA="/srv/data/gpu-task-dispatcher"

sudo mkdir -p "$BACKUP_DIR" "$NEW_APP" "$NEW_DATA"
sudo tar -C /srv/apps -czf "$BACKUP_DIR/gpu-dispatcher.tgz" gpu-dispatcher
sudo tar -C /srv/apps -czf "$BACKUP_DIR/ai-monitor-mvp.tgz" ai-monitor-mvp
sudo docker cp gpu-dispatcher-dispatcher-1:/data/dispatcher.db "$BACKUP_DIR/dispatcher.db"
sudo cp "$BACKUP_DIR/dispatcher.db" "$NEW_DATA/dispatcher.db"
sudo tar -C "$NEW_APP" -xzf /tmp/gpu-task-dispatcher.tgz
sudo cp "$OLD_DISPATCHER/.env" "$NEW_APP/.env"
sudo chmod 600 "$NEW_APP/.env"

rollback() {
  cd "$NEW_APP" && sudo docker compose down || true
  cd "$OLD_DISPATCHER" && sudo docker compose up -d || true
  sudo systemctl enable --now ai-monitor.service || true
}
trap rollback INT TERM HUP

cd "$OLD_DISPATCHER"
sudo docker compose down
sudo systemctl disable --now ai-monitor.service

cd "$NEW_APP"
if ! sudo docker compose up -d --build; then
  rollback
  exit 1
fi

for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11435/health >/dev/null \
    && curl -fsS http://127.0.0.1:9999/ >/dev/null; then
    trap - INT TERM HUP
    echo "DEPLOY_OK backup=$BACKUP_DIR"
    exit 0
  fi
  sleep 2
done

rollback
echo "DEPLOY_FAILED_ROLLED_BACK" >&2
exit 1
