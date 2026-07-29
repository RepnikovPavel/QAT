#!/usr/bin/env bash
# Install / refresh the cron entry that runs the grok-4.5 supervisor.
# The supervisor itself runs for up to MAX_RUNTIME (default 1h) per cron tick
# and round-robins the open QAT tasks. So a cron every 30 min keeps a near-
# continuous autonomous loop running 24/7.
set -euo pipefail
REPO="/home/user/ZCodeProject/QAT"
MARKER="# QAT-grok-supervisor"
# 55-min budget per tick (leaves slack before the next cron fires at :30)
CRON_LINE="*/30 * * * * MAX_RUNTIME=3300 GROK_TURNS=45 cd $REPO && bash scripts/supervisor.sh $MARKER"

( crontab -l 2>/dev/null | grep -v "$MARKER" ; echo "$CRON_LINE" ) | crontab -
echo "Installed supervisor cron:"
crontab -l | grep "$MARKER"
echo
echo "Manual one-shot (1h, 45 turns/chunk):"
echo "  cd $REPO && MAX_RUNTIME=3600 GROK_TURNS=45 bash scripts/supervisor.sh"
echo "Tail: tail -f $REPO/logs/supervisor.log"
