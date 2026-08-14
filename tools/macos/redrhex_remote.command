#!/bin/sh

REDRHEX_SSH_HOST="100.90.246.97"
REDRHEX_SSH_USER="lab_user1"
REDRHEX_PANEL_PORT="8080"
REDRHEX_TENSORBOARD_PORT="6006"
REDRHEX_PANEL_URL="http://localhost:8080/?remote_client=macos"
REDRHEX_TENSORBOARD_URL="http://localhost:6006"
REDRHEX_PANEL_ROOT="/home/lab_user1/Py/RedRHex"
REDRHEX_CONDA_INIT="/home/lab_user1/miniconda3/etc/profile.d/conda.sh"
REDRHEX_CONDA_ENVIRONMENT="env_isaaclab_bin"
REDRHEX_PANEL_SESSION="redrhex_panel"
REDRHEX_READINESS_TIMEOUT="45"
REDRHEX_MONITOR_PID=""

redrhex_remote_session_command() {
    cat <<EOF
set -eu

panel_url="http://127.0.0.1:${REDRHEX_PANEL_PORT}"
panel_root="${REDRHEX_PANEL_ROOT}"
conda_init="${REDRHEX_CONDA_INIT}"
conda_environment="${REDRHEX_CONDA_ENVIRONMENT}"
tmux_session="${REDRHEX_PANEL_SESSION}"
panel_log="\$panel_root/logs/training_panel/remote_panel.log"
panel_command="source \$conda_init && conda activate \$conda_environment && cd \$panel_root && exec python -m tools.training_panel --host 127.0.0.1 --port ${REDRHEX_PANEL_PORT}"

if curl -fsS --max-time 2 "\$panel_url" >/dev/null 2>&1; then
    echo "Training Panel is already running."
elif command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "\$tmux_session" 2>/dev/null; then
        echo "Training Panel session already exists: \$tmux_session"
    else
        tmux new-session -d -s "\$tmux_session" -- bash -lc "\$panel_command"
        echo "Started Training Panel in tmux session: \$tmux_session"
    fi
else
    mkdir -p "\$(dirname "\$panel_log")"
    nohup bash -lc "\$panel_command" >"\$panel_log" 2>&1 </dev/null &
    echo "Started Training Panel with nohup. Log: \$panel_log"
fi

attempts=${REDRHEX_READINESS_TIMEOUT}
while [ "\$attempts" -gt 0 ]; do
    if curl -fsS --max-time 2 "\$panel_url" >/dev/null 2>&1; then
        echo "Training Panel is ready at \$panel_url"
        while :; do sleep 3600; done
    fi
    sleep 1
    attempts=\$((attempts - 1))
done

echo "Training Panel did not become ready."
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "\$tmux_session" 2>/dev/null; then
    tmux capture-pane -p -t "\$tmux_session" -S -40 || true
elif [ -f "\$panel_log" ]; then
    tail -40 "\$panel_log" || true
fi
exit 1
EOF
}

redrhex_remote_command() {
    redrhex_encoded_command=$(redrhex_remote_session_command | base64 | tr -d '\r\n') || return 1
    printf "printf %%s '%s' | base64 -d | bash" "$redrhex_encoded_command"
}

redrhex_endpoint_ready() {
    curl -fsS --max-time 2 "$1" >/dev/null 2>&1
}

redrhex_open_pages() {
    open "$REDRHEX_PANEL_URL"
    if redrhex_endpoint_ready "$REDRHEX_TENSORBOARD_URL"; then
        open "$REDRHEX_TENSORBOARD_URL"
    else
        printf '%s\n' "TensorBoard is not running; start it from the Training Panel when needed."
    fi
}

redrhex_monitor_panel() {
    redrhex_attempt=0
    while [ "$redrhex_attempt" -lt "$REDRHEX_READINESS_TIMEOUT" ]; do
        if redrhex_endpoint_ready "$REDRHEX_PANEL_URL"; then
            redrhex_open_pages
            printf '%s\n' "RedRHex Remote is ready. Keep this Terminal window open."
            return 0
        fi
        sleep 1
        redrhex_attempt=$((redrhex_attempt + 1))
    done

    printf '%s\n' "The Training Panel did not respond within ${REDRHEX_READINESS_TIMEOUT} seconds." >&2
    return 1
}

redrhex_invoke_ssh() {
    ssh "$@"
}

redrhex_run_ssh_tunnel() {
    redrhex_remote_command_argument=$1
    redrhex_invoke_ssh \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "${REDRHEX_PANEL_PORT}:127.0.0.1:${REDRHEX_PANEL_PORT}" \
        -L "${REDRHEX_TENSORBOARD_PORT}:127.0.0.1:${REDRHEX_TENSORBOARD_PORT}" \
        "${REDRHEX_SSH_USER}@${REDRHEX_SSH_HOST}" \
        "$redrhex_remote_command_argument"
}

redrhex_install_path() {
    printf '%s/RedRHex Remote.command' "$1"
}

redrhex_install_launcher() {
    redrhex_source_path=$1
    redrhex_desktop_path=$2

    if [ ! -f "$redrhex_source_path" ]; then
        printf '%s\n' "Cannot locate the launcher script being installed: $redrhex_source_path" >&2
        return 1
    fi
    if [ ! -d "$redrhex_desktop_path" ]; then
        printf '%s\n' "macOS Desktop directory is unavailable: $redrhex_desktop_path" >&2
        return 1
    fi

    redrhex_destination=$(redrhex_install_path "$redrhex_desktop_path") || return 1
    if [ "$redrhex_source_path" != "$redrhex_destination" ]; then
        cp "$redrhex_source_path" "$redrhex_destination" || return 1
    fi
    chmod 700 "$redrhex_destination" || return 1

    printf '%s\n' "Installed RedRHex Remote."
    printf 'Launcher: %s\n' "$redrhex_destination"
}

redrhex_cleanup_monitor() {
    if [ -n "${REDRHEX_MONITOR_PID:-}" ]; then
        kill "$REDRHEX_MONITOR_PID" 2>/dev/null || true
        wait "$REDRHEX_MONITOR_PID" 2>/dev/null || true
        REDRHEX_MONITOR_PID=""
    fi
}

redrhex_require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Required macOS command is unavailable: %s\n' "$1" >&2
        return 1
    fi
}

redrhex_start_remote() {
    for redrhex_required_command in ssh curl base64 open; do
        redrhex_require_command "$redrhex_required_command" || return 1
    done

    if redrhex_endpoint_ready "$REDRHEX_PANEL_URL"; then
        printf '%s\n' "An existing RedRHex tunnel is ready; opening the browser pages."
        redrhex_open_pages
        return 0
    fi

    redrhex_command=$(redrhex_remote_command) || return 1
    printf '%s\n' "Connecting to ${REDRHEX_SSH_USER}@${REDRHEX_SSH_HOST}."
    printf '%s\n' "Verify the host and enter the Ubuntu password here if prompted."
    printf '%s\n' "Keep this Terminal window open while using RedRHex Remote."

    redrhex_monitor_panel &
    REDRHEX_MONITOR_PID=$!
    redrhex_run_ssh_tunnel "$redrhex_command"
    redrhex_ssh_status=$?
    redrhex_cleanup_monitor

    if [ "$redrhex_ssh_status" -ne 0 ]; then
        printf '%s\n' "SSH failed. Check Tailscale, authentication, the host-key prompt, and local port conflicts." >&2
        return "$redrhex_ssh_status"
    fi

    printf '%s\n' "The RedRHex SSH tunnel has closed."
}

redrhex_usage() {
    printf '%s\n' "Usage: redrhex_remote.command [--install|--help]"
}

redrhex_main() {
    case "${1:-}" in
        "")
            if [ "$#" -ne 0 ]; then
                redrhex_usage >&2
                return 2
            fi
            trap redrhex_cleanup_monitor 0
            trap 'exit 129' HUP
            trap 'exit 130' INT
            trap 'exit 143' TERM
            redrhex_start_remote
            ;;
        --install)
            if [ "$#" -ne 1 ]; then
                redrhex_usage >&2
                return 2
            fi
            redrhex_install_launcher "$0" "${REDRHEX_DESKTOP_DIR:-$HOME/Desktop}"
            ;;
        --help|-h)
            redrhex_usage
            ;;
        *)
            redrhex_usage >&2
            return 2
            ;;
    esac
}

if [ "${REDRHEX_REMOTE_SOURCE_ONLY:-0}" != "1" ]; then
    redrhex_main "$@"
fi
