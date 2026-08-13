#!/bin/sh

set -eu

REDRHEX_TEST_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
REDRHEX_LAUNCHER_PATH=$(CDPATH= cd "$REDRHEX_TEST_DIR/.." && pwd)/redrhex_remote.command

if [ ! -f "$REDRHEX_LAUNCHER_PATH" ]; then
    printf 'Launcher not found: %s\n' "$REDRHEX_LAUNCHER_PATH" >&2
    exit 1
fi

REDRHEX_REMOTE_SOURCE_ONLY=1
export REDRHEX_REMOTE_SOURCE_ONLY
. "$REDRHEX_LAUNCHER_PATH"
unset REDRHEX_REMOTE_SOURCE_ONLY

redrhex_failure_count=0

assert_equal() {
    redrhex_actual=$1
    redrhex_expected=$2
    redrhex_name=$3
    if [ "$redrhex_actual" != "$redrhex_expected" ]; then
        printf 'FAIL: %s\n  expected: %s\n  actual:   %s\n' "$redrhex_name" "$redrhex_expected" "$redrhex_actual" >&2
        redrhex_failure_count=$((redrhex_failure_count + 1))
        return
    fi
    printf 'PASS: %s\n' "$redrhex_name"
}

assert_contains() {
    redrhex_value=$1
    redrhex_fragment=$2
    redrhex_name=$3
    case "$redrhex_value" in
        *"$redrhex_fragment"*)
            printf 'PASS: %s\n' "$redrhex_name"
            ;;
        *)
            printf 'FAIL: %s\n  missing: %s\n' "$redrhex_name" "$redrhex_fragment" >&2
            redrhex_failure_count=$((redrhex_failure_count + 1))
            ;;
    esac
}

assert_equal "$(redrhex_install_path /Users/Test/Desktop)" "/Users/Test/Desktop/RedRHex Remote.command" "install path"

redrhex_session_command=$(redrhex_remote_session_command)
assert_contains "$redrhex_session_command" 'python -m tools.training_panel --host 127.0.0.1 --port 8080' "panel command"
assert_contains "$redrhex_session_command" 'tmux new-session -d -s "$tmux_session"' "tmux fallback"
assert_contains "$redrhex_session_command" 'while :; do sleep 3600; done' "persistent remote session"

redrhex_encoded_command=$(redrhex_remote_command)
assert_contains "$redrhex_encoded_command" "printf %s '" "encoded command prefix"
assert_contains "$redrhex_encoded_command" "' | base64 -d | bash" "encoded command suffix"

redrhex_invoke_ssh() {
    printf '%s\n' "$@"
}

redrhex_ssh_arguments=$(redrhex_run_ssh_tunnel REMOTE_COMMAND)
redrhex_expected_arguments=$(cat <<'EOF'
-o
ExitOnForwardFailure=yes
-o
ServerAliveInterval=30
-o
ServerAliveCountMax=3
-L
8080:127.0.0.1:8080
-L
6006:127.0.0.1:6006
lab_user1@100.90.246.97
REMOTE_COMMAND
EOF
)
assert_equal "$redrhex_ssh_arguments" "$redrhex_expected_arguments" "SSH arguments"

redrhex_temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/redrhex-remote-test.XXXXXX")
trap 'rm -rf "$redrhex_temp_dir"' EXIT HUP INT TERM
mkdir "$redrhex_temp_dir/Desktop"
redrhex_install_launcher "$REDRHEX_LAUNCHER_PATH" "$redrhex_temp_dir/Desktop" >/dev/null
redrhex_installed_path="$redrhex_temp_dir/Desktop/RedRHex Remote.command"
if [ -x "$redrhex_installed_path" ] && cmp -s "$REDRHEX_LAUNCHER_PATH" "$redrhex_installed_path"; then
    printf 'PASS: launcher installation\n'
else
    printf 'FAIL: launcher installation\n' >&2
    redrhex_failure_count=$((redrhex_failure_count + 1))
fi

if [ "$redrhex_failure_count" -ne 0 ]; then
    printf '%s test(s) failed.\n' "$redrhex_failure_count" >&2
    exit 1
fi

printf 'All RedRHex Remote macOS tests passed.\n'
