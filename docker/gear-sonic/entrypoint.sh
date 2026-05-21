#!/usr/bin/env bash
set -eo pipefail

# DDS multicast on loopback (matches sim container)
ip link set lo multicast on 2>/dev/null || true
ip route add 224.0.0.0/4 dev lo 2>/dev/null || true

# Socket buffer optimization for DDS
sysctl -w net.core.rmem_max=67108864 2>/dev/null || true
sysctl -w net.core.rmem_default=67108864 2>/dev/null || true

# Remove healthcheck sentinel from any previous run so Docker's healthcheck
# always reflects the current process.
rm -f /tmp/gear-sonic-ready

# The upstream image ships aarch64 DDS libs in /usr/local/lib (wrong for x86_64 sim).
# Use x86_64 .so files from the bind-mounted gear_sonic_deploy volume instead.
# Create versioned symlinks (.so.0) on first start if they don't exist yet.
DDS_X86=/opt/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64
if [ -d "${DDS_X86}" ]; then
    [ -f "${DDS_X86}/libddsc.so"   ] && [ ! -e "${DDS_X86}/libddsc.so.0"   ] \
        && ln -sf libddsc.so   "${DDS_X86}/libddsc.so.0"
    [ -f "${DDS_X86}/libddscxx.so" ] && [ ! -e "${DDS_X86}/libddscxx.so.0" ] \
        && ln -sf libddscxx.so "${DDS_X86}/libddscxx.so.0"
    export LD_LIBRARY_PATH="${DDS_X86}:${LD_LIBRARY_PATH:-}"
fi

exec "$@"
