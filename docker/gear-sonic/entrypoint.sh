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

exec "$@"
