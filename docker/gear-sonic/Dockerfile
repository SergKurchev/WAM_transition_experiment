# GEAR-SONIC policy container.
# Uses the pre-built binary from mws-sim-gear-sonic-policy:latest.
# No source compilation — binary and all runtime deps are already in that image.
# Rebuild only if the upstream image changes.

FROM mws-sim-gear-sonic-policy:latest

# Override entrypoint to fix DDS lib path for x86_64 sim.
COPY docker/gear-sonic/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
