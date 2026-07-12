# Local append-only run evidence

Each run creates one immutable child directory. Raw evidence is intentionally
not committed by default because the real panel is gated and records can be
large. Preserve reportable run directories in the project evidence store; do
not overwrite or reuse a run id.

