# Immutable freeze snapshot

`confirmatory_freeze.json` is created exactly once at the unique path declared
by `configs/confirmatory.json`. It is ignored until the authorized freeze step;
do not create it during development.
