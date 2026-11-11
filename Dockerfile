# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy only necessary files for installation (hatch_build.py compiles the native Smith-Waterman kernel)
COPY pyproject.toml README.md LICENSE.md hatch_build.py ./
COPY src/ ./src/

# A failed kernel compile fails the image build instead of producing a pure-Python install
ENV FOLDMATCH_REQUIRE_NATIVE=1

# Install build dependencies, build (non-editable; the hook compiles the kernel) and run the kernel selftest
# (DP golden hash, BLAS path within the S16 tolerance). Kept apart from the cleanup below, whose `|| true` would
# otherwise swallow a failed install or selftest.
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ binutils && \
    pip install --no-cache-dir . && \
    python -m foldmatch.search.swkern

# Strip, clean up and remove the build dependencies in one layer
# hadolint ignore=SC2015
RUN find /usr/local/lib/python3.12 -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -type f -executable -exec strip --strip-unneeded {} \; 2>/dev/null || true && \
    # Remove unnecessary files in builder stage
    find /usr/local/lib/python3.12 -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12 -name "*.pyc" -delete && \
    find /usr/local/lib/python3.12 -name "*.pyo" -delete && \
    # Remove build dependencies immediately
    apt-get purge -y gcc g++ binutils && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy the cleaned packages from builder (a regular install: the source tree in /app is not needed at runtime)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Final cleanup for runtime image
# hadolint ignore=SC2015
RUN find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name 'en*' -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/share/i18n/locales -mindepth 1 -maxdepth 1 ! -name 'en_*' -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/* 2>/dev/null || true

# The stripped kernel libraries work in the runtime image, which has no compiler
RUN python -m foldmatch.search.swkern > /dev/null

ENTRYPOINT ["inference"]
