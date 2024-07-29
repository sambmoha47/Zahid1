FROM --platform=linux/x86-64 mambaorg/micromamba:1.4.9

# Install Conda/Mamba dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes && \
    find -name '*.a' -delete && \
    find -name '__pycache__' -type d -exec rm -rf '{}' '+' && \
    rm -rf $MAMBA_ROOT_PREFIX/conda-meta \
    $MAMBA_ROOT_PREFIX/include \
    $MAMBA_ROOT_PREFIX/lib/libpython*.so.* \
    $MAMBA_ROOT_PREFIX/lib/python*/idlelib \
    $MAMBA_ROOT_PREFIX/lib/libasan.so.* \
    $MAMBA_ROOT_PREFIX/lib/libtsan.so.* \
    $MAMBA_ROOT_PREFIX/lib/liblsan.so.* \
    $MAMBA_ROOT_PREFIX/lib/libubsan.so.* \
    $MAMBA_ROOT_PREFIX/bin/x86_64-conda-linux-gnu-ld \
    $MAMBA_ROOT_PREFIX/bin/sqlite3 \
    $MAMBA_ROOT_PREFIX/bin/openssl \
    $MAMBA_ROOT_PREFIX/share/terminfo

EXPOSE 9090

WORKDIR /home/app/

# Copy application files
COPY --chown=$MAMBA_USER:$MAMBA_USER ./.env /home/app/.env
COPY --chown=$MAMBA_USER:$MAMBA_USER ./.chainlit /home/app/.chainlit
COPY --chown=$MAMBA_USER:$MAMBA_USER ./chainlit.md /home/app/chainlit.md
COPY --chown=$MAMBA_USER:$MAMBA_USER ./src /home/app/src
COPY --chown=$MAMBA_USER:$MAMBA_USER ./secrets /home/app/secrets
COPY --chown=$MAMBA_USER:$MAMBA_USER ./data /home/app/data

USER $MAMBA_USER

ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

CMD ["/opt/conda/bin/python", "-m", "chainlit", "run", "src/app.py", "--port=9090", "--headless"]
