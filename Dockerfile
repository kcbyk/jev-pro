FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir numpy transformers \
 && useradd -r -m app
# deploy bundle: sunucu + TAM model paketi (config.json sayesinde HF hub'a gerek yok)
COPY serve_pro.py ./
COPY jev-pro-model-r3w2/ ./jev-pro-model/
RUN chown -R app /app
USER app
ENV OMP_NUM_THREADS=2
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=4s CMD python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8100/health',timeout=3)" || exit 1
ENV JEVPRO_KEYS="" JEVPRO_API_KEY=""
# anahtarlar: -v /yol/keys.json:/keys/keys.json:ro  +  -e JEVPRO_KEYS=/keys/keys.json
CMD ["sh","-c","exec python3 serve_pro.py --port 8100 --model-dir jev-pro-model --state jev-pro-model/ft_state_fp16.pt ${JEVPRO_KEYS:+--keys $JEVPRO_KEYS} ${JEVPRO_API_KEY:+--api-key $JEVPRO_API_KEY}"]
