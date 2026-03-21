FROM python:3.10-slim

ARG RUN_ID

RUN echo "Downloading model for Run ID: ${RUN_ID}"

CMD ["python", "--version"]