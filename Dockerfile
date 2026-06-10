FROM public.ecr.aws/lambda/python:3.9 AS builder

RUN yum install -y gcc-c++ && \
	pip install uv wheel && \
	yum clean all && \
	rm -rf /var/cache/yum

WORKDIR /app
COPY requirements.in .
RUN uv pip compile requirements.in -o requirements.txt --python-version 3.9

RUN pip wheel --wheel-dir=/app/wheels -r requirements.txt

FROM public.ecr.aws/lambda/python:3.9

WORKDIR /var/task

COPY --from=builder /app/requirements.txt .
COPY --from=builder /app/wheels /wheels/

RUN pip install -r requirements.txt --no-index --find-links /wheels/

RUN rm -rf /wheels requirements.txt

COPY src/ .
