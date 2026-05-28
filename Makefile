IMAGE=ghcr.io/gyorgygutai/flux-klein-worker
VERSION=v1.0.10

venv:
	python3 -m venv venv
	./venv/bin/pip install pydantic

build: venv
	./venv/bin/python export_schema.py
	docker build --platform linux/amd64 -t $(IMAGE):$(VERSION) .

push:
	docker push $(IMAGE):$(VERSION)

deploy: build push