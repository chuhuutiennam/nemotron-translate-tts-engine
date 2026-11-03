PYTHON   ?= python3.12
REGISTRY ?= docker-registry:4000
VERSION  ?= 0.3.3

BASE_IMAGE ?= $(REGISTRY)/nemotron-translate-tts-base:$(VERSION)
APP_IMAGE  ?= $(REGISTRY)/nemotron-translate-tts-engine:$(VERSION)

.PHONY: test compile server proto \
        docker-build-base docker-build-app docker-build \
        docker-push-base  docker-push-app  docker-push \
        deploy-gpu deploy-cpu

# ── Dev ────────────────────────────────────────────────────────────────

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

compile:
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests

server:
	PYTHONPATH=src $(PYTHON) -m nemotron_testbed.main

proto:
	./scripts/generate_proto.sh

# ── Docker build ───────────────────────────────────────────────────────

docker-build-base:
	docker build -f Dockerfile.base \
	    -t $(BASE_IMAGE) \
	    .

docker-build-app:
	docker build -f Dockerfile \
	    --build-arg APP_BASE_IMAGE=$(BASE_IMAGE) \
	    -t $(APP_IMAGE) \
	    .

docker-build: docker-build-base docker-build-app

# ── Docker push ────────────────────────────────────────────────────────

docker-push-base:
	docker push $(BASE_IMAGE)

docker-push-app:
	docker push $(APP_IMAGE)

docker-push: docker-push-base docker-push-app

# ── Deploy (ConfigMap only — image không đổi) ──────────────────────────

deploy-gpu:
	kubectl apply -f k8s-configmap-gpu.yaml
	kubectl rollout restart deployment/nemotron-engine -n ims

deploy-cpu:
	kubectl apply -f k8s-configmap-cpu.yaml
	kubectl rollout restart deployment/nemotron-engine -n ims
