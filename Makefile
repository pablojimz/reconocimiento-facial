.DEFAULT_GOAL := help

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3.11)
DATABASE ?= database/registration/users.csv
DATASET ?= database/enrollment
PROBES ?=
IMAGE ?= reconocimiento-facial:cpu
MODELS_DIR ?= $(CURDIR)/models

PROBE_ARGS = $(if $(PROBES),--probes "$(PROBES)")
SCRIPT = $(PYTHON) -u -m scripts
EVAL_ARGS = --dataset "$(DATASET)" --database "$(DATABASE)" $(PROBE_ARGS)

.PHONY: help venv models enroll calibrate thresholds evaluate-verification \
        evaluate-identification evaluate evaluate-images all test clean \
        docker-build docker-models docker-all docker-evaluate-images docker-shell

help:  ## List available commands
	@awk -F':.*## ' '/^[a-z-]+:.*## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv:  ## Install the project in a local virtual environment
	python3.11 -m venv .venv
	@if [ "$$(uname -s)" = Linux ]; then \
	    .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
	fi
	.venv/bin/python -m pip install -e '.[calibration]'

models:  ## Download the models
	$(PYTHON) -m reconocimiento_facial.cli models download --model all

enroll:  ## Enroll the image dataset
	$(SCRIPT).enroll_dataset --dataset "$(DATASET)" --database "$(DATABASE)"

calibrate:  ## Extract templates and generate calibration curves
	$(SCRIPT).calibrate_thresholds --dataset "$(DATASET)"

thresholds: outputs/calibration/templates.npz  ## Compute optimal thresholds
	$(SCRIPT).optimize_thresholds

outputs/calibration/templates.npz:
	$(MAKE) calibrate

evaluate-verification:  ## Evaluate 1:1 verification
	$(SCRIPT).evaluate_verification $(EVAL_ARGS)

evaluate-identification:  ## Evaluate 1:N identification
	$(SCRIPT).evaluate_identification $(EVAL_ARGS)

evaluate:  ## Evaluate both modes using the local gallery
	$(SCRIPT).evaluate_verification $(EVAL_ARGS)
	$(SCRIPT).evaluate_identification $(EVAL_ARGS)

evaluate-images:  ## Run the 91-image evaluation
	$(SCRIPT).evaluate_images

all: enroll  ## Enroll and evaluate the local gallery
	$(MAKE) evaluate

test:  ## Run tests
	$(PYTHON) -m unittest discover -s tests -v

clean:  ## Remove generated results and local enrollment
	rm -rf outputs/calibration "$(dir $(DATABASE))images"
	rm -f outputs/evaluation/images*.json outputs/evaluation/verification.json \
	      outputs/evaluation/identification.json outputs/evaluation/probes.npz \
	      "$(DATABASE)" "$(DATABASE).lock"

docker-build: docker-models  ## Build the CPU image and prepare the models
	docker build -t "$(IMAGE)" .

TTY := $(shell [ -t 0 ] && echo -it)
DOCKER_RUN = docker run --rm $(TTY) -u "$$(id -u):$$(id -g)" -e HOME=/tmp \
             -e BIOMETRIC_DB_PASSWORD
DOCKER_DATA = --mount "type=bind,source=$(CURDIR)/database,target=/app/database" \
              --mount "type=bind,source=$(CURDIR)/outputs,target=/app/outputs"
DOCKER_MODELS = --mount "type=bind,source=$(MODELS_DIR),target=/app/models,readonly"
DOCKER_MAKE = $(DOCKER_RUN) $(DOCKER_DATA) $(DOCKER_MODELS) --entrypoint make "$(IMAGE)"

docker-models:  ## Export the models to the host if needed
	@if test -s "$(MODELS_DIR)/retinaface-r50/upstream/models/retinaface.py" \
	    && test -s "$(MODELS_DIR)/retinaface-r50/weights/Resnet50_Final.pth" \
	    && test -s "$(MODELS_DIR)/transface-b/upstream/backbones/vit.py" \
	    && test -s "$(MODELS_DIR)/transface-b/weights/glint360k_model_TransFace_B.pt"; then \
	    printf '%s\n' "Models available at $(MODELS_DIR)."; \
	else \
	    docker build --target model-files --output "type=local,dest=$(MODELS_DIR)" .; \
	fi

docker-all:  ## Enroll and evaluate the local gallery in Docker
	mkdir -p outputs
	$(DOCKER_MAKE) all DATABASE="$(DATABASE)" DATASET="$(DATASET)" PROBES="$(PROBES)"

docker-evaluate-images:  ## Run the 91-image evaluation in Docker
	mkdir -p outputs
	$(DOCKER_MAKE) evaluate-images

docker-shell:  ## Open a Docker shell
	mkdir -p outputs
	$(DOCKER_RUN) $(DOCKER_DATA) $(DOCKER_MODELS) --entrypoint bash "$(IMAGE)"
