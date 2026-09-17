PYTHON ?= python

.PHONY: data preprocess train evaluate test app docker-build docker-run all clean

data:
	$(PYTHON) src/get-data.py

preprocess:
	$(PYTHON) src/preprocess.py

train:
	$(PYTHON) src/train.py

evaluate:
	$(PYTHON) src/evaluate.py

test:
	$(PYTHON) -m pytest tests/ -v

app:
	streamlit run app/app.py

docker-build:
	docker build -t bias-in-bios-app .

docker-run:
	docker run --rm -p 8501:8501 bias-in-bios-app

all: data preprocess train evaluate test

clean:
	rm -rf data/raw/* data/processed/* models/*
