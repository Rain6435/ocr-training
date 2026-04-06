.PHONY: install run-api run-dashboard run-tensorboard docker-up docker-down test train-classifier train-ocr benchmark benchmark-google ablation error-analysis lint clean

install:
	pip install -r requirements.txt

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

run-tensorboard:
	tensorboard --logdir logs/tensorboard --port 6006 --bind_all

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

test:
	pytest tests/ -v

train-classifier:
	python -m src.classifier.train

train-ocr:
	python -m src.ocr.custom_model.train

benchmark:
	python -m src.evaluation.benchmark

benchmark-google:
	python scripts/run_google_benchmark.py

ablation:
	python -m src.evaluation.ablation

error-analysis:
	python -m src.evaluation.error_analysis

lint:
	python -m py_compile src/config.py
	python -m py_compile src/preprocessing/pipeline.py
	python -m py_compile src/classifier/model.py
	python -m py_compile src/ocr/custom_model/architecture.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf logs/pipeline/*
