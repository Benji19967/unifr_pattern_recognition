include ../build_tools/poetry.mk

knn:
	python src/knn/main.py

kmeans:
	python src/kmeans/main.py

test:
	pytest tests/ -vvvv -s
