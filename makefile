include ../build_tools/poetry.mk

knn:
	python src/classification/knn.py

kmeans:
	python src/clustering/kmeans.py

test:
	pytest tests/ -vvvv -s
