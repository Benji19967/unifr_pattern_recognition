include ../build_tools/poetry.mk

knn:
	python src/classification/knn.py

kmeans:
	python src/clustering/kmeans.py

svm:
	python src/classification/svm.py

test:
	pytest tests/ -vvvv -s
