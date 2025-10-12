# KMEANS Clustering

## Results

### Index scores

```
k: 5, c_index: 0.29, dunn_index: 0.12, davis_bouldin_index: 2.40
k: 7, c_index: 0.26, dunn_index: 0.12, davis_bouldin_index: 2.50
k: 9, c_index: 0.29, dunn_index: 0.11, davis_bouldin_index: 2.36
k: 10, c_index: 0.28, dunn_index: 0.11, davis_bouldin_index: 2.47
k: 12, c_index: 0.31, dunn_index: 0.11, davis_bouldin_index: 2.30
k: 15, c_index: 0.30, dunn_index: 0.11, davis_bouldin_index: 2.33
```

Good scores:
    C-index: small in [0, 1]
    Dunn-index: large in [0, inf]
    DB-Index: small in [0, inf] 

So the clustering quality seems to be ok according to the metrics but not stellar. 
The high dimensionality of the data can play a role in generating degraded scores. 

I compared with the KMEANS clustering used by sklearn, and the DB-Score was similar to
the ones observed in my results above. 
