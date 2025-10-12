# KNN Classifier

## Results

### Accuracy 
```
k: 1, distance: euclidean, accuracy: 0.886
k: 1, distance: manhattan, accuracy: 0.875
k: 3, distance: euclidean, accuracy: 0.88
k: 3, distance: manhattan, accuracy: 0.853
k: 5, distance: euclidean, accuracy: 0.866
k: 5, distance: manhattan, accuracy: 0.855
k: 10, distance: euclidean, accuracy: 0.858
k: 10, distance: manhattan, accuracy: 0.826
k: 15, distance: euclidean, accuracy: 0.83
k: 15, distance: manhattan, accuracy: 0.798
```

So the best classifier is with `k = 1` and `distance = euclidean`.

It is also interesting to note that the accuracy is always higher using
the eculidean distance vs the manhattan one. 

### Confusion matrix

Confusion matrix for `k = 1` and `distance = euclidean` with 
- `row: actual label (0, ..., 9)`
- `col: predicted label (0, ..., 9)`
```
[[ 99   0   0   0   0   0   2   0   0   0]
 [  0 121   0   0   0   0   0   0   0   0]
 [  2   5  78   1   1   0   1   6   1   0]
 [  1   1   1  86   1   8   0   0   3   0]
 [  0   1   0   1  78   0   1   1   1  14]
 [  0   1   0   6   2  85   1   0   1   3]
 [  0   3   0   0   2   0  83   0   0   0] 
 [  1   2   0   0   3   0   0  95   0   6]
 [  0   3   3   3   1   5   0   1  92   2]                                                                                                                                                                 
 [  0   0   0   1   5   2   0   4   0  69]]
```

The two most missclassified value is 9 predicted when 4 is the actual label.