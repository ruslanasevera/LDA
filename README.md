# Linear Discriminant Analysis on Stream of Features

Python implementation of LDA classifier expanded with **feature incrementation**. When a new feature is added to the model, the internal structure is being updated instead of retraing the model from the beginning. Multicollinearity is handled with regularization using a constant input regularization parameter. An inverse of a total covariance matrix is solved with the Cholesky decomposition.  
## Main methods 
```
add_feature
```
