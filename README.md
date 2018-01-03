# TsPy
> Python Time Series Prediction Framework

## Setup

### macOS

* `source scripts/setup.sh`

## Autoregressive Models

### Theory

Under the assumption that future values of a time series
<img src="https://latex.codecogs.com/gif.latex?\mathbf{x}" title="\mathbf{x}" />
depend on its past values, the **AR(n)** (n-th order Generalised Autoregressive Model) is given by:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;\fn_phv&space;\small&space;\mathbf{x}(t)&space;=&space;\alpha_{1}\phi(\mathbf{x}(t-1))&space;&plus;&space;\alpha_{2}\phi(\mathbf{x}(t-2))&space;&plus;&space;\cdots&space;&plus;&space;\alpha_{n}\phi(\mathbf{x}(t-n))&space;=&space;\sum_{i=1}^{n}&space;\alpha_{i}\phi(\mathbf{x}(t-i))&space;=&space;\boldsymbol{\alpha}^{T}&space;\Phi_{x}" title="\small \mathbf{x}(t) = \alpha_{1}\phi(\mathbf{x}(t-1)) + \alpha_{2}\phi(\mathbf{x}(t-2)) + \cdots + \alpha_{n}\phi(\mathbf{x}(t-n)) = \sum_{i=1}^{n} \alpha_{i}\phi(\mathbf{x}(t-i)) = \mathbf{\alpha}^{T} \Phi_{x}" />

where:

* <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;\fn_phv&space;\small&space;t" title="\small t" />: the time index

* <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;\fn_phv&space;\small&space;\mathbf{x}(t)" title="\small \mathbf{x}(t)" />: time series value at time <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;\fn_phv&space;\small&space;t" title="\small t" />

* <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;\fn_phv&space;\small&space;\boldsymbol{\alpha}&space;=&space;[\alpha_{1},&space;\alpha_{2},&space;\cdots,&space;\alpha_{n}]" title="\small \boldsymbol{\alpha} = [\alpha_{1}, \alpha_{2}, \cdots, \alpha_{n}]" />: the autoregression coefficients (parameters, weights)

* <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{150}&space;\fn_phv&space;\small&space;\Phi" title="\small \Phi" />: any kernel function

### Data Preprocessing

Formulate autoregression as a supervised learning problem where:

* features `X`: the `n` past values of `ts`

* target `y`: the `n+1`th value of `ts`

```python
# raw time series
N = 100
ts = time_series__generator(N=N)

# AR(n)
n = 10
X, y = tspy.data.ar.Xy(series=ts, window=n)

# data shape
assert(X.shape == [N - n + 1, n])
assert(y.shape == [N - n + 1, 1])
```

### Models

#### `DNN`

**Deep Neural Network**, backed by `tensorflow.layers.DNNRegressor`.

```python
# hyperparameters
n = 10
HIDDEN_UNITS = [50]
NUM_EPOCHS = 5000

# regressor / model
dnn = tspy.model.ar.DNN(
    hidden_units=HIDDEN_UNITS,
    window=n
).fit(X, y, num_epochs=NUM_EPOCHS)

# predictions
y_hat = dnn.predict(X)
y_hat = dnn(X)

# model evaluation
mse = dnn.score(X, y)
mse = dnn(X, y)
```

#### `SKReg`

**`scikit-learn` Regressor**, backed by `sklearn`.

```python
# hyperparameters
n = 10
REGRESSOR_TYPE = 'ridge'

# regressor / model
reg = tspy.model.ar.SKReg(
    regressor_type=REGRESSOR_TYPE,
    window=n
).fit(X_train, y_train)

# predictions
y_hat = reg.predict(X)
y_hat = reg(X)

# model evaluation
mse = reg.score(X, y)
mse = reg(X, y)
```
