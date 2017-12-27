import tspy

WINDOW = 10
TICKER = 'AAPL'

X_train, y_train = tspy.data.ar.FinanceXy(
    TICKER, WINDOW, '2016-01-01', '2017-01-01')

dnn = tspy.model.ar.DNN(hidden_units=[100], window=WINDOW, name='dnn_model')

dnn.fit(X_train, y_train)
