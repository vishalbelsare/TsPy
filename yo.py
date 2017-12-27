import tspy

WINDOW = 50
TICKER = 'AAPL'

X_train, y_train = tspy.data.ar.FinanceXy(
    TICKER, WINDOW, '2015-01-01', '2017-01-01')

dnn = tspy.model.ar.DNN(
    hidden_units=[50], window=WINDOW, name='dnn_model').fit(X_train, y_train, num_epochs=10000)

y_hat = dnn(X_train)

score = dnn(X_train, y_train)

ttt = dnn.next(10)

# # matplotlib backtest for missing $DISPLAY
# import matplotlib
# matplotlib.use('TkAgg')

# import matplotlib.pyplot as plt

# plt.plot(y_train, label='Original')
# plt.plot(y_hat, label='Model')
# plt.legend()
# plt.show()
