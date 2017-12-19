if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from data import generator as gn
    from model import Model

    np.random.seed( 189 )
    window = 50
    N = 5000 + window

    ts = pd.DataFrame.from_csv('./data/in/SPY.csv', header=0, index_col=0, parse_dates=True)['Adj Close']
    
    _ticker = 'AAPL.OQ'
    _split_index = int( len(ts) * 0.6 )
    ts_train, ts_test = ts.iloc[:_split_index], ts.iloc[_split_index:]

    #model = Model(estimator='lstm', order=window, normalise=True)
    model = Model(estimator='ridge', order=window)

    epochs  = 1
    batch_size = 64

    #model.fit(ts_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.fit(ts_train)
    backtest = model.backtest(ts_test)
    backtest.plot(zscore=False, anomalies=True, baseline=True, title='SPY')
    print(backtest.summary())

    plt.show()
