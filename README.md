```markdown
# Streamlit App for Backtesting Trading Strategies

This article provides a comprehensive guide to building a Streamlit application for backtesting trading strategies using Backtrader, yFinance, and Matplotlib.

## Code Overview

### Imports and Setup

The following Python libraries are used:
- `streamlit` for the web interface.
- `pandas` for data manipulation.
- `backtrader` for strategy backtesting.
- `yfinance` for financial data retrieval.
- `matplotlib` for plotting.

```python
import streamlit as st
import pandas as pd
import backtrader as bt
import yfinance as yf
import matplotlib
# Use a backend that doesn't display the plot to the user
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### Backtest Function

This function initializes Backtrader, sets up the trading environment, and executes the backtest.

```python
def run_backtest(strategy_class, symbol, start_date, end_date, **params):
    cerebro = bt.Cerebro()
    cerebro.broker.setcommission(commission=0.00)

    data = bt.feeds.PandasData(dataname=yf.download(symbol, start=start_date, end=end_date, interval='1d'))
    cerebro.adddata(data)
    
    cerebro.addstrategy(strategy_class, **params)
    
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    
    cerebro.broker.setcash(100.)
    results = cerebro.run()
    strat = results[0]

    returns = strat.analyzers.returns.get_analysis()
    returns_df = pd.DataFrame(list(returns.items()), columns=['Date', 'Return'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    returns_df.set_index('Date', inplace=True)
    
    plt.rcParams["figure.figsize"] = (10, 6)
    fig = cerebro.plot()
    return fig[0][0]
```

### Streamlit Application

The Streamlit app allows users to interactively select trading strategies, set parameters, and visualize backtest results.

```python
def main():
    st.title('Backtest Trading Strategies')

    import strategies
    strategy_names = [name for name in dir(strategies) if name.endswith('Strategy')]
    selected_strategy = st.selectbox('Select Strategy', strategy_names)

    selected_strategy_class = getattr(strategies, selected_strategy)
    
    def to_number(s):
        n = float(s)
        return int(n) if n.is_integer() else n
    
    strategy_params = {}
    for param_name in dir(selected_strategy_class.params):
        if not param_name.startswith("_") and param_name not in ['isdefault', 'notdefault']:
            param_value = getattr(selected_strategy_class.params, param_name)
            strategy_params[param_name] = st.text_input(f'{param_name}', value=param_value)
    
    strategy_params = {param_name: to_number(strategy_params[param_name]) for param_name in strategy_params}

    symbol = st.text_input('Enter symbol (e.g., BTC-USD, AMZN, ...):', 'BTC-USD')
    start_date = st.date_input('Select start date:', pd.to_datetime('2023-01-01'))
    end_date = st.date_input('Select end date:', pd.to_datetime('2023-12-31'))

    if st.button('Run Backtest'):
        st.write(f"Running backtest for {symbol} from {start_date} to {end_date} with {selected_strategy} strategy")
        fig = run_backtest(selected_strategy_class, symbol, start_date, end_date, **strategy_params)
    
        st.pyplot(fig)

if __name__ == '__main__':
    main()
```
![image](https://github.com/user-attachments/assets/07c5c7f5-4f66-4369-8f78-ce66a0e63d8d)

## Summary

This application provides a user-friendly interface to backtest various trading strategies using Backtrader. Users can select strategies, input parameters, and visualize the results interactively. Integrating Streamlit with Backtrader and yFinance facilitates easy experimentation with different trading strategies and data.
```

