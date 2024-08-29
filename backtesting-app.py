# app.py

import streamlit as st
import pandas as pd
import backtrader as bt
import yfinance as yf
import matplotlib
# Use a backend that doesn't display the plot to the user
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Function to run backtest and generate HTML report
def run_backtest(strategy_class, symbol, start_date, end_date, **params):
    cerebro = bt.Cerebro()
    cerebro.broker.setcommission(commission=0.00)

    data = bt.feeds.PandasData(dataname=yf.download(symbol, start=start_date, end=end_date, interval='1d'))
    cerebro.adddata(data)
    
    # Pass the parameters to the strategy
    cerebro.addstrategy(strategy_class, **params)
    
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    
    cerebro.broker.setcash(100.)
    # Run the backtest
    results = cerebro.run()
    strat = results[0]

    # Retrieve the returns from the analyzer
    returns = strat.analyzers.returns.get_analysis()

    # Convert the OrderedDict to a Pandas DataFrame
    returns_df = pd.DataFrame(list(returns.items()), columns=['Date', 'Return'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    returns_df.set_index('Date', inplace=True)
    
    plt.rcParams["figure.figsize"] = (10, 6)
    fig = cerebro.plot()
    return fig[0][0]

# Streamlit app
def main():
    st.title('Backtest Trading Strategies')

    # Import available strategies
    import strategies
    strategy_names = [name for name in dir(strategies) if name.endswith('Strategy')]
    selected_strategy = st.selectbox('Select Strategy', strategy_names)

    # Get the selected strategy class
    selected_strategy_class = getattr(strategies, selected_strategy)
    
    def to_number(s):
        n = float(s)
        return int(n) if n.is_integer() else n
    
    # Display input fields for strategy parameters
    strategy_params = {}
    for param_name in dir(selected_strategy_class.params):
        # Ignore special methods and attributes
        if not param_name.startswith("_") and param_name not in ['isdefault', 'notdefault']:
            # Get the parameter value using getattr
            param_value = getattr(selected_strategy_class.params, param_name)
            strategy_params[param_name] = st.text_input(f'{param_name}', value=param_value)
    
    strategy_params = {param_name: to_number(strategy_params[param_name]) for param_name in strategy_params}

    symbol = st.text_input('Enter symbol (e.g., BTC-USD, AMZN, ...):', 'BTC-USD')
    start_date = st.date_input('Select start date:', pd.to_datetime('2023-01-01'))
    end_date = st.date_input('Select end date:', pd.to_datetime('2023-12-31'))

    if st.button('Run Backtest'):
        st.write(f"Running backtest for {symbol} from {start_date} to {end_date} with {selected_strategy} strategy")
        fig = run_backtest(selected_strategy_class, symbol, start_date, end_date, **strategy_params)
    
        # Display the Plotly figure
        st.pyplot(fig)

if __name__ == '__main__':
    main()

