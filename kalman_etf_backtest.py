import click
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from pandas_datareader.yahoo.daily import YahooDailyReader

from qstrader import settings
from qstrader.compat import queue
from qstrader.price_parser import PriceParser
from qstrader.price_handler.yahoo_daily_csv_bar import YahooDailyCsvBarPriceHandler
from qstrader.strategy.base import Strategies
from qstrader.position_sizer.naive import NaivePositionSizer
from qstrader.risk_manager.example import ExampleRiskManager
from qstrader.portfolio_handler import PortfolioHandler
from qstrader.compliance.example import ExampleCompliance
from qstrader.execution_handler.ib_simulated import IBSimulatedExecutionHandler

from kalman_etf_strategy import KalmanPairsTradingStrategy
from custom_tearsheet import CustomTearsheetStatistics as TearsheetStatistics
from custom_trading_session import CustomTradingSession as TradingSession
from kalman_filter_strategy_etf import KalmanFilterStrategyETF


def run(config, testing, tickers, start_date,
        end_date, filename, training,
        train_years_num, use_ols,
        benchmark, scale, returns):

    events_queue = queue.Queue()
    csv_dir = os.path.join(os.getcwd(), config.CSV_DATA_DIR)
    config.OUTPUT_DIR = os.path.join(os.getcwd(), config.OUTPUT_DIR)

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    kalman_filter = KalmanFilterStrategyETF(dim_x=2, dim_z=1)

    if training:

        train_start_date = (datetime.strptime(start_date, '%Y-%m-%d') -
                           relativedelta(years=train_years_num)).strftime('%Y-%m-%d')

        train_end_date = (datetime.strptime(start_date, '%Y-%m-%d') -
                         timedelta(days=1)).strftime('%Y-%m-%d')

        for ticker in tickers:
            YahooDailyReader(ticker, train_start_date, train_end_date).read() \
                .to_csv(os.path.join(csv_dir,
                                     "fit_%s" % train_years_num +
                                     "_%s" % start_date +
                                     "_%s.csv" % ticker))

        kalman_filter.fit_filter(path=csv_dir,
                                 tickers=tickers,
                                 n=train_years_num,
                                 start_date=start_date,
                                 use_ols=use_ols,
                                 returns=returns)

    for ticker in tickers + [benchmark]:    
        YahooDailyReader(ticker, start_date, end_date).read() \
            .to_csv(os.path.join(csv_dir,
                                 "%s.csv" % ticker))
    initial_equity = PriceParser.parse(100000.00)

    price_handler = YahooDailyCsvBarPriceHandler(
        csv_dir, events_queue, tickers + [benchmark]
    )

    strategy = KalmanPairsTradingStrategy(tickers, events_queue,
                                          kalman_filter, scale,
                                          returns)
    strategy = Strategies(strategy)

    position_sizer = NaivePositionSizer()

    risk_manager = ExampleRiskManager()

    portfolio_handler = PortfolioHandler(
        initial_equity, events_queue, price_handler,
        position_sizer, risk_manager
    )

    compliance = ExampleCompliance(config)

    execution_handler = IBSimulatedExecutionHandler(
        events_queue, price_handler, compliance
    )

    statistics = TearsheetStatistics(
        config, portfolio_handler, title="",
        benchmark=benchmark
    )

    backtest = TradingSession(
        config, strategy, tickers,
        initial_equity,
        start_date, end_date,
        events_queue,
        statistics=statistics,
        execution_handler=execution_handler,
        price_handler=price_handler,
        portfolio_handler=portfolio_handler,
        position_sizer=position_sizer,
        risk_manager=risk_manager,
        benchmark=benchmark
    )
    results = backtest.start_trading(testing=testing)
    statistics.save(filename)
    return results

# @click.command()
# @click.option('--config', default=settings.DEFAULT_CONFIG_FILENAME, help='Config filename')
# @click.option('--testing/--no-testing', default=False, help='Enable testing mode')
# @click.option('--tickers', default='SP500TR', help='Tickers (use comma)')
# @click.option('--start_date', default='2010-01-01', help='Start Date of Backtest')
# @click.option('--end_date', default='2021-08-13', help='End Date of Backtest')
# @click.option('--filename', default='', help='Pickle (.pkl) statistics filename')
# @click.option('--training/--no-training', default=True, help='Enable fitting Kalman filter mode')
# @click.option('--train_years_num', default=1,
#              help='Number of previous years for Kalman Filter fitting via Linear Regression Results')
# @click.option('--use_ols/--no-use_ols', default=False,
#              help='Use OLS Estimate of State for Kalman Filter fitting via Linear Regression Results')
# @click.option('--benchmark', default='GOVT', help='Benchmark for the Strategy')
# @click.option('--scale', default=1.0, help='Range of SDs to comare with measurement errors')
# @click.option('--returns/--no-returns', default=False, help='Use ETF returns or prices')
def backtest(config=settings.DEFAULT_CONFIG_FILENAME,
         testing=False, 
         tickers="SP500TR",
         start_date='2010-01-01',
         end_date='2021-08-13',
         filename='',
         training=True,
         train_years_num=1,
         use_ols=False,
         benchmark="GOVT",
         scale=1,
         returns=False):

    tickers = tickers.split(",")
    config = settings.from_file(config, testing)
    train_years_num = int(train_years_num)
    return run(config, testing, tickers, start_date,
        end_date, filename, training,
        train_years_num, use_ols,
        benchmark, scale, returns)


if __name__ == "__main__":
    backtest()
