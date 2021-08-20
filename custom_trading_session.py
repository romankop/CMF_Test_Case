from qstrader.trading_session import TradingSession

class CustomTradingSession(TradingSession):

    def __init__(
        self, config, strategy, tickers,
        equity, start_date, end_date, events_queue,
        session_type="backtest", end_session_time=None,
        price_handler=None, portfolio_handler=None,
        compliance=None, position_sizer=None,
        execution_handler=None, risk_manager=None,
        statistics=None, sentiment_handler=None,
        title=None, benchmark=None
    ):
        super().__init__(config, strategy, tickers,
                         equity, start_date,
                         end_date, events_queue,
                         session_type=session_type,
                         end_session_time=end_session_time,
                         price_handler=price_handler,
                         portfolio_handler=portfolio_handler,
                         compliance=compliance,
                         position_sizer=position_sizer,
                         execution_handler=execution_handler,
                         risk_manager=risk_manager,
                         statistics=statistics,
                         sentiment_handler=sentiment_handler,
                         title=title, benchmark=benchmark)

    def start_trading(self, testing=False):

        self._run_session()
        results = self.statistics.get_results()
        print("---------------------------------")
        print("Backtest complete.")
        print("Sharpe Ratio: %0.2f" % results["sharpe"])
        print(
            "Max Drawdown: %0.2f%%" % (
                results["max_drawdown_pct"] * 100.0
            )
        )
        print(
            "Cumulative Return: %0.2f" %  (
                float(results["cum_returns"][-1]) - 1.0
            )
        )
        print(
            "Cumulative Return / Max Drawdown: %0.2f" % (
                float(results["cum_ret/max_dd"])
            )
        )

        return results