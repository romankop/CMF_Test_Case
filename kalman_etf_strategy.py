from math import floor

import numpy as np

from qstrader.price_parser import PriceParser
from qstrader.event import (SignalEvent, EventType)
from qstrader.strategy.base import AbstractStrategy
from kalman_filter_strategy_etf import KalmanFilterStrategyETF


class KalmanPairsTradingStrategy(AbstractStrategy):

    def __init__(
        self, tickers, events_queue, kalman_filter, scale, returns
    ):
        self.tickers = tickers
        self.events_queue = events_queue
        self.time = None
        self.latest_prices = np.array([-1.0, -1.0])
        self.invested = None

        self.days = 0
        self.qty = 2000
        self.cur_hedge_qty = self.qty
        self.scale = scale
        self.returns = returns

        self.prev_prices = None

        self.filter = kalman_filter

    def _set_correct_time_and_price(self, event):

        if self.time is None:
            self.time = event.time

        price = event.adj_close_price/PriceParser.PRICE_MULTIPLIER
        if event.time == self.time:
            if event.ticker == self.tickers[0]:
                self.latest_prices[0] = price
            else:
                self.latest_prices[1] = price
        else:
            self.time = event.time
            self.days += 1
            self.latest_prices = np.array([-1.0, -1.0])
            if event.ticker == self.tickers[0]:
                self.latest_prices[0] = price
            else:
                self.latest_prices[1] = price

    def calculate_signals(self, event):

        if event.type == EventType.BAR:
            self._set_correct_time_and_price(event)

            if all(self.latest_prices > -1.0):

                if self.returns:

                    if self.prev_prices is None:
                        self.prev_prices = self.latest_prices
                        return
                    else:
                        self.latest_prices, self.prev_prices = (
                            self.latest_prices / self.prev_prices - 1,
                            self.latest_prices
                        )

                H = np.array([[self.latest_prices[0], 1.0]])
                z = self.latest_prices[1]

                e, sigma = self.filter.update(z, H=H)

                if self.days > 1:
                    if self.invested is None:
                        if e < - self.scale * sigma:
                            print("LONG: %s" % event.time)
                            self.cur_hedge_qty = int(floor(self.qty * self.filter.x[0]))
                            self.events_queue.put(SignalEvent(self.tickers[1], "BOT", self.qty))
                            self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.cur_hedge_qty))
                            self.invested = "long"
                        elif e > self.scale * sigma:
                            print("SHORT: %s" % event.time)
                            self.cur_hedge_qty = int(floor(self.qty * self.filter.x[0]))
                            self.events_queue.put(SignalEvent(self.tickers[1], "SLD", self.qty))
                            self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.cur_hedge_qty))
                            self.invested = "short"
                    if self.invested is not None:
                        if self.invested == "long" and e > - self.scale * sigma:
                            print("CLOSING LONG: %s" % event.time)
                            self.events_queue.put(SignalEvent(self.tickers[1], "SLD", self.qty))
                            self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.cur_hedge_qty))
                            self.invested = None
                        elif self.invested == "short" and e < self.scale * sigma:
                            print("CLOSING SHORT: %s" % event.time)
                            self.events_queue.put(SignalEvent(self.tickers[1], "BOT", self.qty))
                            self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.cur_hedge_qty))
                            self.invested = None