from qstrader.statistics.base import AbstractStatistics
from qstrader.price_parser import PriceParser

from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from datetime import datetime

import qstrader.statistics.performance as perf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
import os


def create_sharpe_ratio(returns, periods=252, benchmark=None):
    if benchmark is None:
        return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)
    else:
        return np.sqrt(periods) * (np.mean(returns - benchmark)) / np.std(returns)


class CustomTearsheetStatistics(AbstractStatistics):

    def __init__(
            self, config, portfolio_handler,
            title=None, benchmark=None, periods=252,
            rolling_sharpe=False
    ):

        self.config = config
        self.portfolio_handler = portfolio_handler
        self.price_handler = portfolio_handler.price_handler
        self.title = '\n'.join(title)
        self.benchmark = benchmark
        self.periods = periods
        self.rolling_sharpe = rolling_sharpe
        self.equity = {}
        self.equity_benchmark = {}
        self.log_scale = False

    def update(self, timestamp, portfolio_handler):

        self.equity[timestamp] = PriceParser.display(
            self.portfolio_handler.portfolio.equity
        )
        if self.benchmark is not None:
            self.equity_benchmark[timestamp] = PriceParser.display(
                self.price_handler.get_last_close(self.benchmark)
            )

    def get_results(self):

        equity_s = pd.Series(self.equity).sort_index()

        returns_s = equity_s.pct_change().fillna(0.0)

        rolling = returns_s.rolling(window=self.periods)
        rolling_sharpe_s = np.sqrt(self.periods) * (
                rolling.mean() / rolling.std()
        )

        cum_returns_s = np.exp(np.log(1 + returns_s).cumsum())

        dd_s, max_dd, dd_dur = perf.create_drawdowns(cum_returns_s)

        statistics = {}

        statistics["sharpe"] = create_sharpe_ratio(
            returns_s, self.periods
        )
        statistics["drawdowns"] = dd_s
        statistics["max_drawdown"] = max_dd
        statistics["max_drawdown_pct"] = max_dd
        statistics["max_drawdown_duration"] = dd_dur
        statistics["equity"] = equity_s
        statistics["returns"] = returns_s
        statistics["rolling_sharpe"] = rolling_sharpe_s
        statistics["cum_returns"] = cum_returns_s
        statistics["cum_ret/max_dd"] = (
                                               float(statistics["cum_returns"][-1]) - 1) / \
                                       float(statistics["max_drawdown_pct"] + 1e-32
                                             )

        positions = self._get_positions()
        if positions is not None:
            statistics["positions"] = positions

        if self.benchmark is not None:
            equity_b = pd.Series(self.equity_benchmark).sort_index()
            returns_b = equity_b.pct_change().fillna(0.0)
            rolling_b = returns_b.rolling(window=self.periods)
            rolling_sharpe_b = np.sqrt(self.periods) * (
                    rolling_b.mean() / rolling_b.std()
            )
            cum_returns_b = np.exp(np.log(1 + returns_b).cumsum())
            dd_b, max_dd_b, dd_dur_b = perf.create_drawdowns(cum_returns_b)
            statistics["sharpe_b"] = create_sharpe_ratio(returns_b)
            statistics["sharpe"] = create_sharpe_ratio(
                returns_s, self.periods, returns_b)
            statistics["drawdowns_b"] = dd_b
            statistics["max_drawdown_pct_b"] = max_dd_b
            statistics["max_drawdown_duration_b"] = dd_dur_b
            statistics["equity_b"] = equity_b
            statistics["returns_b"] = returns_b
            statistics["rolling_sharpe_b"] = rolling_sharpe_b
            statistics["cum_returns_b"] = cum_returns_b

        return statistics

    def _get_positions(self):

        def x(p):
            return PriceParser.display(p)

        pos = self.portfolio_handler.portfolio.closed_positions
        a = []
        for p in pos:
            a.append(p.__dict__)
        if len(a) == 0:
            return None
        else:
            df = pd.DataFrame(a)
            df['avg_bot'] = df['avg_bot'].apply(x)
            df['avg_price'] = df['avg_price'].apply(x)
            df['avg_sld'] = df['avg_sld'].apply(x)
            df['cost_basis'] = df['cost_basis'].apply(x)
            df['init_commission'] = df['init_commission'].apply(x)
            df['init_price'] = df['init_price'].apply(x)
            df['market_value'] = df['market_value'].apply(x)
            df['net'] = df['net'].apply(x)
            df['net_incl_comm'] = df['net_incl_comm'].apply(x)
            df['net_total'] = df['net_total'].apply(x)
            df['realised_pnl'] = df['realised_pnl'].apply(x)
            df['total_bot'] = df['total_bot'].apply(x)
            df['total_commission'] = df['total_commission'].apply(x)
            df['total_sld'] = df['total_sld'].apply(x)
            df['unrealised_pnl'] = df['unrealised_pnl'].apply(x)
            df['trade_pct'] = (df['avg_sld'] / df['avg_bot'] - 1.0)
            return df

    def _plot_equity(self, stats, ax=None, **kwargs):

        def format_two_dec(x, pos):
            return '%.2f' % x

        equity = stats['cum_returns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_two_dec)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        if self.benchmark is not None:
            benchmark = stats['cum_returns_b']
            benchmark.plot(
                lw=2, color='gray', label=self.benchmark, alpha=0.60,
                ax=ax, **kwargs
            )

        equity.plot(lw=2, color='green', alpha=0.6, x_compat=False,
                    label='Backtest', ax=ax, **kwargs)

        ax.axhline(1.0, linestyle='--', color='black', lw=1)
        ax.set_ylabel('Cumulative returns')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        if self.log_scale:
            ax.set_yscale('log')

        return ax

    def _plot_rolling_sharpe(self, stats, ax=None, **kwargs):

        def format_two_dec(x, pos):
            return '%.2f' % x

        sharpe = stats['rolling_sharpe']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_two_dec)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        if self.benchmark is not None:
            benchmark = stats['rolling_sharpe_b']
            benchmark.plot(
                lw=2, color='gray', label=self.benchmark, alpha=0.60,
                ax=ax, **kwargs
            )

        sharpe.plot(lw=2, color='green', alpha=0.6, x_compat=False,
                    label='Backtest', ax=ax, **kwargs)

        ax.axvline(sharpe.index[252], linestyle="dashed", c="gray", lw=2)
        ax.set_ylabel('Rolling Annualised Sharpe')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        return ax

    def _plot_drawdown(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        drawdown = stats['drawdowns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_tick_params(reset=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        underwater = -100 * drawdown
        underwater.plot(ax=ax, lw=2, kind='area', color='red', alpha=0.3, **kwargs)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
        ax.set_title('Drawdown (%)', fontweight='bold')
        return ax

    def _plot_monthly_returns(self, stats, ax=None, **kwargs):

        returns = stats['returns']
        if ax is None:
            ax = plt.gca()

        monthly_ret = perf.aggregate_returns(returns, 'monthly')
        monthly_ret = monthly_ret.unstack()
        monthly_ret = np.round(monthly_ret, 3)
        monthly_ret.rename(
            columns={1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                     5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                     9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'},
            inplace=True
        )

        sns.heatmap(
            monthly_ret.fillna(0) * 100.0,
            annot=True,
            fmt="0.1f",
            annot_kws={"size": 5},
            alpha=1.0,
            center=0.0,
            cbar=False,
            cmap=cm.RdYlGn,
            ax=ax, **kwargs)
        ax.set_title('Monthly Returns (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('')

        return ax

    def _plot_yearly_returns(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats['returns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.yaxis.grid(linestyle=':')

        yly_ret = perf.aggregate_returns(returns, 'yearly') * 100.0
        yly_ret.plot(ax=ax, kind="bar")
        ax.set_title('Yearly Returns (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.xaxis.grid(False)

        return ax

    def _plot_txt_curve(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats["returns"]
        cum_returns = stats['cum_returns']

        if 'positions' not in stats:
            trd_yr = 0
        else:
            positions = stats['positions']
            trd_yr = positions.shape[0] / (
                    (returns.index[-1] - returns.index[0]).days / 365.0
            )

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        tot_ret = cum_returns[-1] - 1.0
        cagr = perf.create_cagr(cum_returns, self.periods)

        sharpe = create_sharpe_ratio(returns, self.periods,
                                     stats.get('returns_b', None))
        sortino = perf.create_sortino_ratio(returns, self.periods)
        rsq = perf.rsquared(range(cum_returns.shape[0]), cum_returns)
        dd, dd_max, dd_dur = perf.create_drawdowns(cum_returns)

        ax.text(0.25, 8.7, 'Total Return', fontsize=4)
        ax.text(7.50, 8.7, '{:.0%}'.format(tot_ret), fontweight='bold', horizontalalignment='right', fontsize=4)

        ax.text(0.25, 7.7, 'CAGR', fontsize=4)
        ax.text(7.50, 7.7, '{:.2%}'.format(cagr), fontweight='bold', horizontalalignment='right', fontsize=4)

        ax.text(0.25, 6.7, 'Sharpe Ratio', fontsize=4)
        ax.text(7.50, 6.7, '{:.2f}'.format(sharpe), fontweight='bold', horizontalalignment='right', fontsize=4)

        ax.text(0.25, 5.7, 'Sortino Ratio', fontsize=4)
        ax.text(7.50, 5.7, '{:.2f}'.format(sortino), fontweight='bold', horizontalalignment='right', fontsize=4)

        ax.text(0.25, 4.7, 'Annual Volatility', fontsize=4)
        ax.text(7.50, 4.7, '{:.2%}'.format(returns.std() * np.sqrt(252)), fontweight='bold',
                horizontalalignment='right', fontsize=4)

        ax.text(0.25, 3.7, 'Сumulative Return / Max Daily Drawdown', fontsize=4)
        ax.text(7.50, 3.7, '{:.2f}'.format(tot_ret / dd_max), fontweight='bold', horizontalalignment='right',
                fontsize=4)

        ax.text(0.25, 2.7, 'Max Daily Drawdown', fontsize=4)
        ax.text(7.50, 2.7, '{:.2%}'.format(dd_max), color='red', fontweight='bold', horizontalalignment='right',
                fontsize=4)

        ax.text(0.25, 1.7, 'Max Drawdown Duration', fontsize=4)
        ax.text(7.50, 1.7, '{:.0f}'.format(dd_dur), fontweight='bold', horizontalalignment='right', fontsize=4)

        ax.text(0.25, 0.7, 'Trades per Year', fontsize=4)
        ax.text(7.50, 0.7, '{:.1f}'.format(trd_yr), fontweight='bold', horizontalalignment='right', fontsize=4)
        ax.set_title('Curve', fontweight='bold')

        if self.benchmark is not None:
            returns_b = stats['returns_b']
            equity_b = stats['cum_returns_b']
            tot_ret_b = equity_b[-1] - 1.0
            cagr_b = perf.create_cagr(equity_b)
            sharpe_b = perf.create_sharpe_ratio(returns_b)
            sortino_b = perf.create_sortino_ratio(returns_b)
            rsq_b = perf.rsquared(range(equity_b.shape[0]), equity_b)
            dd_b, dd_max_b, dd_dur_b = perf.create_drawdowns(equity_b)

            ax.text(9.75, 8.7, '{:.0%}'.format(tot_ret_b), fontweight='bold', horizontalalignment='right', fontsize=4)
            ax.text(9.75, 7.7, '{:.2%}'.format(cagr_b), fontweight='bold', horizontalalignment='right', fontsize=4)
            ax.text(9.75, 6.7, '{:.2f}'.format(sharpe_b), fontweight='bold', horizontalalignment='right', fontsize=4)
            ax.text(9.75, 5.7, '{:.2f}'.format(sortino_b), fontweight='bold', horizontalalignment='right', fontsize=4)
            ax.text(9.75, 4.7, '{:.2%}'.format(returns_b.std() * np.sqrt(252)), fontweight='bold',
                    horizontalalignment='right', fontsize=4)
            ax.text(9.75, 3.7, '{:.2f}'.format(rsq_b), fontweight='bold', horizontalalignment='right', fontsize=4)
            ax.text(9.75, 2.7, '{:.2%}'.format(dd_max_b), color='red', fontweight='bold', horizontalalignment='right',
                    fontsize=4)
            ax.text(9.75, 1.7, '{:.0f}'.format(dd_dur_b), fontweight='bold', horizontalalignment='right', fontsize=4)

            ax.set_title('Curve vs. Benchmark', fontweight='bold')

        ax.grid(False)
        ax.spines['top'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax

    def _plot_txt_trade(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        if ax is None:
            ax = plt.gca()

        if 'positions' not in stats:
            num_trades = 0
            win_pct = "N/A"
            win_pct_str = "N/A"
            avg_trd_pct = "N/A"
            avg_win_pct = "N/A"
            avg_loss_pct = "N/A"
            max_win_pct = "N/A"
            max_loss_pct = "N/A"
        else:
            pos = stats['positions']
            num_trades = pos.shape[0]
            win_pct = pos[pos["trade_pct"] > 0].shape[0] / float(num_trades)
            win_pct_str = '{:.0%}'.format(win_pct)
            avg_trd_pct = '{:.2%}'.format(np.mean(pos["trade_pct"]))
            avg_win_pct = '{:.2%}'.format(np.mean(pos[pos["trade_pct"] > 0]["trade_pct"]))
            avg_loss_pct = '{:.2%}'.format(np.mean(pos[pos["trade_pct"] <= 0]["trade_pct"]))
            max_win_pct = '{:.2%}'.format(np.max(pos["trade_pct"]))
            max_loss_pct = '{:.2%}'.format(np.min(pos["trade_pct"]))

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        max_loss_dt = 'TBD'  # pos[pos["trade_pct"] == np.min(pos["trade_pct"])].entry_date.values[0]
        avg_dit = '0.0'  # = '{:.2f}'.format(np.mean(pos.time_in_pos))

        ax.text(0.5, 8.7, 'Trade Winning %', fontsize=4)
        ax.text(9.5, 8.7, win_pct_str, fontsize=4, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 7.7, 'Average Trade %', fontsize=4)
        ax.text(9.5, 7.7, avg_trd_pct, fontsize=4, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 6.7, 'Average Win %', fontsize=4)
        ax.text(9.5, 6.7, avg_win_pct, fontsize=4, fontweight='bold', color='green', horizontalalignment='right')

        ax.text(0.5, 5.7, 'Average Loss %', fontsize=4)
        ax.text(9.5, 5.7, avg_loss_pct, fontsize=4, fontweight='bold', color='red', horizontalalignment='right')

        ax.text(0.5, 4.7, 'Best Trade %', fontsize=4)
        ax.text(9.5, 4.7, max_win_pct, fontsize=4, fontweight='bold', color='green', horizontalalignment='right')

        ax.text(0.5, 3.7, 'Worst Trade %', fontsize=4)
        ax.text(9.5, 3.7, max_loss_pct, color='red', fontsize=4, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 2.7, 'Worst Trade Date', fontsize=4)
        ax.text(9.5, 2.7, max_loss_dt, fontsize=4, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 1.7, 'Avg Days in Trade', fontsize=4)
        ax.text(9.5, 1.7, avg_dit, fontsize=4, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 0.7, 'Trades', fontsize=4)
        ax.text(9.5, 0.7, num_trades, fontsize=4, fontweight='bold', horizontalalignment='right')

        ax.set_title('Trade', fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax

    def _plot_txt_time(self, stats, ax=None, **kwargs):

        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats['returns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        mly_ret = perf.aggregate_returns(returns, 'monthly')
        yly_ret = perf.aggregate_returns(returns, 'yearly')

        mly_pct = mly_ret[mly_ret >= 0].shape[0] / float(mly_ret.shape[0])
        mly_avg_win_pct = np.mean(mly_ret[mly_ret >= 0])
        mly_avg_loss_pct = np.mean(mly_ret[mly_ret < 0])
        mly_max_win_pct = np.max(mly_ret)
        mly_max_loss_pct = np.min(mly_ret)
        yly_pct = yly_ret[yly_ret >= 0].shape[0] / float(yly_ret.shape[0])
        yly_max_win_pct = np.max(yly_ret)
        yly_max_loss_pct = np.min(yly_ret)

        ax.text(0.5, 8.7, 'Winning Months %', fontsize=4)
        ax.text(9.5, 8.7, '{:.0%}'.format(mly_pct), fontsize=4, fontweight='bold',
                horizontalalignment='right')

        ax.text(0.5, 7.7, 'Average Winning Month %', fontsize=4)
        ax.text(9.5, 7.7, '{:.2%}'.format(mly_avg_win_pct), fontsize=4, fontweight='bold',
                color='red' if mly_avg_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 6.7, 'Average Losing Month %', fontsize=4)
        ax.text(9.5, 6.7, '{:.2%}'.format(mly_avg_loss_pct), fontsize=4, fontweight='bold',
                color='red' if mly_avg_loss_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 5.7, 'Best Month %', fontsize=4)
        ax.text(9.5, 5.7, '{:.2%}'.format(mly_max_win_pct), fontsize=4, fontweight='bold',
                color='red' if mly_max_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 4.7, 'Worst Month %', fontsize=4)
        ax.text(9.5, 4.7, '{:.2%}'.format(mly_max_loss_pct), fontsize=4, fontweight='bold',
                color='red' if mly_max_loss_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 3.7, 'Winning Years %', fontsize=4)
        ax.text(9.5, 3.7, '{:.0%}'.format(yly_pct), fontsize=4, fontweight='bold',
                horizontalalignment='right')

        ax.text(0.5, 2.7, 'Best Year %', fontsize=4)
        ax.text(9.5, 2.7, '{:.2%}'.format(yly_max_win_pct), fontsize=4,
                fontweight='bold', color='red' if yly_max_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 1.7, 'Worst Year %', fontsize=4)
        ax.text(9.5, 1.7, '{:.2%}'.format(yly_max_loss_pct), fontsize=4,
                fontweight='bold', color='red' if yly_max_loss_pct < 0 else 'green',
                horizontalalignment='right')

        # ax.text(0.5, 0.7, 'Positive 12 Month Periods', fontsize=4)
        # ax.text(9.5, 0.7, num_trades, fontsize=4, fontweight='bold', horizontalalignment='right')

        ax.set_title('Time', fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax

    def plot_results(self, filename=None):

        rc = {
            'lines.linewidth': 1.0,
            'axes.facecolor': '0.795',
            'figure.facecolor': '0.77',
            'font.family': 'serif',
            'font.serif': 'Ubuntu',
            'font.monospace': 'Ubuntu Mono',
            'font.size': 6,
            'axes.labelsize': 6,
            'axes.labelweight': 'bold',
            'axes.titlesize': 6,
            'xtick.labelsize': 4,
            'ytick.labelsize': 4,
            'legend.fontsize': 6,
            'figure.titlesize': 8
        }
        sns.set_context(rc)
        sns.set_style("whitegrid")
        sns.set_palette("deep", desat=.6)

        if self.rolling_sharpe:
            offset_index = 1
        else:
            offset_index = 0
        vertical_sections = 5 + offset_index
        fig = plt.figure(figsize=(10, vertical_sections * 5))
        fig.suptitle(self.title, y=1, weight='bold')
        gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.25, hspace=0.75)

        stats = self.get_results()
        ax_equity = plt.subplot(gs[:2, :])
        if self.rolling_sharpe:
            ax_sharpe = plt.subplot(gs[2, :])
        ax_drawdown = plt.subplot(gs[2 + offset_index, :])
        ax_monthly_returns = plt.subplot(gs[3 + offset_index, :2])
        ax_yearly_returns = plt.subplot(gs[3 + offset_index, 2])
        ax_txt_curve = plt.subplot(gs[4 + offset_index, 0])
        ax_txt_trade = plt.subplot(gs[4 + offset_index, 1])
        ax_txt_time = plt.subplot(gs[4 + offset_index, 2])

        self._plot_equity(stats, ax=ax_equity)
        if self.rolling_sharpe:
            self._plot_rolling_sharpe(stats, ax=ax_sharpe)
        self._plot_drawdown(stats, ax=ax_drawdown)
        self._plot_monthly_returns(stats, ax=ax_monthly_returns)
        self._plot_yearly_returns(stats, ax=ax_yearly_returns)
        self._plot_txt_curve(stats, ax=ax_txt_curve)
        self._plot_txt_trade(stats, ax=ax_txt_trade)
        self._plot_txt_time(stats, ax=ax_txt_time)

        plt.show()

        if filename is not None:
            fig.savefig(filename, dpi=600, bbox_inches='tight')

    def get_filename(self, filename=""):
        if filename == "":
            now = datetime.utcnow()
            filename = "tearsheet_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            filename = os.path.expanduser(os.path.join(self.config.OUTPUT_DIR, filename))
        return filename

    def save(self, filename=""):
        filename = self.get_filename(filename)
        self.plot_results(filename=filename)
