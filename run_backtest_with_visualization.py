#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
バックテスト実行および可視化の一括スクリプト
"""

import os
import sys
import argparse
import subprocess
import datetime
import logging
import pandas as pd

from src.simple_backtest import SimpleBacktest
from src.data.simple_data_fetcher import SimpleDataFetcher
from src.visualization.backtest_visualizer import BacktestVisualizer

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='バックテストを実行して結果を可視化')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='取引する通貨ペア')
    parser.add_argument('--timeframe', type=str, default='1h', help='時間足 (1m, 5m, 15m, 30m, 1h, 4h, 1d)')
    parser.add_argument('--years', type=int, default=1, help='バックテスト期間（年数）')
    parser.add_argument('--initial_capital', type=float, default=1000000, help='初期資本')
    
    args = parser.parse_args()
    
    # ロガーの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 必ず実データを使用する設定
        data_fetcher = SimpleDataFetcher(timeframe=args.timeframe, use_real_data=True)
        
        # 履歴データを取得
        historical_data = data_fetcher.get_data(args.symbol, years=args.years)
        
        if historical_data is None or historical_data.empty:
            logger.error("履歴データが取得できませんでした。バックテストを中止します。")
            return
        
        # バックテストを実行
        backtest = SimpleBacktest(
            data=historical_data,
            symbol=args.symbol,
            timeframe=args.timeframe,
            initial_capital=args.initial_capital
        )
        
        results, trades = backtest.run()
        
        # 結果ディレクトリの確認と作成
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 結果をCSVに保存
        results_file = os.path.join(results_dir, 'advanced_backtest_results.csv')
        trades_file = os.path.join(results_dir, 'advanced_backtest_trades.csv')
        
        results.to_csv(results_file)
        trades.to_csv(trades_file)
        
        logger.info(f"バックテスト結果を保存しました: {results_file}, {trades_file}")
        
        # バックテスト結果の可視化
        visualizer = BacktestVisualizer(results, trades)
        
        # グラフを保存
        plot_file = os.path.join(results_dir, f'backtest_result_{args.symbol.replace("/", "_")}.png')
        visualizer.plot_backtest_results(plot_file)
        
        # HTMLレポートの生成
        html_reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html_reports')
        if not os.path.exists(html_reports_dir):
            os.makedirs(html_reports_dir)
            
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        html_report_file = os.path.join(html_reports_dir, f'backtest_report_{timestamp}.html')
        
        visualizer.generate_html_report(
            html_report_file,
            backtest_params={
                'symbol': args.symbol,
                'timeframe': args.timeframe,
                'years': args.years,
                'initial_capital': args.initial_capital
            }
        )
        
        logger.info(f"HTML可視化レポートを生成しました: {html_report_file}")
        
        # メトリクスの表示
        backtest.print_metrics()
        
    except Exception as e:
        logger.error(f"バックテスト実行中にエラーが発生しました: {e}", exc_info=True)

if __name__ == "__main__":
    main() 