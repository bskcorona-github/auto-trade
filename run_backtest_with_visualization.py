#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
バックテスト実行および可視化の一括スクリプト
市場環境認識と動的シグナル重み付けによる改良版
"""

import os
import sys
import argparse
import subprocess
import datetime
import logging
import pandas as pd
import yaml

from src.simple_backtest import SimpleBacktest
from src.data.simple_data_fetcher import SimpleDataFetcher
from src.visualization.backtest_visualizer import BacktestVisualizer

def load_config():
    """設定ファイル読み込み"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"警告: 設定ファイルが見つかりません: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"警告: 設定ファイルの読み込みエラー: {e}")
        return {}
        
def setup_logging(log_level='INFO'):
    """詳細なロギング設定"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'backtest_{timestamp}.log')
    
    # ログレベル設定
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # ロガー設定
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    return logging.getLogger('Backtest')

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='バックテストを実行して結果を可視化')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='取引する通貨ペア')
    parser.add_argument('--timeframe', type=str, default='1h', help='時間足 (1m, 5m, 15m, 30m, 1h, 4h, 1d)')
    parser.add_argument('--years', type=int, default=3, help='バックテスト期間（年数）')
    parser.add_argument('--initial_capital', type=float, default=1000000, help='初期資本')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='ログレベル設定')
    parser.add_argument('--optimize_weights', action='store_true', 
                        help='シグナル重み付け最適化を有効にする')
    parser.add_argument('--market_regime', action='store_true', 
                        help='市場環境認識機能を有効にする')
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = load_config()
    
    # ロギング設定
    logger = setup_logging(args.log_level)
    
    logger.info("=====================================================")
    logger.info(f"バックテスト開始: {args.symbol} {args.timeframe} {args.years}年分")
    logger.info(f"市場環境認識: {'有効' if args.market_regime else '無効'}")
    logger.info(f"シグナル最適化: {'有効' if args.optimize_weights else '無効'}")
    logger.info("=====================================================")
    
    try:
        # 必ず実データを使用する設定
        data_fetcher = SimpleDataFetcher(timeframe=args.timeframe, use_real_data=True)
        
        # 履歴データを取得
        logger.info(f"{args.symbol}の{args.years}年分の履歴データ取得中...")
        historical_data = data_fetcher.get_data(args.symbol, years=args.years)
        
        if historical_data is None or historical_data.empty:
            logger.error("履歴データが取得できませんでした。バックテストを中止します。")
            return
            
        logger.info(f"履歴データ取得完了: {len(historical_data)}件のデータポイント")
        
        # バックテストを実行
        logger.info("バックテスト実行中...")
        
        # 戦略パラメータの指定
        strategy_params = config.get('strategy_params', {})
        risk_params = config.get('risk_params', {})
        
        # 改良された機能（市場環境認識、シグナル最適化）のオプション設定
        advanced_options = {
            'use_market_regime': args.market_regime,
            'optimize_signal_weights': args.optimize_weights
        }
        
        backtest = SimpleBacktest(
            data=historical_data,
            symbol=args.symbol,
            timeframe=args.timeframe,
            initial_capital=args.initial_capital,
            strategy_params=strategy_params,
            risk_params=risk_params,
            advanced_options=advanced_options
        )
        
        results, trades = backtest.run()
        
        # 結果ディレクトリの確認と作成
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 結果をCSVに保存（タイムスタンプ付きで保存）
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'backtest_results_{timestamp}.csv')
        trades_file = os.path.join(results_dir, f'backtest_trades_{timestamp}.csv')
        
        results.to_csv(results_file)
        trades.to_csv(trades_file)
        
        logger.info(f"バックテスト結果を保存しました: {results_file}, {trades_file}")
        
        # シグナル最適化情報のログ出力（有効時）
        if args.optimize_weights:
            logger.info("シグナル重み付け最終状態:")
            for signal_type, weight in backtest.strategy.signal_weights.items():
                logger.info(f"  - {signal_type}: {weight:.4f}")
                
            logger.info("シグナル成功率:")
            for signal_type, history in backtest.strategy.signal_history.items():
                if history['total'] > 0:
                    success_rate = history['correct'] / history['total'] * 100
                    logger.info(f"  - {signal_type}: {success_rate:.2f}% ({history['correct']}/{history['total']})")
        
        # バックテスト結果の可視化
        logger.info("バックテスト結果の可視化中...")
        visualizer = BacktestVisualizer(results, trades)
        
        # グラフを保存
        plot_file = os.path.join(results_dir, f'backtest_result_{args.symbol.replace("/", "_")}_{timestamp}.png')
        visualizer.plot_backtest_results(plot_file)
        
        # HTMLレポートの生成
        html_reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html_reports')
        if not os.path.exists(html_reports_dir):
            os.makedirs(html_reports_dir)
            
        html_report_file = os.path.join(html_reports_dir, f'backtest_report_{timestamp}.html')
        
        visualizer.generate_html_report(
            html_report_file,
            backtest_params={
                'symbol': args.symbol,
                'timeframe': args.timeframe,
                'years': args.years,
                'initial_capital': args.initial_capital,
                'market_regime': args.market_regime,
                'optimize_weights': args.optimize_weights
            }
        )
        
        logger.info(f"HTML可視化レポートを生成しました: {html_report_file}")
        
        # メトリクスの表示
        backtest.print_metrics()
        
        logger.info("=====================================================")
        logger.info("バックテスト完了")
        logger.info("=====================================================")
        
    except Exception as e:
        logger.error(f"バックテスト実行中にエラーが発生しました: {e}", exc_info=True)

if __name__ == "__main__":
    main() 