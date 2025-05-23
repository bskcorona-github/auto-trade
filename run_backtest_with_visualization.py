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
import numpy as np

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
    parser.add_argument('--debug_mode', action='store_true',
                        help='デバッグモードを有効にする（詳細なトレース情報を出力）')
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = load_config()
    
    # デバッグモードの場合はログレベルをDEBUGに設定
    if args.debug_mode:
        log_level = 'DEBUG'
    else:
        log_level = args.log_level
    
    # ロギング設定
    logger = setup_logging(log_level)
    
    logger.info("=====================================================")
    logger.info(f"バックテスト開始: {args.symbol} {args.timeframe} {args.years}年分")
    logger.info(f"市場環境認識: {'有効' if args.market_regime else '無効'}")
    logger.info(f"シグナル最適化: {'有効' if args.optimize_weights else '無効'}")
    logger.info(f"デバッグモード: {'有効' if args.debug_mode else '無効'}")
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
            
        # データの検証
        logger.info(f"履歴データ取得完了: {len(historical_data)}件のデータポイント")
        
        if args.debug_mode:
            # データ内の欠損値や無限大値をチェック
            numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                nan_count = historical_data[col].isna().sum()
                inf_count = np.isinf(historical_data[col]).sum()
                if nan_count > 0 or inf_count > 0:
                    logger.warning(f"列 '{col}' に {nan_count} 個の欠損値と {inf_count} 個の無限大値が含まれています")
        
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
        
        # バックテスト結果の検証
        if args.debug_mode:
            if results is not None and not results.empty:
                logger.debug(f"結果データ: {len(results)}行, 列: {results.columns.tolist()}")
                # 無限大値や欠損値をチェック
                numeric_cols = results.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    nan_count = results[col].isna().sum()
                    inf_count = np.isinf(results[col]).sum()
                    if nan_count > 0 or inf_count > 0:
                        logger.warning(f"結果の列 '{col}' に {nan_count} 個の欠損値と {inf_count} 個の無限大値が含まれています")
            
            if trades is not None and not trades.empty:
                logger.debug(f"取引データ: {len(trades)}行, 列: {trades.columns.tolist()}")
                # 無限大値や欠損値をチェック
                numeric_cols = trades.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    nan_count = trades[col].isna().sum()
                    inf_count = np.isinf(trades[col]).sum()
                    if nan_count > 0 or inf_count > 0:
                        logger.warning(f"取引の列 '{col}' に {nan_count} 個の欠損値と {inf_count} 個の無限大値が含まれています")
        
        # 結果ディレクトリの確認と作成
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 結果をCSVに保存（タイムスタンプ付きで保存）
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f'backtest_results_{timestamp}.csv')
        trades_file = os.path.join(results_dir, f'backtest_trades_{timestamp}.csv')
        
        # 保存前にデータをクリーニング
        if 'date' in results.columns and not pd.api.types.is_datetime64_any_dtype(results['date']):
            results['date'] = pd.to_datetime(results['date'])
        
        # NumPyの制限を超える値を扱えるように調整
        # 無限大を大きな数値に置換
        results = results.replace([np.inf, -np.inf], [1e10, -1e10])
        trades = trades.replace([np.inf, -np.inf], [1e10, -1e10])
        
        # 欠損値を0に置換
        results = results.fillna(0)
        trades = trades.fillna(0)
        
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
        
        try:
            visualizer = BacktestVisualizer(results, trades)
            
            # グラフを保存
            plot_file = os.path.join(results_dir, f'backtest_result_{args.symbol.replace("/", "_")}_{timestamp}.png')
            visualizer.plot_backtest_results(plot_file)
            
            # HTMLレポートの生成
            html_reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html_reports')
            if not os.path.exists(html_reports_dir):
                os.makedirs(html_reports_dir)
                
            html_report_file = f'backtest_report_{timestamp}.html'
            
            # HTMLレポート生成でエラーが発生した場合に詳細ログを出力
            try:
                output_path = visualizer.generate_html_report(
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
                logger.info(f"HTML可視化レポートを生成しました: {output_path}")
            except Exception as e:
                logger.error(f"HTMLレポート生成中にエラー発生: {e}")
                if args.debug_mode:
                    import traceback
                    logger.error(traceback.format_exc())
                    
                # 簡易HTMLレポートを代替として作成
                try:
                    simple_html_path = os.path.join(html_reports_dir, f'simple_report_{timestamp}.html')
                    with open(simple_html_path, 'w', encoding='utf-8') as f:
                        f.write(f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>簡易バックテストレポート</title>
                            <style>body {{font-family: Arial; padding: 20px;}}</style>
                        </head>
                        <body>
                            <h1>バックテスト結果 - {args.symbol}</h1>
                            <p>HTMLレポート生成でエラーが発生したため、簡易レポートを代わりに表示しています。</p>
                            <p>結果CSVファイル: <a href="../{results_file}">{os.path.basename(results_file)}</a></p>
                            <p>取引CSVファイル: <a href="../{trades_file}">{os.path.basename(trades_file)}</a></p>
                            <img src="../{plot_file}" style="max-width:100%">
                        </body>
                        </html>
                        """)
                    logger.info(f"代替の簡易HTMLレポートを生成しました: {simple_html_path}")
                except Exception as e2:
                    logger.error(f"簡易レポート生成中にもエラー発生: {e2}")
            
        except Exception as e:
            logger.error(f"可視化処理中にエラー発生: {e}")
            if args.debug_mode:
                import traceback
                logger.error(traceback.format_exc())
        
        # メトリクスの表示
        try:
            backtest.print_metrics()
        except Exception as e:
            logger.error(f"メトリクス表示でエラー発生: {e}")
        
        logger.info("=====================================================")
        logger.info("バックテスト完了")
        logger.info("=====================================================")
        
    except Exception as e:
        logger.error(f"バックテスト実行中にエラーが発生しました: {e}")
        if args.debug_mode:
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 