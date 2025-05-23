#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
バックテスト結果のHTMLレポート生成スクリプト
既存のCSVデータからHTMLレポートのみを生成
"""

import os
import sys
import argparse
import logging
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# モジュールのインポート
from src.visualization.backtest_visualizer import BacktestVisualizer
from src.visualization.validate_results import validate_and_fix_dataframe

def setup_logging(log_level='INFO'):
    """ロギング設定"""
    # プロジェクトのルートディレクトリを取得
    project_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 現在時刻を含むログファイル名
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'report_gen_{timestamp}.log')
    
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
    
    return logging.getLogger('ReportGenerator')

def generate_report(results_file, trades_file, output_file=None, backtest_params=None, validate=True, debug=False):
    """
    バックテスト結果のHTMLレポートを生成
    
    Args:
        results_file (str): 結果CSVファイルのパス
        trades_file (str): 取引CSVファイルのパス
        output_file (str, optional): 出力HTMLファイル名
        backtest_params (dict, optional): バックテストパラメータ
        validate (bool): データ検証を行うかどうか
        debug (bool): デバッグモードを有効にするかどうか
        
    Returns:
        str: 生成されたHTMLファイルのパス
    """
    logger = logging.getLogger('ReportGenerator')
    
    # 結果と取引データの読み込み
    try:
        logger.info(f"結果ファイル読み込み: {results_file}")
        results_df = pd.read_csv(results_file)
        
        logger.info(f"取引ファイル読み込み: {trades_file}")
        trades_df = pd.read_csv(trades_file)
        
        # データの検証と修正
        if validate:
            logger.info("データ検証と修正開始")
            
            # データ検証用のロガー
            validator_logger = logging.getLogger('DataValidator')
            
            # データフレームの検証と修正
            results_df = validate_and_fix_dataframe(results_df, "結果データ", validator_logger)
            trades_df = validate_and_fix_dataframe(trades_df, "取引データ", validator_logger)
            
            logger.info("データ検証と修正完了")
        
        # 可視化クラスの初期化
        visualizer = BacktestVisualizer(results_df, trades_df)
        
        # バックテストパラメータがない場合は基本情報から生成
        if backtest_params is None:
            symbol = Path(results_file).stem.split('_')[0] if '_' in Path(results_file).stem else "Unknown"
            backtest_params = {
                'symbol': symbol,
                'timeframe': 'Unknown',
                'years': 'Unknown',
                'initial_capital': results_df['equity'].iloc[0] if 'equity' in results_df.columns else 1000000
            }
        
        # レポート生成
        try:
            logger.info("HTMLレポート生成開始")
            output_path = visualizer.generate_html_report(output_file, backtest_params)
            logger.info(f"HTMLレポート生成完了: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"HTMLレポート生成中にエラー: {e}")
            if debug:
                import traceback
                logger.error(traceback.format_exc())
                
            # 最低限のHTMLレポートを作成
            try:
                html_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html_reports')
                os.makedirs(html_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                simple_html_path = os.path.join(html_dir, f'simple_report_{timestamp}.html')
                
                # シンプルなHTMLを作成
                with open(simple_html_path, 'w', encoding='utf-8') as f:
                    f.write(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>簡易バックテストレポート</title>
                        <style>
                        body {{ font-family: Arial; padding: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        </style>
                    </head>
                    <body>
                        <h1>バックテスト結果レポート</h1>
                        <p>通常のレポート生成でエラーが発生したため、簡易レポートを表示しています。</p>
                        <p>レポート生成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        
                        <h2>バックテスト設定</h2>
                        <table>
                            <tr><th>パラメータ</th><th>値</th></tr>
                    """)
                    
                    # パラメータ表示
                    for key, value in backtest_params.items():
                        f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
                    
                    f.write("</table>")
                    
                    # 基本メトリクス計算と表示
                    if 'equity' in results_df.columns:
                        initial_equity = results_df['equity'].iloc[0]
                        final_equity = results_df['equity'].iloc[-1]
                        total_return = ((final_equity / initial_equity) - 1) * 100
                        
                        f.write(f"""
                        <h2>基本メトリクス</h2>
                        <table>
                            <tr><th>メトリクス</th><th>値</th></tr>
                            <tr><td>初期資本</td><td>{initial_equity:,.2f}</td></tr>
                            <tr><td>最終資本</td><td>{final_equity:,.2f}</td></tr>
                            <tr><td>トータルリターン</td><td>{total_return:.2f}%</td></tr>
                        """)
                        
                        if 'drawdown' in results_df.columns:
                            max_drawdown = results_df['drawdown'].max()
                            f.write(f"<tr><td>最大ドローダウン</td><td>{max_drawdown:.2f}%</td></tr>")
                            
                        if not trades_df.empty and 'pnl' in trades_df.columns:
                            win_trades = len(trades_df[trades_df['pnl'] > 0])
                            total_trades = len(trades_df)
                            win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
                            f.write(f"<tr><td>取引回数</td><td>{total_trades}</td></tr>")
                            f.write(f"<tr><td>勝率</td><td>{win_rate:.2f}%</td></tr>")
                            
                        f.write("</table>")
                    
                    f.write("""
                    <p>詳細な分析には、データを修正してから完全なレポートを再生成してください。</p>
                    </body>
                    </html>
                    """)
                
                logger.info(f"代替の簡易HTMLレポートを生成しました: {simple_html_path}")
                return simple_html_path
                
            except Exception as e2:
                logger.error(f"簡易HTMLレポート生成中にもエラー: {e2}")
                return None
            
    except Exception as e:
        logger.error(f"データ読み込み中にエラー: {e}")
        if debug:
            import traceback
            logger.error(traceback.format_exc())
        return None

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='バックテスト結果のHTMLレポート生成')
    parser.add_argument('--results', type=str, required=True,
                        help='バックテスト結果のCSVファイルパス')
    parser.add_argument('--trades', type=str, required=True,
                        help='取引履歴のCSVファイルパス')
    parser.add_argument('--output', type=str, default=None,
                        help='出力HTMLファイル名 (デフォルト: 自動生成)')
    parser.add_argument('--symbol', type=str, default=None,
                        help='取引通貨ペア名（レポート表示用）')
    parser.add_argument('--timeframe', type=str, default=None,
                        help='時間足（レポート表示用）')
    parser.add_argument('--years', type=int, default=None,
                        help='バックテスト期間（レポート表示用）')
    parser.add_argument('--initial_capital', type=float, default=None,
                        help='初期資本金（レポート表示用）')
    parser.add_argument('--no_validate', action='store_true',
                        help='データ検証をスキップする')
    parser.add_argument('--debug', action='store_true',
                        help='デバッグモードを有効にする')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='ログレベル設定')
    
    args = parser.parse_args()
    
    # デバッグモードの場合はログレベルをDEBUGに設定
    if args.debug:
        log_level = 'DEBUG'
    else:
        log_level = args.log_level
    
    # ロガー設定
    logger = setup_logging(log_level)
    
    # バックテストパラメータの設定
    backtest_params = {}
    if args.symbol:
        backtest_params['symbol'] = args.symbol
    if args.timeframe:
        backtest_params['timeframe'] = args.timeframe
    if args.years:
        backtest_params['years'] = args.years
    if args.initial_capital:
        backtest_params['initial_capital'] = args.initial_capital
    
    # レポート生成
    output_path = generate_report(
        args.results,
        args.trades,
        args.output,
        backtest_params if backtest_params else None,
        not args.no_validate,
        args.debug
    )
    
    if output_path:
        logger.info(f"レポート生成成功: {output_path}")
        return 0
    else:
        logger.error("レポート生成失敗")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 