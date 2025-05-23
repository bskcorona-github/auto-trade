#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
バックテスト結果検証ユーティリティ
HTMLレポート生成前にデータの問題を特定・修正
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path

# 親ディレクトリをパスに追加（インポート用）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging(log_level='INFO'):
    """ロギング設定"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'data_validation.log')
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    return logging.getLogger('DataValidator')

def validate_and_fix_dataframe(df, df_name, logger):
    """
    データフレームの検証と修正
    
    Args:
        df (pandas.DataFrame): 検証対象のデータフレーム
        df_name (str): データフレームの名前（ログ用）
        logger (logging.Logger): ロガー
        
    Returns:
        pandas.DataFrame: 修正後のデータフレーム
    """
    if df is None:
        logger.error(f"{df_name} データフレームがNoneです")
        return None
    
    if df.empty:
        logger.error(f"{df_name} データフレームが空です")
        return df
    
    logger.info(f"{df_name} の検証を開始: {len(df)}行, {len(df.columns)}列")
    
    # コピーを作成して元データを変更しない
    fixed_df = df.copy()
    
    # 1. 日付列の処理
    date_columns = [col for col in fixed_df.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_columns:
        if not pd.api.types.is_datetime64_any_dtype(fixed_df[col]):
            try:
                logger.info(f"{df_name} の列 '{col}' を日時型に変換します")
                fixed_df[col] = pd.to_datetime(fixed_df[col])
            except Exception as e:
                logger.warning(f"{df_name} の列 '{col}' の日時変換に失敗: {e}")
    
    # 2. 数値列の処理（NaN, Infinity値）
    numeric_cols = fixed_df.select_dtypes(include=[np.number]).columns
    issue_found = False
    
    for col in numeric_cols:
        # NaNと無限大の値の数を確認
        nan_count = fixed_df[col].isna().sum()
        inf_count = np.isinf(fixed_df[col]).sum()
        extreme_count = ((fixed_df[col] > 1e10) | (fixed_df[col] < -1e10)).sum()
        
        if nan_count > 0:
            logger.warning(f"{df_name} の列 '{col}' に {nan_count}個の欠損値があります")
            fixed_df[col] = fixed_df[col].fillna(0)
            issue_found = True
            
        if inf_count > 0:
            logger.warning(f"{df_name} の列 '{col}' に {inf_count}個の無限大値があります")
            fixed_df[col] = fixed_df[col].replace([np.inf, -np.inf], [1e10, -1e10])
            issue_found = True
            
        if extreme_count > 0 and inf_count == 0:
            logger.warning(f"{df_name} の列 '{col}' に {extreme_count}個の極端な値があります")
            # 極端な値を持つ行を特定
            extreme_rows = fixed_df[((fixed_df[col] > 1e10) | (fixed_df[col] < -1e10))].index
            logger.debug(f"極端な値を持つ行のインデックス: {extreme_rows.tolist()}")
            issue_found = True
    
    if issue_found:
        logger.info(f"{df_name} のデータに問題が見つかり、修正されました")
    else:
        logger.info(f"{df_name} のデータに問題は見つかりませんでした")
    
    return fixed_df

def validate_and_fix_results(results_file, trades_file, output_dir=None, replace=False):
    """
    バックテスト結果と取引データの検証と修正
    
    Args:
        results_file (str): 結果CSVファイルのパス
        trades_file (str): 取引CSVファイルのパス
        output_dir (str, optional): 出力ディレクトリ
        replace (bool): 元ファイルを上書きするかどうか
        
    Returns:
        tuple: (修正済み結果ファイルパス, 修正済み取引ファイルパス)
    """
    logger = setup_logging()
    
    logger.info(f"バックテスト結果検証開始")
    logger.info(f"結果ファイル: {results_file}")
    logger.info(f"取引ファイル: {trades_file}")
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイルが存在するか確認
    if not os.path.exists(results_file):
        logger.error(f"結果ファイルが見つかりません: {results_file}")
        return None, None
    
    if not os.path.exists(trades_file):
        logger.error(f"取引ファイルが見つかりません: {trades_file}")
        return None, None
    
    try:
        # CSVファイル読み込み
        results_df = pd.read_csv(results_file)
        trades_df = pd.read_csv(trades_file)
        
        # データフレームの検証と修正
        fixed_results_df = validate_and_fix_dataframe(results_df, "結果データ", logger)
        fixed_trades_df = validate_and_fix_dataframe(trades_df, "取引データ", logger)
        
        # 修正されたデータを保存
        results_basename = Path(results_file).stem
        trades_basename = Path(trades_file).stem
        
        if replace:
            # 元ファイルを上書き
            output_results_file = results_file
            output_trades_file = trades_file
            logger.info("元ファイルを修正データで上書きします")
        else:
            # 新しいファイル名で保存
            output_results_file = os.path.join(output_dir, f"{results_basename}_fixed.csv")
            output_trades_file = os.path.join(output_dir, f"{trades_basename}_fixed.csv")
        
        # データ保存
        fixed_results_df.to_csv(output_results_file, index=False)
        fixed_trades_df.to_csv(output_trades_file, index=False)
        
        logger.info(f"修正済み結果ファイル: {output_results_file}")
        logger.info(f"修正済み取引ファイル: {output_trades_file}")
        
        return output_results_file, output_trades_file
        
    except Exception as e:
        logger.error(f"データ検証中にエラー発生: {e}", exc_info=True)
        return None, None

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='バックテスト結果データの検証と修正')
    parser.add_argument('--results', type=str, required=True,
                        help='バックテスト結果のCSVファイルパス')
    parser.add_argument('--trades', type=str, required=True,
                        help='取引履歴のCSVファイルパス')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='出力ディレクトリ (デフォルト: 結果ファイルと同じディレクトリ)')
    parser.add_argument('--replace', action='store_true',
                        help='元ファイルを上書きするかどうか')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='ログレベル設定')
    
    args = parser.parse_args()
    
    # ロガー設定
    logger = setup_logging(args.log_level)
    
    # データ検証と修正
    fixed_results, fixed_trades = validate_and_fix_results(
        args.results, 
        args.trades, 
        args.output_dir, 
        args.replace
    )
    
    if fixed_results and fixed_trades:
        logger.info(f"データ検証と修正が完了しました。")
        logger.info(f"修正済み結果ファイル: {fixed_results}")
        logger.info(f"修正済み取引ファイル: {fixed_trades}")
        return 0
    else:
        logger.error(f"データ検証と修正に失敗しました。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 