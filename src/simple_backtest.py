#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高信頼性バックテストプログラム
複数の市場環境で信頼性の高い戦略評価を行う
"""
import argparse
import logging
import pandas as pd
import numpy as np
import sys
import os
import codecs
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from datetime import datetime
import yaml
from pathlib import Path

# コンソール出力をUTF-8に設定
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# 設定ファイルの読み込み関数 (simple_data_fetcher.pyにも同様の関数があるが、共有も検討可)
# ここでは簡単のため、各ファイルで読み込む
def load_config():
    # スクリプト(simple_backtest.py)が src にあるため、プロジェクトルートは2階層上
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'config.yml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"警告(Backtest): 設定ファイルが見つかりません: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"警告(Backtest): 設定ファイルの読み込みエラー: {e}")
        return {}

config = load_config()

# ロギング設定 (configから読み込むように変更)
log_file_from_config = config.get('logging', {}).get('log_file', 'advanced_backtest.log')
log_level_str = config.get('logging', {}).get('log_level', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)

# ディレクトリ設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(project_root, config.get('logs_dir', 'logs'))
results_dir = os.path.join(project_root, config.get('results_dir', 'results'))

# 必要なディレクトリを作成
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

log_file_path = os.path.join(logs_dir, log_file_from_config)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, encoding='utf-8')
    ]
)
logger = logging.getLogger('AdvancedBacktest')

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.simple_data_fetcher import SimpleDataFetcher

# 日本語フォント設定
def setup_japanese_fonts():
    # 日本語フォントのリスト（優先順）
    japanese_fonts = ['IPAexGothic', 'MS Gothic', 'Hiragino Sans GB', 'Yu Gothic', 'Meiryo', 'TakaoGothic', 'Noto Sans CJK JP']
    
    # 利用可能なフォントを探す
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    # 日本語フォントが見つかったかどうか
    font_found = False
    
    # 利用可能な日本語フォントを探す
    for font in japanese_fonts:
        if font in available_fonts:
            print(f"日本語フォント '{font}' を使用します")
            matplotlib.rcParams['font.family'] = font
            font_found = True
            break
    
    if not font_found:
        # 日本語フォントが見つからない場合は警告を出して、デフォルトフォントを使用
        print("警告: 日本語フォントが見つかりません。テキストが正しく表示されない場合があります。")
        print("以下のコマンドで日本語フォントをインストールすることをお勧めします:")
        print("Windowsの場合: IPAフォントをダウンロードしてインストール")
        print("Linuxの場合: sudo apt-get install fonts-ipafont")
        
        # フォールバックとしてデフォルトのサンセリフフォントを使用
        matplotlib.rcParams['font.family'] = 'sans-serif'
    
    # その他のmatplotlib設定
    matplotlib.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示
    matplotlib.rcParams['figure.figsize'] = [12, 8]  # デフォルトの図のサイズ
    matplotlib.rcParams['figure.dpi'] = 100  # 解像度

class AdvancedStrategy:
    """高度な取引戦略"""
    
    def __init__(self, symbol, strategy_params=None, risk_params=None, commission_rate=0.001, slippage_pct=0.001, use_stop_loss=True):
        """
        初期化 (configからパラメータを受け取るように変更)
        
        Args:
            symbol (str): 取引通貨ペア
            strategy_params (dict): 戦略関連のパラメータ
            risk_params (dict): リスク管理関連のパラメータ
            commission_rate (float): 取引手数料率（片道）
            slippage_pct (float): スリッページ率
            use_stop_loss (bool): ストップロスを使用するかどうか
        """
        self.symbol = symbol
        self.commission_rate = commission_rate
        self.slippage_pct = slippage_pct
        self.use_stop_loss = use_stop_loss
        self.logger = logging.getLogger('AdvancedStrategy')
        
        # パラメータのデフォルト値設定
        self.strategy_params = strategy_params if strategy_params is not None else {}
        self.risk_params = risk_params if risk_params is not None else {}
        
        # リスク管理パラメータをconfigから取得 (ATRベースと固定フォールバック)
        self.risk_per_trade = self.risk_params.get('risk_per_trade', 0.02)
        self.atr_period_for_risk = self.risk_params.get('atr_period_for_risk', 14)
        self.atr_multiplier_stop_loss = self.risk_params.get('atr_multiplier_stop_loss', 2.0)
        self.atr_multiplier_take_profit = self.risk_params.get('atr_multiplier_take_profit', 4.0)
        self.atr_multiplier_trailing_stop = self.risk_params.get('atr_multiplier_trailing_stop', 1.5)
        self.fallback_stop_loss_pct = self.risk_params.get('fallback_stop_loss_percentage', 0.03)
        self.fallback_take_profit_pct = self.risk_params.get('fallback_take_profit_percentage', 0.06)
        self.fallback_trailing_stop_pct = self.risk_params.get('fallback_trailing_stop_percentage', 0.02)
        
        self.logger.info(f"高度な取引戦略を初期化 - {symbol} - パラメータ: {strategy_params}, リスクパラメータ: {risk_params}")
    
    def apply_slippage(self, price, direction):
        """
        スリッページを適用
        
        Args:
            price (float): 理論上の取引価格
            direction (int): 取引方向（1: 買い, -1: 売り）
            
        Returns:
            float: スリッページ適用後の価格
        """
        if direction == 1:  # 買い注文
            return price * (1 + self.slippage_pct)
        else:  # 売り注文
            return price * (1 - self.slippage_pct)
    
    def calculate_commission(self, position_value):
        """
        手数料計算
        
        Args:
            position_value (float): ポジション価値
            
        Returns:
            float: 手数料
        """
        return position_value * self.commission_rate
    
    def generate_signal(self, df):
        """
        シグナル生成 - 複数の戦略を組み合わせて信頼性を向上
        (戦略パラメータをconfigから利用するように変更)
        
        Args:
            df (pandas.DataFrame): テクニカル指標付きのデータフレーム
            
        Returns:
            int: 1(買い), -1(売り), 0(何もしない)
        """
        if len(df) < 200: # 最小データポイントチェックは維持
            return 0
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        signals = []
        params = self.strategy_params # configから読み込んだ戦略パラメータ

        # 1. クロスオーバー戦略
        if df.iloc[-2][f'sma_{params.get("sma_short_period", 50)}'] > df.iloc[-2][f'sma_{params.get("sma_long_period", 200)}'] and \
           df.iloc[-3][f'sma_{params.get("sma_short_period", 50)}'] <= df.iloc[-3][f'sma_{params.get("sma_long_period", 200)}']:
            signals.append(1)
        elif df.iloc[-2][f'sma_{params.get("sma_short_period", 50)}'] < df.iloc[-2][f'sma_{params.get("sma_long_period", 200)}'] and \
             df.iloc[-3][f'sma_{params.get("sma_short_period", 50)}'] >= df.iloc[-3][f'sma_{params.get("sma_long_period", 200)}']:
            signals.append(-1)
            
        if prev_row[f'ema_{params.get("ema_short_period", 20)}'] <= prev_row[f'ema_{params.get("ema_long_period", 50)}'] and \
           last_row[f'ema_{params.get("ema_short_period", 20)}'] > last_row[f'ema_{params.get("ema_long_period", 50)}']:
            signals.append(1)
        elif prev_row[f'ema_{params.get("ema_short_period", 20)}'] >= prev_row[f'ema_{params.get("ema_long_period", 50)}'] and \
             last_row[f'ema_{params.get("ema_short_period", 20)}'] < last_row[f'ema_{params.get("ema_long_period", 50)}']:
            signals.append(-1)
            
        # 2. RSI戦略
        rsi_period = params.get("rsi_period_trade", 14)
        rsi_oversold = params.get("rsi_oversold_threshold", 30)
        rsi_overbought = params.get("rsi_overbought_threshold", 70)
        if last_row[f'rsi_{rsi_period}'] < rsi_oversold and prev_row[f'rsi_{rsi_period}'] < rsi_oversold and last_row[f'rsi_{rsi_period}'] > prev_row[f'rsi_{rsi_period}']:
            signals.append(1)
        elif last_row[f'rsi_{rsi_period}'] > rsi_overbought and prev_row[f'rsi_{rsi_period}'] > rsi_overbought and last_row[f'rsi_{rsi_period}'] < prev_row[f'rsi_{rsi_period}']:
            signals.append(-1)
            
        # 3. MACD戦略
        if prev_row['macd'] <= prev_row['macd_signal'] and last_row['macd'] > last_row['macd_signal']:
            signals.append(1)
        elif prev_row['macd'] >= prev_row['macd_signal'] and last_row['macd'] < last_row['macd_signal']:
            signals.append(-1)
            
        # 4. ボリンジャーバンド戦略
        if last_row['close'] < last_row['bb_lower'] and prev_row['close'] <= prev_row['bb_lower']:
            signals.append(1)
        elif last_row['close'] > last_row['bb_upper'] and prev_row['close'] >= prev_row['bb_upper']:
            signals.append(-1)
            
        # 5. ADX戦略（トレンド強度）
        adx_trend_threshold = params.get("adx_trend_threshold", 25)
        if last_row['adx'] > adx_trend_threshold:
            if last_row['di_plus'] > last_row['di_minus']:
                signals.append(1)
            elif last_row['di_plus'] < last_row['di_minus']:
                signals.append(-1)
                
        # 6. ストキャスティクスクロス
        stoch_oversold = params.get("stoch_oversold", 30)
        stoch_overbought = params.get("stoch_overbought", 70)
        if prev_row['stoch_k'] <= prev_row['stoch_d'] and last_row['stoch_k'] > last_row['stoch_d']:
            if last_row['stoch_k'] < stoch_oversold:
                signals.append(1)
        elif prev_row['stoch_k'] >= prev_row['stoch_d'] and last_row['stoch_k'] < last_row['stoch_d']:
            if last_row['stoch_k'] > stoch_overbought:
                signals.append(-1)
        
        # 7. ボリュームサージ
        volume_surge_ratio = params.get("volume_surge_ratio", 2.0)
        if last_row['volume_ratio'] > volume_surge_ratio:
            if last_row['close'] > last_row['open']:
                signals.append(1)
            elif last_row['close'] < last_row['open']:
                signals.append(-1)
                
        # 8. トレンド方向 + ボラティリティ増加
        volatility_threshold = params.get("volatility_threshold", 2.0)
        #  sma_medium_period_fallback を使用 (config.ymlでは sma_medium_periodがないため)
        sma_medium_period = config.get('strategy_params', {}).get('sma_medium_period_fallback', 50) 
        if last_row['close'] > last_row[f'sma_{sma_medium_period}'] and last_row['volatility_20'] > volatility_threshold:
            signals.append(1)
        elif last_row['close'] < last_row[f'sma_{sma_medium_period}'] and last_row['volatility_20'] > volatility_threshold:
            signals.append(-1)
            
        # 9. ATRブレイクアウト
        atr_breakout_period = params.get("atr_breakout_period", 5)
        atr_breakout_multiplier = params.get("atr_breakout_multiplier", 1.0)
        recent_high = df['high'].iloc[-atr_breakout_period:].max()
        recent_low = df['low'].iloc[-atr_breakout_period:].min()
        # risk_params から atr_period_for_risk を取得
        atr_period_for_current_signal = self.risk_params.get('atr_period_for_risk', 14)
        if last_row['close'] > recent_high + (last_row[f'atr_{atr_period_for_current_signal}'] * atr_breakout_multiplier):
            signals.append(1)
        elif last_row['close'] < recent_low - (last_row[f'atr_{atr_period_for_current_signal}'] * atr_breakout_multiplier):
            signals.append(-1)
            
        # 10. チャネルブレイクアウト
        channel_breakout_period = params.get("channel_breakout_period", 20)
        upper_channel = df['high'].rolling(channel_breakout_period).max()
        lower_channel = df['low'].rolling(channel_breakout_period).min()
        if last_row['close'] > upper_channel.iloc[-2] and prev_row['close'] <= upper_channel.iloc[-3]:
            signals.append(1)
        elif last_row['close'] < lower_channel.iloc[-2] and prev_row['close'] >= lower_channel.iloc[-3]:
            signals.append(-1)
        
        # シグナルの集計
        if not signals:
            return 0
            
        # 過半数以上のシグナルが同じ方向であれば、その方向に取引
        buy_signals = len([s for s in signals if s == 1])
        sell_signals = len([s for s in signals if s == -1])
        
        if buy_signals > sell_signals and buy_signals >= 2:
            return 1
        elif sell_signals > buy_signals and sell_signals >= 2:
            return -1
        
        return 0
    
    def calculate_position_size(self, capital, price, current_atr, risk_pct=None):
        """
        ポジションサイズ計算（リスク管理） - ATRベース
        
        Args:
            capital (float): 利用可能な資金
            price (float): 現在価格
            current_atr (float): 現在のATR値
            risk_pct (float, optional): リスク率
            
        Returns:
            float: ポジションサイズ（単位数）
        """
        if risk_pct is None:
            risk_pct = self.risk_per_trade
            
        risk_amount = capital * risk_pct
        
        if current_atr is not None and current_atr > 0:
            # ATRベースのストップロス幅
            stop_loss_amount_per_unit = current_atr * self.atr_multiplier_stop_loss
        else:
            # ATRが利用できない場合は固定パーセンテージを使用
            self.logger.warning("ATRが利用できないため、固定ストップロス率でポジションサイズを計算します。")
            stop_loss_amount_per_unit = price * self.fallback_stop_loss_pct

        if stop_loss_amount_per_unit == 0: # ゼロ除算を避ける
             self.logger.warning("ストップロス幅が0のため、ポジションサイズを計算できません。デフォルト値を返します。")
             return 0.001 # 小さなデフォルト値

        position_size = risk_amount / stop_loss_amount_per_unit
        
        # 手数料考慮
        position_size = position_size * (1 - self.commission_rate * 2)
        position_size = max(position_size, 0) # ポジションサイズが負にならないように
        
        return position_size
    
    def calculate_stop_loss(self, entry_price, direction, current_atr):
        """
        ストップロス価格計算 - ATRベース
        
        Args:
            entry_price (float): エントリー価格
            direction (int): ポジション方向（1: 買い, -1: 売り）
            current_atr (float): 現在のATR値
            
        Returns:
            float: ストップロス価格
        """
        if current_atr is not None and current_atr > 0:
            atr_stop_value = current_atr * self.atr_multiplier_stop_loss
            if direction == 1:  # 買いポジション
                return entry_price - atr_stop_value
            else:  # 売りポジション
                return entry_price + atr_stop_value
        else:
            # ATRが利用できない場合は固定パーセンテージを使用
            if direction == 1:
                return entry_price * (1 - self.fallback_stop_loss_pct)
            else:
                return entry_price * (1 + self.fallback_stop_loss_pct)

    def calculate_take_profit(self, entry_price, direction, current_atr):
        """
        利確価格計算 - ATRベース
        
        Args:
            entry_price (float): エントリー価格
            direction (int): ポジション方向（1: 買い, -1: 売り）
            current_atr (float): 現在のATR値
            
        Returns:
            float: 利確価格
        """
        if current_atr is not None and current_atr > 0:
            atr_take_profit_value = current_atr * self.atr_multiplier_take_profit
            if direction == 1:  # 買いポジション
                return entry_price + atr_take_profit_value
            else:  # 売りポジション
                return entry_price - atr_take_profit_value
        else:
            # ATRが利用できない場合は固定パーセンテージを使用
            if direction == 1:
                return entry_price * (1 + self.fallback_take_profit_pct)
            else:
                return entry_price * (1 - self.fallback_take_profit_pct)

    def update_trailing_stop(self, current_price, entry_price, extremum_price_in_position, direction, current_atr):
        """
        トレーリングストップ更新 - ATRベース
        
        Args:
            current_price (float): 現在価格
            entry_price (float): エントリー価格
            extremum_price_in_position (float): ポジション保有中の最高価格（買い）または最安価格（売り）
            direction (int): ポジション方向（1: 買い, -1: 売り）
            current_atr (float): 現在のATR値
            
        Returns:
            float: トレーリングストップ価格
        """
        if current_atr is not None and current_atr > 0:
            atr_trailing_value = current_atr * self.atr_multiplier_trailing_stop
            if direction == 1:  # 買いポジション
                initial_stop = self.calculate_stop_loss(entry_price, direction, current_atr)
                trailing_stop_level = extremum_price_in_position - atr_trailing_value
                return max(trailing_stop_level, initial_stop)
            else:  # 売りポジション
                initial_stop = self.calculate_stop_loss(entry_price, direction, current_atr)
                trailing_stop_level = extremum_price_in_position + atr_trailing_value
                return min(trailing_stop_level, initial_stop)
        else:
            # ATRが利用できない場合は固定パーセンテージを使用
            if direction == 1:
                initial_stop = entry_price * (1 - self.fallback_stop_loss_pct)
                trailing_stop_level = extremum_price_in_position * (1 - self.fallback_trailing_stop_pct)
                return max(trailing_stop_level, initial_stop)
            else:
                initial_stop = entry_price * (1 + self.fallback_stop_loss_pct)
                trailing_stop_level = extremum_price_in_position * (1 + self.fallback_trailing_stop_pct)
                return min(trailing_stop_level, initial_stop)

    def backtest(self, df, initial_capital=100000):
        """
        拡張バックテスト実行
        
        Args:
            df (pandas.DataFrame): テクニカル指標付きのデータフレーム
            initial_capital (float): 初期資本金
            
        Returns:
            pandas.DataFrame: バックテスト結果
            pandas.DataFrame: 取引履歴
        """
        # 結果格納用の配列
        dates = []
        equity = []  # 純資産
        returns = []  # リターン
        drawdowns = []  # ドローダウン
        positions = []  # ポジション
        trades = []  # 取引履歴
        
        # 初期値設定
        capital = initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        highest_price = 0  # 買いポジション中の最高価格
        lowest_price = float('inf')  # 売りポジション中の最低価格
        direction = 0  # 現在のポジション方向
        trailing_stop = 0  # トレーリングストップ価格
        
        # 資産推移の初期値
        dates.append(df.index[0])
        equity.append(capital)
        returns.append(0)
        drawdowns.append(0)
        positions.append(0)
        
        # 最大資産額の記録
        max_equity = initial_capital
        
        # バックテスト実行
        min_data_points = 200 # シグナル生成に必要な最小データポイント数
        
        for i in range(min_data_points, len(df)):
            current_data_for_signal = df.iloc[max(0, i - min_data_points*2) : i+1]
            current_date = df.index[i]
            current_price = df.iloc[i]['close']
            current_atr = df.iloc[i].get(f'atr_{self.atr_period_for_risk}', None)
            
            if current_atr is None:
                self.logger.debug(f"{current_date} - ATR is None, risk calculations will use fallback.")
            
            # オーバーフローを防ぐために極端に大きい価格を制限
            if position > 0 and position * current_price > 1e12:
                # 異常なポジションを検出した場合、ポジションをリセット
                self.logger.warning(f"異常に大きいポジション価値を検出: {position * current_price}、ポジションをリセットします")
                position = 0
                entry_price = 0
                direction = 0
            
            # ドローダウン計算
            portfolio_value = capital
            if position != 0:
                # 正確な計算のためにオーバーフローを防止
                try:
                    if position > 0:
                        portfolio_value += position * current_price
                    else:
                        position_abs = abs(position)
                        # 空売りの場合の計算
                        short_value = position_abs * (2 * entry_price - current_price)
                        if short_value > 0 and short_value < 1e12:  # 極端な値を制限
                            portfolio_value += short_value
                except OverflowError:
                    self.logger.warning(f"ポートフォリオ価値計算でオーバーフロー: position={position}, price={current_price}")
                    # オーバーフロー時は現在の資金を使用
                    portfolio_value = capital
            
            # ポートフォリオ価値が極端に大きい場合は制限
            if portfolio_value > 1e12:
                portfolio_value = 1e12
                
            max_equity = max(max_equity, portfolio_value)
            current_drawdown = (max_equity - portfolio_value) / max_equity * 100 if max_equity > 0 else 0
            
            # ポジションがない場合、新規シグナルをチェック
            if position == 0:
                signal = self.generate_signal(current_data_for_signal)
                
                if signal == 1:  # 買いシグナル
                    self.logger.info(f"{current_date} - BUY signal generated at {current_price:.2f}")
                    position_size = self.calculate_position_size(capital, current_price, current_atr)
                    position_size = min(position_size, 100000)
                    
                    # スリッページ適用
                    actual_entry_price = self.apply_slippage(current_price, 1)
                    cost = position_size * actual_entry_price
                    commission = self.calculate_commission(cost)
                    
                    if cost + commission <= capital:
                        position = position_size
                        entry_price = actual_entry_price
                        entry_date = current_date
                        capital -= (cost + commission)
                        direction = 1
                        highest_price = current_price # extremum_price_in_position
                        trailing_stop = self.calculate_stop_loss(entry_price, direction, current_atr)
                        
                        current_atr_str = f"{current_atr:.2f}" if current_atr is not None else "N/A"
                        trades.append({
                            'date': current_date,
                            'type': 'BUY',
                            'price': actual_entry_price,
                            'size': position_size,
                            'cost': cost,
                            'commission': commission,
                            'capital': capital,
                            'signal_details': f'Signal: {signal}, ATR: {current_atr_str}'
                        })
                        self.logger.info(f"{current_date} - EXECUTED BUY: Price={actual_entry_price:.2f}, Size={position_size:.4f}, Capital={capital:.2f}")
                
                elif signal == -1:  # 売りシグナル
                    self.logger.info(f"{current_date} - SELL signal generated at {current_price:.2f}")
                    position_size = self.calculate_position_size(capital, current_price, current_atr)
                    position_size = min(position_size, 100000)
                    
                    # スリッページ適用
                    actual_entry_price = self.apply_slippage(current_price, -1)
                    cost = position_size * actual_entry_price
                    commission = self.calculate_commission(cost)
                    
                    if cost <= capital:
                        position = -position_size
                        entry_price = actual_entry_price
                        entry_date = current_date
                        capital -= commission  # 空売りの場合は手数料のみ
                        direction = -1
                        lowest_price = current_price # extremum_price_in_position
                        trailing_stop = self.calculate_stop_loss(entry_price, direction, current_atr)
                        
                        current_atr_str = f"{current_atr:.2f}" if current_atr is not None else "N/A"
                        trades.append({
                            'date': current_date,
                            'type': 'SELL',
                            'price': actual_entry_price,
                            'size': position_size,
                            'cost': 0, # 空売りのコストは実際には発生しない（証拠金）
                            'commission': commission,
                            'capital': capital,
                            'signal_details': f'Signal: {signal}, ATR: {current_atr_str}'
                        })
                        self.logger.info(f"{current_date} - EXECUTED SELL (SHORT): Price={actual_entry_price:.2f}, Size={position_size:.4f}, Capital={capital:.2f}")
            
            # 買いポジションの場合
            elif position > 0:
                # 最高価格更新
                highest_price = max(highest_price, current_price)
                
                # トレーリングストップ更新
                trailing_stop = self.update_trailing_stop(current_price, entry_price, highest_price, direction, current_atr)
                
                # 損切り条件
                if current_price <= trailing_stop and self.use_stop_loss:
                    self.logger.info(f"{current_date} - STOP LOSS triggered for BUY position at {current_price:.2f} (Stop: {trailing_stop:.2f})")
                    # スリッページ適用
                    actual_exit_price = self.apply_slippage(current_price, -1)
                    proceeds = position * actual_exit_price
                    commission = self.calculate_commission(proceeds)
                    gain = proceeds - (position * entry_price) - commission
                    
                    # パーセント計算でオーバーフロー防止
                    try:
                        pct_gain = (actual_exit_price / entry_price - 1) * 100
                    except (OverflowError, ZeroDivisionError):
                        pct_gain = 0
                        
                    capital += proceeds - commission
                    
                    # 資金が極端に大きい場合は制限
                    if capital > 1e12:
                        capital = 1e12
                    
                    trades.append({
                        'date': current_date,
                        'type': 'STOP_LOSS',
                        'price': actual_exit_price,
                        'size': position,
                        'proceeds': proceeds,
                        'commission': commission,
                        'pnl': gain,
                        'pnl_pct': pct_gain,
                        'capital': capital,
                        'hold_days': (current_date - entry_date).days,
                        'exit_reason': 'Stop Loss'
                    })
                    self.logger.info(f"{current_date} - CLOSED BUY (Stop Loss): Price={actual_exit_price:.2f}, PnL={gain:.2f}, Capital={capital:.2f}")
                    
                    position = 0
                    entry_price = 0
                    direction = 0
                
                # 利確条件
                elif current_price >= self.calculate_take_profit(entry_price, direction, current_atr):
                    self.logger.info(f"{current_date} - TAKE PROFIT triggered for BUY position at {current_price:.2f} (Target: {self.calculate_take_profit(entry_price, direction, current_atr):.2f})")
                    # スリッページ適用
                    actual_exit_price = self.apply_slippage(current_price, -1)
                    proceeds = position * actual_exit_price
                    commission = self.calculate_commission(proceeds)
                    gain = proceeds - (position * entry_price) - commission
                    
                    # パーセント計算でオーバーフロー防止
                    try:
                        pct_gain = (actual_exit_price / entry_price - 1) * 100
                    except (OverflowError, ZeroDivisionError):
                        pct_gain = 0
                        
                    capital += proceeds - commission
                    
                    # 資金が極端に大きい場合は制限
                    if capital > 1e12:
                        capital = 1e12
                    
                    trades.append({
                        'date': current_date,
                        'type': 'TAKE_PROFIT',
                        'price': actual_exit_price,
                        'size': position,
                        'proceeds': proceeds,
                        'commission': commission,
                        'pnl': gain,
                        'pnl_pct': pct_gain,
                        'capital': capital,
                        'hold_days': (current_date - entry_date).days,
                        'exit_reason': 'Take Profit'
                    })
                    self.logger.info(f"{current_date} - CLOSED BUY (Take Profit): Price={actual_exit_price:.2f}, PnL={gain:.2f}, Capital={capital:.2f}")
                    
                    position = 0
                    entry_price = 0
                    direction = 0
                
                # 売りシグナル発生で決済
                elif self.generate_signal(current_data_for_signal) == -1:
                    self.logger.info(f"{current_date} - EXIT BUY signal generated at {current_price:.2f}")
                    # スリッページ適用
                    actual_exit_price = self.apply_slippage(current_price, -1)
                    proceeds = position * actual_exit_price
                    commission = self.calculate_commission(proceeds)
                    gain = proceeds - (position * entry_price) - commission
                    
                    # パーセント計算でオーバーフロー防止
                    try:
                        pct_gain = (actual_exit_price / entry_price - 1) * 100
                    except (OverflowError, ZeroDivisionError):
                        pct_gain = 0
                        
                    capital += proceeds - commission
                    
                    # 資金が極端に大きい場合は制限
                    if capital > 1e12:
                        capital = 1e12
                    
                    trades.append({
                        'date': current_date,
                        'type': 'SELL_SIGNAL',
                        'price': actual_exit_price,
                        'size': position,
                        'proceeds': proceeds,
                        'commission': commission,
                        'pnl': gain,
                        'pnl_pct': pct_gain,
                        'capital': capital,
                        'hold_days': (current_date - entry_date).days,
                        'exit_reason': 'Opposing Signal'
                    })
                    self.logger.info(f"{current_date} - CLOSED BUY (Opposing Signal): Price={actual_exit_price:.2f}, PnL={gain:.2f}, Capital={capital:.2f}")
                    
                    position = 0
                    entry_price = 0
                    direction = 0
            
            # 売りポジションの場合（同様の修正をここにも適用）
            elif position < 0:
                # 最低価格更新
                lowest_price = min(lowest_price, current_price)
                
                # トレーリングストップ更新
                trailing_stop = self.update_trailing_stop(current_price, entry_price, lowest_price, direction, current_atr)
                
                # 損切り条件
                if current_price >= trailing_stop and self.use_stop_loss:
                    self.logger.info(f"{current_date} - STOP LOSS triggered for SELL position at {current_price:.2f} (Stop: {trailing_stop:.2f})")
                    actual_exit_price = self.apply_slippage(current_price, 1)
                    position_abs = abs(position)
                    
                    # オーバーフロー防止
                    try:
                        proceeds = position_abs * (2 * entry_price - actual_exit_price)
                        proceeds = min(proceeds, 1e12)  # 極端な値を制限
                    except OverflowError:
                        proceeds = 1e12
                        
                    commission = self.calculate_commission(position_abs * actual_exit_price)
                    gain = proceeds - commission
                    
                    # パーセント計算でオーバーフロー防止
                    try:
                        pct_gain = (entry_price / actual_exit_price - 1) * 100
                    except (OverflowError, ZeroDivisionError):
                        pct_gain = 0
                        
                    capital += proceeds - commission
                    
                    # 資金が極端に大きい場合は制限
                    if capital > 1e12:
                        capital = 1e12
                    
                    trades.append({
                        'date': current_date,
                        'type': 'STOP_LOSS',
                        'price': actual_exit_price,
                        'size': position_abs,
                        'proceeds': proceeds,
                        'commission': commission,
                        'pnl': gain,
                        'pnl_pct': pct_gain,
                        'capital': capital,
                        'hold_days': (current_date - entry_date).days,
                        'exit_reason': 'Stop Loss'
                    })
                    self.logger.info(f"{current_date} - CLOSED SELL (Stop Loss): Price={actual_exit_price:.2f}, PnL={gain:.2f}, Capital={capital:.2f}")
                    
                    position = 0
                    entry_price = 0
                    direction = 0
                
                # 利確条件
                elif current_price <= self.calculate_take_profit(entry_price, direction, current_atr):
                    self.logger.info(f"{current_date} - TAKE PROFIT triggered for SELL position at {current_price:.2f} (Target: {self.calculate_take_profit(entry_price, direction, current_atr):.2f})")
                    actual_exit_price = self.apply_slippage(current_price, 1)
                    position_abs = abs(position)
                    
                    # オーバーフロー防止
                    try:
                        proceeds = position_abs * (2 * entry_price - actual_exit_price)
                        proceeds = min(proceeds, 1e12)  # 極端な値を制限
                    except OverflowError:
                        proceeds = 1e12
                        
                    commission = self.calculate_commission(position_abs * actual_exit_price)
                    gain = proceeds - commission
                    
                    # パーセント計算でオーバーフロー防止
                    try:
                        pct_gain = (entry_price / actual_exit_price - 1) * 100
                    except (OverflowError, ZeroDivisionError):
                        pct_gain = 0
                        
                    capital += proceeds - commission
                    
                    # 資金が極端に大きい場合は制限
                    if capital > 1e12:
                        capital = 1e12
                    
                    trades.append({
                        'date': current_date,
                        'type': 'TAKE_PROFIT',
                        'price': actual_exit_price,
                        'size': position_abs,
                        'proceeds': proceeds,
                        'commission': commission,
                        'pnl': gain,
                        'pnl_pct': pct_gain,
                        'capital': capital,
                        'hold_days': (current_date - entry_date).days,
                        'exit_reason': 'Take Profit'
                    })
                    self.logger.info(f"{current_date} - CLOSED SELL (Take Profit): Price={actual_exit_price:.2f}, PnL={gain:.2f}, Capital={capital:.2f}")
                    
                    position = 0
                    entry_price = 0
                    direction = 0
                
                # 買いシグナル発生で決済
                elif self.generate_signal(current_data_for_signal) == 1:
                    self.logger.info(f"{current_date} - EXIT SELL signal generated at {current_price:.2f}")
                    actual_exit_price = self.apply_slippage(current_price, 1)
                    position_abs = abs(position)
                    
                    # オーバーフロー防止
                    try:
                        proceeds = position_abs * (2 * entry_price - actual_exit_price)
                        proceeds = min(proceeds, 1e12)  # 極端な値を制限
                    except OverflowError:
                        proceeds = 1e12
                        
                    commission = self.calculate_commission(position_abs * actual_exit_price)
                    gain = proceeds - commission
                    
                    # パーセント計算でオーバーフロー防止
                    try:
                        pct_gain = (entry_price / actual_exit_price - 1) * 100
                    except (OverflowError, ZeroDivisionError):
                        pct_gain = 0
                        
                    capital += proceeds - commission
                    
                    # 資金が極端に大きい場合は制限
                    if capital > 1e12:
                        capital = 1e12
                    
                    trades.append({
                        'date': current_date,
                        'type': 'BUY_SIGNAL',
                        'price': actual_exit_price,
                        'size': position_abs,
                        'proceeds': proceeds,
                        'commission': commission,
                        'pnl': gain,
                        'pnl_pct': pct_gain,
                        'capital': capital,
                        'hold_days': (current_date - entry_date).days,
                        'exit_reason': 'Opposing Signal'
                    })
                    self.logger.info(f"{current_date} - CLOSED SELL (Opposing Signal): Price={actual_exit_price:.2f}, PnL={gain:.2f}, Capital={capital:.2f}")
                    
                    position = 0
                    entry_price = 0
                    direction = 0
            
            # ポートフォリオ価値の更新（オーバーフロー防止）
            portfolio_value = capital
            if position != 0:
                try:
                    if position > 0:
                        portfolio_value += position * current_price
                    else:
                        position_abs = abs(position)
                        # 空売りの場合の計算
                        short_value = position_abs * (2 * entry_price - current_price)
                        if short_value > 0 and short_value < 1e12:  # 極端な値を制限
                            portfolio_value += short_value
                except OverflowError:
                    self.logger.warning(f"ポートフォリオ価値計算でオーバーフロー: position={position}, price={current_price}")
            
            # 価値が極端に大きい場合は制限
            if portfolio_value > 1e12:
                portfolio_value = 1e12
            
            # 資産推移の記録
            dates.append(current_date)
            equity.append(portfolio_value)
            positions.append(position)
            drawdowns.append(current_drawdown)
            
            # リターン計算（オーバーフロー防止）
            try:
                if len(equity) > 1 and equity[-2] > 0:
                    daily_return = (portfolio_value / equity[-2] - 1)
                else:
                    daily_return = 0
            except (OverflowError, ZeroDivisionError):
                daily_return = 0
                
            returns.append(daily_return)
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame({
            'date': dates,
            'equity': equity,
            'returns': returns,
            'drawdowns': drawdowns,
            'positions': positions
        })
        
        # 取引履歴をDataFrameに変換
        trades_df = pd.DataFrame(trades)
        
        # 結果ディレクトリのパス
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, 'results')
        
        # 結果ディレクトリが存在しない場合は作成
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # 結果をCSV保存
        results_csv = os.path.join(results_dir, 'advanced_backtest_results.csv')
        trades_csv = os.path.join(results_dir, 'advanced_backtest_trades.csv')
        
        results_df.to_csv(results_csv, index=False)
        trades_df.to_csv(trades_csv, index=False)
        
        self.logger.info(f"バックテスト結果を保存しました: {results_csv}, {trades_csv}")
        
        return results_df, trades_df

class PerformanceMetrics:
    """パフォーマンス評価指標の計算"""
    
    @staticmethod
    def calculate_metrics(results_df, trades_df, risk_free_rate=0.0):
        """
        パフォーマンス指標を計算
        
        Args:
            results_df (pandas.DataFrame): バックテスト結果
            trades_df (pandas.DataFrame): 取引履歴
            risk_free_rate (float): リスクフリーレート（年率）
            
        Returns:
            dict: パフォーマンス指標
        """
        metrics = {}
        
        # 初期資本と最終資本
        initial_equity = results_df['equity'].iloc[0]
        final_equity = results_df['equity'].iloc[-1]
        
        # 資本が極端に大きい場合、制限する
        if initial_equity > 1e12:
            initial_equity = 1e12
        if final_equity > 1e12:
            final_equity = 1e12
        
        # 総リターン
        try:
            if initial_equity > 0:
                total_return = (final_equity / initial_equity - 1) * 100
                # 極端に大きい値は制限
                total_return = min(total_return, 1e6)
            else:
                total_return = 0
        except (OverflowError, ZeroDivisionError):
            total_return = 1e6  # 非常に大きい値に制限
            
        metrics['total_return'] = total_return
        
        # 年率リターン
        # results_dfのindexがDatetimeIndexか確認して適切に処理
        if isinstance(results_df.index, pd.DatetimeIndex):
            days = (results_df.index[-1] - results_df.index[0]).days
        else:
            # DatetimeIndexでない場合は、バックテスト期間を1年と仮定
            days = 365
        years = days / 365
        
        try:
            if years > 0 and total_return > -100:  # -100%以下の場合は計算できない
                annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
                # 極端に大きい値は制限
                annual_return = min(annual_return, 1e6)
            else:
                annual_return = 0
        except (OverflowError, ValueError):
            annual_return = 1e6  # 非常に大きい値に制限
            
        metrics['annual_return'] = annual_return
        
        # 最大ドローダウン
        # 'drawdowns'列と'drawdown'列の存在をチェック
        if 'drawdowns' in results_df.columns and 'drawdown' not in results_df.columns:
            results_df['drawdown'] = results_df['drawdowns']
        elif 'drawdown' not in results_df.columns and 'drawdowns' not in results_df.columns:
            # どちらの列も存在しない場合は計算
            # 各時点での最大資産からの下落率を計算
            results_df['equity_peak'] = results_df['equity'].cummax()
            results_df['drawdown'] = (results_df['equity_peak'] - results_df['equity']) / results_df['equity_peak'] * 100
            
        max_drawdown = results_df['drawdown'].max()
        # 極端に小さいドローダウンは計算誤差の可能性があるので、最小値を設定
        max_drawdown = max(max_drawdown, 0.01)
        metrics['max_drawdown'] = max_drawdown
        
        # 最大ドローダウン期間
        drawdown_periods = []
        current_drawdown_start = None
        
        # date列をインデックスに設定（結果をdateでインデックス付けする）
        if 'date' in results_df.columns and not isinstance(results_df.index, pd.DatetimeIndex):
            results_df = results_df.set_index('date')
        
        # Indexがdatetimeかどうかでドローダウン期間の計算方法を変更
        if isinstance(results_df.index, pd.DatetimeIndex):
            for i, row in results_df.iterrows():
                if row['drawdown'] > 0 and current_drawdown_start is None:
                    current_drawdown_start = i
                elif row['drawdown'] == 0 and current_drawdown_start is not None:
                    drawdown_periods.append((current_drawdown_start, i, (i - current_drawdown_start).days))
                    current_drawdown_start = None
            
            if current_drawdown_start is not None:
                drawdown_periods.append((current_drawdown_start, results_df.index[-1], (results_df.index[-1] - current_drawdown_start).days))
        else:
            # dateではなく行番号を使用する場合
            for idx, (i, row) in enumerate(results_df.iterrows()):
                if row['drawdown'] > 0 and current_drawdown_start is None:
                    current_drawdown_start = idx
                elif row['drawdown'] == 0 and current_drawdown_start is not None:
                    drawdown_periods.append((current_drawdown_start, idx, idx - current_drawdown_start))
                    current_drawdown_start = None
            
            if current_drawdown_start is not None:
                drawdown_periods.append((current_drawdown_start, len(results_df)-1, len(results_df)-1 - current_drawdown_start))
        
        max_drawdown_period = max(drawdown_periods, key=lambda x: x[2]) if drawdown_periods else (None, None, 0)
        metrics['max_drawdown_days'] = max_drawdown_period[2]
        
        # 回復係数
        try:
            if max_drawdown > 0:
                recovery_factor = abs(total_return / max_drawdown)
                # 極端に大きい値は制限
                recovery_factor = min(recovery_factor, 1e6)
            else:
                recovery_factor = 1e6  # ドローダウンがない場合は非常に大きい値
        except (OverflowError, ZeroDivisionError):
            recovery_factor = 1e6
            
        metrics['recovery_factor'] = recovery_factor
        
        # シャープレシオ
        daily_returns = results_df['returns'].iloc[1:]  # 最初の0を除外
        # 極端な値を除外
        daily_returns = daily_returns.clip(-1, 1)
        
        try:
            annual_volatility = daily_returns.std() * np.sqrt(252)  # 年率ボラティリティ
            
            # ボラティリティがゼロまたは非常に小さい場合の対処
            if annual_volatility > 0.0001:
                sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
                # 極端に大きい値は制限
                sharpe_ratio = min(sharpe_ratio, 1e6)
            else:
                sharpe_ratio = 0
        except (OverflowError, ZeroDivisionError):
            sharpe_ratio = 0
            
        metrics['sharpe_ratio'] = sharpe_ratio
        
        # ソルティノレシオ（下方リスクのみ考慮）
        downside_returns = daily_returns[daily_returns < 0]
        
        try:
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # 下方ボラティリティがゼロまたは非常に小さい場合の対処
            if downside_volatility > 0.0001:
                sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
                # 極端に大きい値は制限
                sortino_ratio = min(sortino_ratio, 1e6)
            else:
                sortino_ratio = 0
        except (OverflowError, ZeroDivisionError):
            sortino_ratio = 0
            
        metrics['sortino_ratio'] = sortino_ratio
        
        # 利益係数
        if not trades_df.empty and 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            num_winning_trades = len(winning_trades)
            num_losing_trades = len(losing_trades)
            
            try:
                win_rate = num_winning_trades / len(trades_df) if len(trades_df) > 0 else 0
                metrics['win_rate'] = win_rate * 100  # パーセンテージに変換
            except ZeroDivisionError:
                metrics['win_rate'] = 0
            
            try:
                # 異常値を除外して平均計算
                if num_winning_trades > 0:
                    win_values = winning_trades['pnl'].values
                    win_values = win_values[win_values < 1e12]  # 極端な値を除外
                    avg_win = np.mean(win_values) if len(win_values) > 0 else 0
                else:
                    avg_win = 0
                    
                if num_losing_trades > 0:
                    loss_values = np.abs(losing_trades['pnl'].values)
                    loss_values = loss_values[loss_values < 1e12]  # 極端な値を除外
                    avg_loss = np.mean(loss_values) if len(loss_values) > 0 else 0
                else:
                    avg_loss = 0
                
                # 利益係数計算
                if num_losing_trades > 0 and avg_loss > 0:
                    profit_factor = (avg_win * num_winning_trades) / (avg_loss * num_losing_trades)
                    # 極端に大きい値は制限
                    profit_factor = min(profit_factor, 1e6)
                else:
                    profit_factor = 1e6 if num_winning_trades > 0 else 0
            except (OverflowError, ZeroDivisionError):
                profit_factor = 0
                
            metrics['profit_factor'] = profit_factor
            
            metrics['num_trades'] = len(trades_df)
            
            # 平均トレードリターン
            if 'pnl_pct' in trades_df.columns:
                pct_values = trades_df['pnl_pct'].values
                pct_values = pct_values[np.abs(pct_values) < 1e6]  # 極端な値を除外
                metrics['avg_trade_return'] = np.mean(pct_values) if len(pct_values) > 0 else 0
            else:
                metrics['avg_trade_return'] = 0
            
            # 平均保有期間
            if 'hold_days' in trades_df.columns:
                hold_days = trades_df['hold_days'].values
                hold_days = hold_days[hold_days < 365]  # 異常に長い保有期間を除外
                metrics['avg_hold_days'] = np.mean(hold_days) if len(hold_days) > 0 else 0
        
        # カルマーレシオ
        try:
            if max_drawdown > 0:
                calmar_ratio = annual_return / max_drawdown
                # 極端に大きい値は制限
                calmar_ratio = min(calmar_ratio, 1e6)
            else:
                calmar_ratio = 1e6  # ドローダウンがない場合は非常に大きい値
        except (OverflowError, ZeroDivisionError):
            calmar_ratio = 0
            
        metrics['calmar_ratio'] = calmar_ratio
        
        # MAR比率（最小許容リターン比率）
        metrics['mar_ratio'] = calmar_ratio  # カルマーレシオと同じ
        
        # オメガレシオ（目標リターンを超える確率）
        try:
            target_return = 0  # 目標リターン（日次）
            above_target = len(daily_returns[daily_returns > target_return])
            below_target = len(daily_returns[daily_returns <= target_return])
            
            if below_target > 0:
                omega_ratio = above_target / below_target
                # 極端に大きい値は制限
                omega_ratio = min(omega_ratio, 1e6)
            else:
                omega_ratio = 1e6 if above_target > 0 else 0
        except (OverflowError, ZeroDivisionError):
            omega_ratio = 0
            
        metrics['omega_ratio'] = omega_ratio
        
        # リスク調整後リターン
        try:
            if annual_volatility > 0.0001:
                risk_adjusted_return = annual_return / (annual_volatility * 100)
                # 極端に大きい値は制限
                risk_adjusted_return = min(risk_adjusted_return, 1e6)
            else:
                risk_adjusted_return = 0
        except (OverflowError, ZeroDivisionError):
            risk_adjusted_return = 0
            
        metrics['risk_adjusted_return'] = risk_adjusted_return
        
        return metrics

class SimpleBacktest:
    """
    バックテストを実行するためのクラス
    """
    def __init__(self, data, symbol, timeframe, initial_capital=1000000):
        """
        初期化
        
        Args:
            data (pandas.DataFrame): 履歴データ
            symbol (str): 取引通貨ペア
            timeframe (str): 時間枠
            initial_capital (float): 初期資本
        """
        self.data = data
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 設定から戦略パラメータを取得
        strategy_params = config.get('strategy_params', {})
        risk_params = config.get('risk_params', {})
        commission_rate = config.get('commission_rate', 0.001)
        slippage_pct = config.get('slippage_pct', 0.001)
        use_stop_loss = config.get('use_stop_loss', True)
        
        # 戦略インスタンスを作成
        self.strategy = AdvancedStrategy(
            symbol=symbol,
            strategy_params=strategy_params,
            risk_params=risk_params,
            commission_rate=commission_rate,
            slippage_pct=slippage_pct,
            use_stop_loss=use_stop_loss
        )
        
        # パフォーマンス指標
        self.metrics = None
        self.results = None
        self.trades = None
        
    def run(self):
        """
        バックテストを実行
        
        Returns:
            tuple: (結果データフレーム, 取引データフレーム)
        """
        self.logger.info(f"バックテスト開始: {self.symbol}, {self.timeframe}")
        
        # データにテクニカル指標を追加
        from data.simple_data_fetcher import SimpleDataFetcher
        data_fetcher = SimpleDataFetcher(timeframe=self.timeframe, use_real_data=True)
        self.data = data_fetcher.add_technical_indicators(self.data)
        
        # 戦略のバックテストを実行
        self.results, self.trades = self.strategy.backtest(
            df=self.data,
            initial_capital=self.initial_capital
        )
        
        # メトリクスを計算
        self.metrics = PerformanceMetrics.calculate_metrics(self.results, self.trades)
        
        self.logger.info(f"バックテスト完了: {self.symbol}, {self.timeframe}")
        
        return self.results, self.trades
    
    def print_metrics(self):
        """メトリクスを表示"""
        if self.metrics is None:
            self.logger.warning("メトリクスが計算されていません。先にrun()を実行してください。")
            return
            
        self.logger.info("======== バックテスト結果 ========")
        for key, value in self.metrics.items():
            self.logger.info(f"{key}: {value:.4f}")

def parse_args():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(description='高信頼性バックテストツール')
    parser.add_argument('--symbol', type=str, default=config.get('default_symbol', 'BTC/USDT'), help='取引通貨ペア（例: BTC/USDT）')
    parser.add_argument('--timeframe', type=str, default=config.get('default_timeframe', '1h'), help='時間枠（例: 1h, 1d）')
    parser.add_argument('--capital', type=float, default=config.get('default_initial_capital', 1000000), help='初期資本金')
    parser.add_argument('--years', type=int, default=config.get('default_backtest_years', 10), help='バックテスト期間（年数）')
    parser.add_argument('--no_stop_loss', action='store_true', help='ストップロスを無効化')
    parser.add_argument('--commission', type=float, default=config.get('commission_rate', 0.001), help='取引手数料率（片道）')
    parser.add_argument('--slippage', type=float, default=config.get('slippage_percentage', 0.001), help='スリッページ率')
    parser.add_argument('--plot', action='store_true', default=config.get('plot_results_default', False), help='結果をプロットする')
    parser.add_argument('--use_mock_data', action='store_true', default=config.get('use_mock_data_default', False), help='模擬データを使用する (APIキー不要)')
    return parser.parse_args()

def plot_results(results_df, trades_df, symbol):
    """
    バックテスト結果のプロット
    
    Args:
        results_df (pandas.DataFrame): バックテスト結果
        trades_df (pandas.DataFrame): 取引履歴
        symbol (str): 通貨ペア
    """
    # フォント設定
    setup_japanese_fonts()

    plt.figure(figsize=(16, 12))
    
    # 複数のサブプロットを作成
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # 資産推移プロット
    ax1 = plt.subplot(gs[0])
    ax1.plot(results_df.index, results_df['equity'], label='ポートフォリオ価値')
    ax1.set_title(f'{symbol} バックテスト結果 - 資産推移')
    ax1.set_ylabel('資産 (円)')
    ax1.grid(True)
    ax1.legend()
    
    # 取引プロット
    if not trades_df.empty:
        # 買いトレード
        buy_trades = trades_df[trades_df['type'].isin(['BUY', 'BUY_SIGNAL'])]
        if not buy_trades.empty:
            ax1.scatter(buy_trades.index, [results_df.loc[date, 'equity'] for date in buy_trades.index], 
                       marker='^', color='green', s=100, label='買い')
        
        # 売りトレード
        sell_trades = trades_df[trades_df['type'].isin(['SELL', 'SELL_SIGNAL'])]
        if not sell_trades.empty:
            ax1.scatter(sell_trades.index, [results_df.loc[date, 'equity'] for date in sell_trades.index], 
                       marker='v', color='red', s=100, label='売り')
        
        # 利確
        take_profit_trades = trades_df[trades_df['type'] == 'TAKE_PROFIT']
        if not take_profit_trades.empty:
            ax1.scatter(take_profit_trades.index, [results_df.loc[date, 'equity'] for date in take_profit_trades.index], 
                       marker='o', color='blue', s=100, label='利確')
        
        # 損切り
        stop_loss_trades = trades_df[trades_df['type'] == 'STOP_LOSS']
        if not stop_loss_trades.empty:
            ax1.scatter(stop_loss_trades.index, [results_df.loc[date, 'equity'] for date in stop_loss_trades.index], 
                       marker='x', color='black', s=100, label='損切り')
    
    # リターンプロット
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(results_df.index[1:], results_df['returns'].iloc[1:] * 100, label='日次リターン')
    ax2.set_ylabel('リターン (%)')
    ax2.grid(True)
    ax2.legend()
    
    # ドローダウンプロット
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.fill_between(results_df.index, 0, results_df['drawdown'], color='red', alpha=0.3, label='ドローダウン')
    ax3.set_ylabel('ドローダウン (%)')
    ax3.set_xlabel('日付')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    # 結果画像のパス
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plot_file = os.path.join(results_dir, f'backtest_result_{symbol.replace("/", "_")}.png')
    plt.savefig(plot_file)
    plt.close()
    
    logger.info(f"バックテスト結果プロットを保存しました: {plot_file}")

def main():
    """メイン関数"""
    args = parse_args()
    
    symbol = args.symbol
    timeframe = args.timeframe
    initial_capital = args.capital
    years = args.years
    use_stop_loss = not args.no_stop_loss
    commission_rate = args.commission
    slippage = args.slippage
    plot_results_flag = args.plot
    use_mock_data = args.use_mock_data
    
    logger.info(f"======== バックテストを開始します ========")
    logger.info(f"設定: {symbol} / {timeframe} / {years}年間 / 資本金: {initial_capital} / ストップロス: {use_stop_loss}")
    
    try:
        # config.ymlから戦略パラメータとリスクパラメータを取得
        strategy_params = config.get('strategy_params', {})
        risk_params = config.get('risk_params', {})
        
        # データ取得
        data_fetcher = SimpleDataFetcher(timeframe=timeframe, use_real_data=not use_mock_data)
        df = data_fetcher.get_data(symbol, years=years)
        
        # インジケータ追加
        df = data_fetcher.add_technical_indicators(df)
        
        # 戦略クラスのインスタンス化
        strategy = AdvancedStrategy(
            symbol=symbol,
            strategy_params=strategy_params,
            risk_params=risk_params,
            commission_rate=commission_rate,
            slippage_pct=slippage,
            use_stop_loss=use_stop_loss
        )
        
        # バックテスト実行
        results_df, trades_df = strategy.backtest(df, initial_capital=initial_capital)
        
        # パフォーマンス指標計算
        metrics = PerformanceMetrics.calculate_metrics(results_df, trades_df)
        
        # 結果出力
        logger.info(f"======== バックテスト結果 ========")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        # グラフ描画
        if plot_results_flag:
            plot_results(results_df, trades_df, symbol)
            
    except Exception as e:
        logger.error(f"バックテスト実行中にエラーが発生しました: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 