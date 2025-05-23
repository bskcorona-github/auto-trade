"""
改良版データ取得モジュール
実際の取引所データと現実的な価格変動モデルをサポート
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
import math
import ccxt
import time
from dotenv import load_dotenv
import yaml
# pandas-taライブラリのインポートをコメントアウト
# import pandas_ta as ta 

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 設定ファイルの読み込み関数
def load_config():
    # スクリプト(simple_data_fetcher.py)が src/data にあるため、プロジェクトルートは3階層上
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'config.yml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # ロガーがこの時点で利用可能か不明なため、printも残す
        print(f"警告(DataFetcher): 設定ファイルが見つかりません: {config_path}")
        logging.error(f"設定ファイルが見つかりません: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"警告(DataFetcher): 設定ファイルの読み込みエラー: {e}")
        logging.error(f"設定ファイルの読み込みエラー: {e}")
        return {}

config = load_config() # グローバルに設定を読み込む

class SimpleDataFetcher:
    """データ取得クラス"""
    
    def __init__(self, timeframe='1h', use_real_data=True):
        """
        初期化
        
        Args:
            timeframe (str): 時間枠
            use_real_data (bool): 実際のデータを使用するかどうか
        """
        self.timeframe = timeframe
        self.use_real_data = use_real_data
        self.logger = self._setup_logger()
        self.exchange = self._setup_exchange() if use_real_data else None
        self.logger.info(f"データフェッチャーが初期化されました - 時間枠: {timeframe}, 実データ使用: {use_real_data}")
        
    def _setup_logger(self):
        """ロガーセットアップ"""
        logger = logging.getLogger('DataFetcher')
        logger.setLevel(logging.INFO)
        
        # ログディレクトリの作成 (logs/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        logs_dir = os.path.join(project_root, 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # ファイルハンドラを追加
        log_file = os.path.join(logs_dir, 'data_fetcher.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 標準出力ハンドラも追加
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_exchange(self):
        """取引所セットアップ"""
        try:
            # .envファイルから環境変数を読み込み (優先)
            load_dotenv()
            api_key_env = os.getenv('BINANCE_API_KEY')
            secret_key_env = os.getenv('BINANCE_SECRET_KEY')
            
            api_key = api_key_env
            secret_key = secret_key_env

            # .env にない場合、config.yml から読み込み試行 (主に構造確認用、実キーは非推奨)
            if not api_key and 'binance_api_key' in config:
                api_key = config.get('binance_api_key')
                self.logger.info("config.yml から APIキープレースホルダーを読み込みました。(本番環境非推奨)")
            if not secret_key and 'binance_secret_key' in config:
                secret_key = config.get('binance_secret_key')
                self.logger.info("config.yml から Secretキープレースホルダーを読み込みました。(本番環境非推奨)")

            if not api_key or not secret_key or api_key == 'BINANCE_API_KEY':
                self.logger.warning("有効なAPIキーが設定されていません。模擬データを使用します。")
                self.use_real_data = False
                return None
            
            # Binance取引所インスタンスを作成
            exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # 先物取引を使用する場合
                }
            })
            
            # 接続テスト
            exchange.load_markets()
            self.logger.info("Binance APIに接続しました")
            return exchange
            
        except Exception as e:
            self.logger.error(f"取引所接続エラー: {e}")
            self.logger.warning("取引所への接続に失敗しました。模擬データを使用します。")
            self.use_real_data = False
            return None
    
    def fetch_historical_data(self, symbol, years=1):
        """
        取引所から歴史的価格データを取得
        
        Args:
            symbol (str): 取引通貨ペア
            years (int): 取得する年数
            
        Returns:
            pandas.DataFrame: 歴史的OHLCVデータ
        """
        if not self.use_real_data or self.exchange is None:
            self.logger.error("実データが使用できません。APIキーを確認してください。")
            return None
            
        try:
            # APIの形式に合わせて通貨ペア名を変換
            formatted_symbol = symbol.replace('/', '')
            
            # タイムフレームをccxtの形式に変換
            timeframe_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            ccxt_timeframe = timeframe_map.get(self.timeframe, '1h')
            
            # 期間を計算
            end_time = int(time.time() * 1000)  # 現在時刻（ミリ秒）
            start_time = end_time - (years * 365 * 24 * 60 * 60 * 1000)  # 指定年数前
            
            all_data = []
            current_start = start_time
            
            # 取引所のレート制限を考慮して、小さな期間に分割して取得
            while current_start < end_time:
                current_end = min(current_start + (100 * 24 * 60 * 60 * 1000), end_time)  # 100日ごと
                
                # データ取得
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=formatted_symbol,
                    timeframe=ccxt_timeframe,
                    since=current_start,
                    limit=1000  # 取得可能な最大数
                )
                
                if not ohlcv:
                    self.logger.warning(f"期間 {datetime.fromtimestamp(current_start/1000)} から "
                                      f"{datetime.fromtimestamp(current_end/1000)} のデータが取得できませんでした")
                    current_start = current_end
                    continue
                
                all_data.extend(ohlcv)
                self.logger.info(f"{len(ohlcv)}件のデータを取得しました - "
                               f"{datetime.fromtimestamp(current_start/1000)} から "
                               f"{datetime.fromtimestamp(current_end/1000)}")
                
                # レート制限を考慮して少し待機
                time.sleep(1)
                
                # 次の期間へ
                current_start = current_end
            
            if not all_data:
                self.logger.error(f"データが取得できませんでした。APIキーを確認してください。")
                return None
                
            # データフレームに変換
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # タイムスタンプをインデックスに設定
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 重複行を削除
            df = df[~df.index.duplicated(keep='first')]
            
            # 日付順に並べ替え
            df.sort_index(inplace=True)
            
            self.logger.info(f"{symbol}の実際の歴史データを{len(df)}件取得しました（約{years}年分）")
            
            return df
            
        except Exception as e:
            self.logger.error(f"データ取得エラー: {e}")
            self.logger.error("実データの取得に失敗しました。APIキーを確認してください。")
            return None
    
    def get_data(self, symbol, years=10):
        """
        データ取得（実データのみ）
        
        Args:
            symbol (str): 取引通貨ペア
            years (int): データ期間（年数）
            
        Returns:
            pandas.DataFrame: 価格データ
        """
        if self.use_real_data and self.exchange is not None:
            return self.fetch_historical_data(symbol, years=years)
        else:
            self.logger.error("APIキーが正しく設定されていないか、接続に失敗しました。")
            return None
    
    def generate_mock_data(self, symbol, days=365, years=10):
        """
        バックテスト用の模擬データを生成
        複数の市場環境（上昇・下降・レンジ相場）を含む長期データ
        
        Args:
            symbol (str): 取引通貨ペア
            days (int): 1年あたりの取引日数
            years (int): シミュレーション年数
            
        Returns:
            pandas.DataFrame: 模擬データ
        """
        total_days = days * years
        periods = int(total_days * 24 / self._get_timeframe_hours())  # 時間枠に基づいて期間数を計算
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_days)
        
        # 日付範囲を生成
        date_range = pd.date_range(start=start_date, end=end_date, periods=periods)
        
        # 初期価格
        base_price = 10000  # 基本価格（例: BTC/USDT）
        
        # 異なる市場環境を模倣するためのパラメータ
        np.random.seed(42)  # 再現性のためのシード
        
        # トレンド成分（全体の方向性）- より小さな変動を設定
        trend_factor = 0.0001  # 小さいトレンド係数
        
        # 長期的なトレンドサイクル（数年スパン）
        long_term_cycles = np.sin(np.linspace(0, years * 2 * np.pi, periods)) * 0.05
        
        # 中期的な市場サイクル（数ヶ月スパン）
        medium_term_cycles = np.sin(np.linspace(0, years * 24 * np.pi, periods)) * 0.03
        
        # 短期的な変動（日〜週スパン）
        short_term_cycles = np.sin(np.linspace(0, years * 120 * np.pi, periods)) * 0.01
        
        # ランダムノイズ（日内変動）
        noise = np.random.normal(0, 0.005, periods)
        
        # 市場環境の変化を模倣（強気・弱気市場）
        # 最初の1/3は上昇トレンド
        trend_component = np.zeros(periods)
        bull_market_end = periods // 3
        bear_market_end = bull_market_end * 2
        
        # 上昇トレンド（最初の1/3）
        trend_component[:bull_market_end] = np.linspace(0, 0.2, bull_market_end)
        
        # 下降トレンド（次の1/3）
        trend_component[bull_market_end:bear_market_end] = np.linspace(0.2, -0.1, bear_market_end - bull_market_end)
        
        # レンジ相場（最後の1/3）
        trend_component[bear_market_end:] = np.linspace(-0.1, 0.05, periods - bear_market_end)
        
        # 価格変動モデルの組み合わせ
        # 日次リターンを計算（変動幅を抑える）
        daily_returns = long_term_cycles + medium_term_cycles + short_term_cycles + noise + trend_component * trend_factor
        
        # 価格シリーズに変換（対数リターンを使用してオーバーフローを防止）
        log_returns = np.log1p(daily_returns)
        close_prices = base_price * np.exp(np.cumsum(log_returns))
        
        # ボラティリティクラスタリングを追加（大きな価格変動後はボラティリティが高まる傾向）
        volatility = np.zeros(periods)
        volatility[0] = 0.005
        for i in range(1, periods):
            volatility[i] = 0.95 * volatility[i-1] + 0.05 * abs(daily_returns[i-1])
            volatility[i] = min(volatility[i], 0.03)  # ボラティリティ上限を設定
        
        # ボラティリティに基づいて高値・安値を生成
        high_prices = close_prices * (1 + volatility * 0.5)
        low_prices = close_prices * (1 - volatility * 0.5)
        
        # 始値は前日終値からの変動
        open_prices = np.zeros(periods)
        open_prices[0] = close_prices[0] * 0.995  # 最初の始値
        for i in range(1, periods):
            open_prices[i] = close_prices[i-1] * (1 + np.random.normal(0, 0.001))
        
        # ボリュームも価格変動と相関
        volume = np.random.rand(periods) * 100 + 20  # 基本ボリューム
        for i in range(periods):
            # 価格変動が大きいほどボリュームも大きくなる
            volume[i] *= (1 + abs(daily_returns[i]) * 5)
        
        # 特定期間に急騰・急落イベントを追加（市場ショック）
        # 2020年3月のコロナショックのようなクラッシュ
        crash_start = periods // 4
        for i in range(crash_start, crash_start + periods // 50):
            crash_factor = np.exp(-0.3 * (i - crash_start))
            close_prices[i] *= (1 - 0.01 * crash_factor)
            low_prices[i] *= (1 - 0.015 * crash_factor)
            high_prices[i] *= (1 - 0.005 * crash_factor)
            volume[i] *= (1 + crash_factor * 0.5)
        
        # 2017年末のような急騰相場
        bubble_start = periods // 2
        for i in range(bubble_start, bubble_start + periods // 40):
            bubble_factor = min(1, (i - bubble_start) / (periods // 80))
            close_prices[i] *= (1 + 0.01 * bubble_factor)
            high_prices[i] *= (1 + 0.015 * bubble_factor)
            low_prices[i] *= (1 + 0.005 * bubble_factor)
            volume[i] *= (1 + bubble_factor * 0.5)
        
        # DataFrameを作成
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=date_range)
        
        self.logger.info(f"{symbol}のバックテスト用模擬データを{len(df)}件生成しました（{years}年分）")
        return df
    
    def _get_timeframe_hours(self):
        """時間枠を時間単位に変換"""
        if self.timeframe == '1m':
            return 1/60
        elif self.timeframe == '5m':
            return 5/60
        elif self.timeframe == '15m':
            return 15/60
        elif self.timeframe == '30m':
            return 30/60
        elif self.timeframe == '1h':
            return 1
        elif self.timeframe == '4h':
            return 4
        elif self.timeframe == '1d':
            return 24
        else:
            return 1  # デフォルト: 1時間
    
    def add_technical_indicators(self, df):
        """
        拡張版テクニカル指標追加 (自前実装を使用)
        
        Args:
            df (pandas.DataFrame): OHLCVデータ
            
        Returns:
            pandas.DataFrame: テクニカル指標追加後のデータ
        """
        try:
            if df.empty:
                self.logger.warning("入力データフレームが空のため、テクニカル指標を追加できません。")
                return df

            # configからパラメータを取得
            strategy_params = config.get('strategy_params', {})
            risk_params = config.get('risk_params', {})
            
            # SMA期間の取得
            sma_short_period = strategy_params.get('sma_short_period', 50)
            sma_medium_period = strategy_params.get('sma_medium_period', 100)
            sma_long_period = strategy_params.get('sma_long_period', 200)
            
            # EMA期間の取得
            ema_very_short_period = strategy_params.get('ema_very_short_period', 9)
            ema_short_period = strategy_params.get('ema_short_period', 20)
            ema_medium_period = strategy_params.get('ema_medium_period', 50)
            ema_long_period = strategy_params.get('ema_long_period', 200)
            
            # RSI期間の取得
            rsi_period = strategy_params.get('rsi_period_trade', 14)
            rsi_short_period = strategy_params.get('rsi_short_period', 7)
            
            # STOCH パラメータの取得
            stoch_k_period = strategy_params.get('stoch_k_period', 14)
            stoch_d_period = strategy_params.get('stoch_d_period', 3)
            
            # MACD パラメータの取得
            macd_fast_period = strategy_params.get('macd_fast_period', 12)
            macd_slow_period = strategy_params.get('macd_slow_period', 26)
            macd_signal_period = strategy_params.get('macd_signal_period', 9)
            
            # BB パラメータの取得
            bb_period = strategy_params.get('bb_period', 20)
            bb_std_dev = strategy_params.get('bb_std_dev', 2)
            
            # ATR 期間の取得
            atr_period = risk_params.get('atr_period_for_risk', 14)
            
            # ADX 期間の取得
            adx_period = strategy_params.get('adx_period_trade', 14)
            
            # Volume SMA 期間の取得
            volume_sma_period = strategy_params.get('volume_sma_period_trade', 20)

            # 各種テクニカル指標を計算（動的なパラメータを使用）
            
            # SMA系列
            df[f'sma_{sma_short_period}'] = self._sma(df['close'], sma_short_period)
            df[f'sma_{sma_medium_period}'] = self._sma(df['close'], sma_medium_period)
            df[f'sma_{sma_long_period}'] = self._sma(df['close'], sma_long_period)
            # バックアップ/後方互換性のための固定名
            df['sma_20'] = self._sma(df['close'], 20)
            df['sma_50'] = self._sma(df['close'], 50)
            df['sma_100'] = self._sma(df['close'], 100)
            df['sma_200'] = self._sma(df['close'], 200)
            
            # EMA系列
            df[f'ema_{ema_very_short_period}'] = self._ema(df['close'], ema_very_short_period)
            df[f'ema_{ema_short_period}'] = self._ema(df['close'], ema_short_period)
            df[f'ema_{ema_medium_period}'] = self._ema(df['close'], ema_medium_period)
            df[f'ema_{ema_long_period}'] = self._ema(df['close'], ema_long_period)
            # バックアップ/後方互換性のための固定名
            df['ema_9'] = self._ema(df['close'], 9)
            df['ema_20'] = self._ema(df['close'], 20)
            df['ema_50'] = self._ema(df['close'], 50)
            df['ema_200'] = self._ema(df['close'], 200)
            
            # RSI系列
            df[f'rsi_{rsi_period}'] = self._rsi(df['close'], rsi_period)
            df[f'rsi_{rsi_short_period}'] = self._rsi(df['close'], rsi_short_period)
            # バックアップ/後方互換性のための固定名
            df['rsi_14'] = self._rsi(df['close'], 14)
            df['rsi_7'] = self._rsi(df['close'], 7)
            
            # ストキャスティクス
            stoch = self._stochastic(df, stoch_k_period, stoch_d_period)
            df['stoch_k'] = stoch['k']
            df['stoch_d'] = stoch['d']
            
            # MACDの計算
            macd_df = self._macd(df['close'], fast=macd_fast_period, slow=macd_slow_period, signal=macd_signal_period)
            df['macd'] = macd_df['macd']
            df['macd_signal'] = macd_df['signal']
            df['macd_hist'] = macd_df['hist']
            
            # ボリンジャーバンド
            bbands_df = self._bbands(df['close'], bb_period, stdev=bb_std_dev)
            df['bb_upper'] = bbands_df['upper']
            df['bb_middle'] = bbands_df['middle']
            df['bb_lower'] = bbands_df['lower']
            df['bb_width'] = (bbands_df['upper'] - bbands_df['lower']) / (bbands_df['middle'] + 1e-10) # ゼロ除算防止
            
            # ATR（Average True Range）
            df[f'atr_{atr_period}'] = self._atr(df, atr_period)
            
            # ADX（トレンド強度）
            adx_df = self._adx(df, adx_period)
            df['adx'] = adx_df['adx']
            df['di_plus'] = adx_df['di_plus']
            df['di_minus'] = adx_df['di_minus']
            
            # 前日比（％）
            df['pct_change'] = df['close'].pct_change() * 100
            
            # ボリューム指標
            df[f'volume_sma_{volume_sma_period}'] = self._sma(df['volume'], volume_sma_period)
            df['volume_sma_20'] = self._sma(df['volume'], 20) # バックアップ用
            volume_sma_col = f"volume_sma_{volume_sma_period}"
            if volume_sma_col in df.columns and 'volume' in df.columns:
                df['volume_ratio'] = df['volume'] / (df[volume_sma_col] + 1e-10) # ゼロ除算防止
            else:
                # フォールバック
                if "volume_sma_20" in df.columns and 'volume' in df.columns:
                    df['volume_ratio'] = df['volume'] / (df["volume_sma_20"] + 1e-10) # ゼロ除算防止
                else:
                    df['volume_ratio'] = 0.0
            
            # 高値・安値からの乖離率
            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                df['high_low_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10) # ゼロ除算防止
            else:
                df['high_low_ratio'] = 0.0
            
            # 価格変動率（ボラティリティ） (標準偏差を利用)
            volatility_period = strategy_params.get('volatility_period_trade', 20)
            df['volatility_20'] = df['close'].pct_change().rolling(window=volatility_period).std() * 100
            
            self.logger.info(f"拡張版テクニカル指標を追加しました。列数: {len(df.columns)}")
            return df
        except Exception as e:
            self.logger.error(f"テクニカル指標追加エラー: {e}", exc_info=True)
            # エラーが発生した場合でも、元のdfに必要な列（OHLCV）があればそれを返すことを試みる
            # ただし、指標がないとバックテストが失敗する可能性が高い
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in essential_cols):
                self.logger.warning("指標追加に失敗しましたが、OHLCVデータは存在します。不完全なデータで続行します。")
                return df[essential_cols]
            raise

    def _sma(self, series, window):
        """単純移動平均"""
        return series.rolling(window=window).mean()
    
    def _ema(self, series, window):
        """指数移動平均"""
        return series.ewm(span=window, adjust=False).mean()
    
    def _rsi(self, series, window):
        """RSI（相対力指数）"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)  # ゼロ除算を避ける
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _macd(self, series, fast=12, slow=26, signal=9):
        """MACD（移動平均収束拡散）"""
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'hist': histogram
        })
    
    def _bbands(self, series, window, stdev=2):
        """ボリンジャーバンド"""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = sma + (std * stdev)
        lower_band = sma - (std * stdev)
        
        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        })
    
    def _stochastic(self, df, k_period=14, d_period=3):
        """ストキャスティクス・オシレーター"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # %K計算
        k = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
        
        # %D計算（%Kの移動平均）
        d = k.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'k': k,
            'd': d
        })
    
    def _atr(self, df, window=14):
        """Average True Range（ATR） - EMAベースに変更"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range計算
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR計算（TRの指数移動平均）
        atr = self._ema(tr, window) # SMAからEMAに変更
        
        return atr
    
    def _adx(self, df, period=14):
        """Average Directional Index（ADX） - 標準的な計算ロジックに修正"""
        high = df['high']
        low = df['low']
        close = df['close']

        # +DM, -DM の計算
        up_move = high.diff()
        down_move = low.diff()

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
        
        # ATR (EMAベース)
        # Smooth +DM, Smooth -DM, Smooth TR
        # Wilder's Smoothing (RMA) を使うのが一般的だが、ここではEMAで代用
        smooth_plus_dm = self._ema(plus_dm, period)
        smooth_minus_dm = self._ema(minus_dm, period)
        
        # TRの計算 (ATRとは別にADX用にTRを計算)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        smooth_true_range = self._ema(true_range, period)

        # +DI, -DI
        # smooth_true_range が0の場合のゼロ除算を避ける
        plus_di = 100 * (smooth_plus_dm / (smooth_true_range + 1e-10))
        minus_di = 100 * (smooth_minus_dm / (smooth_true_range + 1e-10))
        
        # DX
        # (plus_di + minus_di) が0の場合のゼロ除算を避ける
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        
        # ADX
        adx = self._ema(dx, period)
        
        return pd.DataFrame({
            'adx': adx,
            'di_plus': plus_di,
            'di_minus': minus_di
        }) 