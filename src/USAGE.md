# 自動売買システム 使用方法

## 1. 環境設定

1. 必要なパッケージをインストール

```
pip install -r requirements.txt
```

2. `.env`ファイルの作成
   以下の内容で`.env`ファイルを作成し、API キーなどを設定してください。

```
# 取引所API設定
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BITFLYER_API_KEY=your_bitflyer_api_key
BITFLYER_SECRET_KEY=your_bitflyer_secret_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# 取引設定
MAX_POSITION_SIZE=0.01  # BTC最大ポジションサイズ
RISK_PER_TRADE=0.01  # 1取引あたりのリスク（資金の割合）
STOP_LOSS_PERCENTAGE=0.01  # 損切り率
TAKE_PROFIT_PERCENTAGE=0.02  # 利確率

# ニュース設定
NEWS_API_KEY=your_news_api_key
```

## 2. 実行モード

### バックテストモード

過去のデータを使用して戦略の性能をテストします。
模擬データは絶対に使用してはならない。

```
python src/main.py --mode backtest --symbol BTC/USDT --exchange binance --timeframe 1h --capital 100000 --days 30
```

パラメータ:

- `--symbol`: 取引通貨ペア（例: BTC/USDT, ETH/BTC）
- `--exchange`: 取引所（例: binance, bitflyer）
- `--timeframe`: 時間枠（例: 1m, 5m, 1h, 1d）
- `--capital`: 初期資本金
- `--days`: バックテスト期間（日数）

```
python src/main.py --mode paper --symbol BTC/USDT --exchange binance --timeframe 1h
```

### ライブトレードモード

実際の取引所 API を使用して自動売買を行います。

```
python src/main.py --mode live --symbol BTC/USDT --exchange binance --timeframe 1h
```

## 3. 戦略のカスタマイズ

戦略の調整は以下のファイルで行います：

- `src/strategies/combined_strategy.py`: 複合戦略の実装
- `src/config/config.py`: 設定パラメータ

## 4. 結果の確認

- バックテスト結果: `backtest_results.csv`
- ペーパートレード結果: `paper_trade_history.csv`
- ライブトレード結果: `trade_history.csv`
- ログ: `auto_trade.log`

## 5. 注意事項

- ライブトレードは実際の資金を使用します。十分にテストしてから使用してください。
- 仮想通貨取引は大きなリスクを伴います。自己責任で使用してください。
- API キーの管理には十分注意してください。取引権限のみ付与し、出金権限は付与しないことをおすすめします。
