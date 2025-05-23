#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
バックテスト結果可視化モジュール
Plotlyを使用してインタラクティブなHTMLレポートを生成します
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from plotly.utils import PlotlyJSONEncoder

# 親ディレクトリをパスに追加（インポート用）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BacktestVisualizer:
    """バックテスト結果の可視化クラス"""
    
    def __init__(self, results_df=None, trades_df=None, results_dir='results'):
        """
        初期化
        
        Args:
            results_df (pandas.DataFrame, optional): 結果データフレーム
            trades_df (pandas.DataFrame, optional): 取引データフレーム
            results_dir (str): 結果ファイルディレクトリのパス
        """
        # プロジェクトルートからのパス解決
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.results_dir = os.path.join(self.project_root, results_dir)
        self.html_dir = os.path.join(self.project_root, 'html_reports')
        
        # ディレクトリがなければ作成
        os.makedirs(self.html_dir, exist_ok=True)
        
        # データフレーム初期化
        self.results_df = results_df
        self.trades_df = trades_df
        self.metrics = {}
        
        # 結果と取引データが直接渡された場合は前処理
        if self.results_df is not None:
            # NaN値を削除または置換
            self.results_df = self._clean_data(self.results_df)
            
            # 日付列をdatetime型に変換
            if 'date' in self.results_df.columns and not pd.api.types.is_datetime64_any_dtype(self.results_df['date']):
                self.results_df['date'] = pd.to_datetime(self.results_df['date'])
                # 日付をインデックスに設定
                self.results_df.set_index('date', inplace=True)
        
        if self.trades_df is not None:
            # NaN値を削除または置換
            self.trades_df = self._clean_data(self.trades_df)
            
            # 日付列をdatetime型に変換
            if 'entry_time' in self.trades_df.columns and not pd.api.types.is_datetime64_any_dtype(self.trades_df['entry_time']):
                self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
            if 'exit_time' in self.trades_df.columns and not pd.api.types.is_datetime64_any_dtype(self.trades_df['exit_time']):
                self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
        
        # テーマ設定
        self.theme = {
            'background_color': '#f8f9fa',
            'paper_bgcolor': '#ffffff',
            'plot_bgcolor': '#f8f9fa',
            'font_family': "'Helvetica Neue', 'Meiryo', 'MS Gothic', sans-serif",
            'font_color': '#333333',
            'grid_color': '#dddddd',
            'primary_color': '#1f77b4',
            'profit_color': '#2ca02c',
            'loss_color': '#d62728',
            'accent_color': '#ff7f0e',
            'colorscale': 'RdYlGn',
            'table_header_color': '#e9ecef',
            'table_cell_color': '#f8f9fa',
        }
        
    def _clean_data(self, df):
        """
        データフレームのクリーニング - NaNやInfinity値を処理
        
        Args:
            df (pandas.DataFrame): クリーニング対象のデータフレーム
            
        Returns:
            pandas.DataFrame: クリーニング済みデータフレーム
        """
        # コピーを作成して元のデータを変更しない
        cleaned_df = df.copy()
        
        # 数値型の列のみを処理
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 無限大の値を置換
            if col in cleaned_df:
                # inf を大きな値に置換
                cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], [1e10, -1e10])
                
                # NaN値を0に置換
                cleaned_df[col] = cleaned_df[col].fillna(0)
        
        return cleaned_df

    def _clean_figure_data(self, fig):
        """
        Plotlyの図オブジェクトから無効なデータ（NaN, Infinity）をクリーニング
        
        Args:
            fig (plotly.graph_objects.Figure): クリーニング対象の図
            
        Returns:
            plotly.graph_objects.Figure: クリーニング済みの図
        """
        if fig is None:
            return None
            
        try:
            # データのコピーを作成
            for trace_idx, trace in enumerate(fig.data):
                # X, Y データの処理
                if hasattr(trace, 'x') and trace.x is not None:
                    x_array = np.array(trace.x)
                    # NaN と Infinity を置換
                    mask_x = np.isfinite(x_array)
                    if not np.all(mask_x):
                        # 無効な値を含む場合は、その値を除外または置換
                        valid_x = x_array[mask_x]
                        if len(valid_x) > 0:
                            if isinstance(trace.x, list):
                                for i in range(len(trace.x)):
                                    if not np.isfinite(trace.x[i]):
                                        trace.x[i] = 0
                
                if hasattr(trace, 'y') and trace.y is not None:
                    y_array = np.array(trace.y)
                    # NaN と Infinity を置換
                    mask_y = np.isfinite(y_array)
                    if not np.all(mask_y):
                        # 無効な値を含む場合は、その値を除外または置換
                        valid_y = y_array[mask_y]
                        if len(valid_y) > 0:
                            if isinstance(trace.y, list):
                                for i in range(len(trace.y)):
                                    if not np.isfinite(trace.y[i]):
                                        trace.y[i] = 0
                
                # Z データ（ヒートマップなど）の処理
                if hasattr(trace, 'z') and trace.z is not None:
                    z_array = np.array(trace.z)
                    if z_array.ndim == 2:  # 2次元配列の場合
                        for i in range(z_array.shape[0]):
                            for j in range(z_array.shape[1]):
                                if not np.isfinite(z_array[i, j]):
                                    z_array[i, j] = 0
                        fig.data[trace_idx].z = z_array
            
            return fig
        except Exception as e:
            print(f"図データのクリーニング中にエラー: {e}")
            return fig

    def _safe_json_serialize(self, fig):
        """
        安全にJSON変換できるように図データをシリアライズ
        
        Args:
            fig (plotly.graph_objects.Figure): 変換対象の図
            
        Returns:
            str: JSON文字列
        """
        try:
            # 最初にクリーニング
            cleaned_fig = self._clean_figure_data(fig)
            
            # PlotlyJSONEncoderを使用してJSON文字列に変換
            json_str = json.dumps(cleaned_fig, cls=PlotlyJSONEncoder)
            
            return json_str
        except Exception as e:
            print(f"JSON変換中にエラー: {e}")
            # エラーの場合は空の図を返す
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="データのエラーにより表示できません",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return json.dumps(empty_fig, cls=PlotlyJSONEncoder)
        
    def load_data(self, results_file='advanced_backtest_results.csv', trades_file='advanced_backtest_trades.csv'):
        """
        バックテスト結果データ読み込み
        
        Args:
            results_file (str): 結果CSVファイル名
            trades_file (str): 取引履歴CSVファイル名
            
        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            # 結果データ読み込み
            results_path = os.path.join(self.results_dir, results_file)
            self.results_df = pd.read_csv(results_path)
            
            # NaN値や異常値の処理
            self.results_df = self._clean_data(self.results_df)
            
            # 日付列をdatetime型に変換
            if 'date' in self.results_df.columns:
                self.results_df['date'] = pd.to_datetime(self.results_df['date'])
                # 日付をインデックスに設定
                self.results_df.set_index('date', inplace=True)
            
            # 取引履歴データ読み込み
            trades_path = os.path.join(self.results_dir, trades_file)
            self.trades_df = pd.read_csv(trades_path)
            
            # NaN値や異常値の処理
            self.trades_df = self._clean_data(self.trades_df)
            
            # 日付列をdatetime型に変換
            if 'entry_time' in self.trades_df.columns:
                self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
            if 'exit_time' in self.trades_df.columns:
                self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
            
            print(f"バックテストデータ読み込み完了: {len(self.results_df)}行の結果、{len(self.trades_df)}件の取引")
            return True
            
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return False

    def _create_layout(self, title, height=None, showlegend=True, legend_pos='top'):
        """共通のレイアウト設定を作成"""
        legend_dict = {}
        if legend_pos == 'top':
            legend_dict = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        
        return dict(
            title=dict(
                text=title,
                font=dict(
                    family=self.theme['font_family'],
                    size=24,
                    color=self.theme['font_color']
                ),
                x=0.5,
                xanchor='center'
            ),
            font=dict(
                family=self.theme['font_family'],
                color=self.theme['font_color']
            ),
            paper_bgcolor=self.theme['paper_bgcolor'],
            plot_bgcolor=self.theme['plot_bgcolor'],
            height=height,
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=showlegend,
            legend=legend_dict,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_family=self.theme['font_family']
            ),
            dragmode='zoom',
            selectdirection='h'
        )

    def _calc_returns_and_drawdowns(self):
        """リターンとドローダウンを計算"""
        if 'returns' not in self.results_df.columns:
            # 日次リターンを計算
            self.results_df['returns'] = self.results_df['equity'].pct_change() * 100
        
        if 'drawdown' not in self.results_df.columns:
            # ドローダウンを計算
            running_max = self.results_df['equity'].cummax()
            self.results_df['drawdown'] = (self.results_df['equity'] / running_max - 1) * 100

    def _calculate_metrics(self):
        """メトリクスの計算"""
        if self.results_df is None or self.results_df.empty:
            return
            
        try:
            # 基本メトリクス
            if 'total_return' in self.results_df.columns:
                self.metrics['total_return'] = self.results_df['total_return'].iloc[-1]
            elif 'return' in self.results_df.columns:
                self.metrics['total_return'] = self.results_df['return'].iloc[-1]
            else:
                # 資産の最終値と初期値から計算
                self.metrics['total_return'] = (self.results_df['equity'].iloc[-1] / self.results_df['equity'].iloc[0] - 1) * 100
            
            # 年率リターン
            if isinstance(self.results_df.index, pd.DatetimeIndex):
                # 取引期間（年）
                days = (self.results_df.index[-1] - self.results_df.index[0]).days
                years = days / 365.25
                if years > 0:
                    self.metrics['annual_return'] = ((1 + self.metrics['total_return']/100) ** (1/years) - 1) * 100
                else:
                    self.metrics['annual_return'] = 0
                
            # 勝率
            if not self.trades_df.empty:
                win_trades = len(self.trades_df[self.trades_df['pnl'] > 0])
                total_trades = len(self.trades_df)
                self.metrics['win_rate'] = (win_trades / total_trades * 100) if total_trades > 0 else 0
                
                # 平均保有期間
                if 'holding_bars' in self.trades_df.columns:
                    self.metrics['avg_hold_days'] = self.trades_df['holding_bars'].mean()
                elif 'entry_time' in self.trades_df.columns and 'exit_time' in self.trades_df.columns:
                    self.trades_df['holding_days'] = (self.trades_df['exit_time'] - self.trades_df['entry_time']).dt.total_seconds() / (24 * 3600)
                    self.metrics['avg_hold_days'] = self.trades_df['holding_days'].mean()
                    
                # 損益率
                self.metrics['profit_factor'] = abs(self.trades_df[self.trades_df['pnl'] > 0]['pnl'].sum() / 
                                            self.trades_df[self.trades_df['pnl'] < 0]['pnl'].sum()) if self.trades_df[self.trades_df['pnl'] < 0]['pnl'].sum() != 0 else 0
                
                # 平均利益と平均損失
                self.metrics['avg_profit'] = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] > 0]) > 0 else 0
                self.metrics['avg_loss'] = self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] < 0]) > 0 else 0
                
                # 最大連勝・連敗
                self.trades_df['win'] = self.trades_df['pnl'] > 0
                streak = (self.trades_df['win'] != self.trades_df['win'].shift(1)).cumsum()
                win_streaks = self.trades_df[self.trades_df['win']].groupby(streak).size()
                loss_streaks = self.trades_df[~self.trades_df['win']].groupby(streak).size()
                self.metrics['max_win_streak'] = win_streaks.max() if not win_streaks.empty else 0
                self.metrics['max_loss_streak'] = loss_streaks.max() if not loss_streaks.empty else 0
                
            # 最大ドローダウン
            self._calc_returns_and_drawdowns()
            self.metrics['max_drawdown'] = self.results_df['drawdown'].min()
            
            # シャープレシオ（年率）
            if 'returns' in self.results_df.columns and len(self.results_df) > 1:
                daily_return_std = self.results_df['returns'].std()
                if daily_return_std > 0:
                    # 年率シャープレシオ（無リスク金利を0と仮定）
                    self.metrics['sharpe_ratio'] = (self.metrics.get('annual_return', 0) / (daily_return_std * np.sqrt(252)))
                else:
                    self.metrics['sharpe_ratio'] = 0
                
        except Exception as e:
            print(f"メトリクス計算エラー: {e}")
            
    def _create_equity_curve(self):
        """資産推移グラフの作成"""
        if self.results_df is None or len(self.results_df) == 0:
            return None
        
        # リターンとドローダウンを計算
        self._calc_returns_and_drawdowns()
        
        # サブプロット作成（3行1列）
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            subplot_titles=('資産推移', '日次リターン (%)', 'ドローダウン (%)'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # 資産推移
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index, 
                y=self.results_df['equity'], 
                mode='lines', 
                name='資産推移',
                line=dict(color=self.theme['primary_color'], width=2),
                hovertemplate='日付: %{x}<br>資産: ¥%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 日次リターン
        fig.add_trace(
            go.Bar(
                x=self.results_df.index, 
                y=self.results_df['returns'], 
                name='日次リターン',
                marker=dict(
                    color=self.results_df['returns'].apply(
                        lambda x: self.theme['profit_color'] if x > 0 else self.theme['loss_color']
                    )
                ),
                hovertemplate='日付: %{x}<br>リターン: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # ドローダウン
        fig.add_trace(
            go.Scatter(
                x=self.results_df.index, 
                y=self.results_df['drawdown'], 
                mode='lines', 
                name='ドローダウン',
                line=dict(color=self.theme['loss_color'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba({','.join(str(int(x)) for x in px.colors.hex_to_rgb(self.theme['loss_color']))}, 0.3)",
                hovertemplate='日付: %{x}<br>ドローダウン: %{y:.2f}%<extra></extra>'
            ),
            row=3, col=1
        )

        # X軸の日付フォーマット設定
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
            tickangle=-45,
            gridcolor=self.theme['grid_color'],
            row=3, col=1
        )
        
        # Y軸の設定
        fig.update_yaxes(
            gridcolor=self.theme['grid_color'],
            tickformat=",.0f",
            row=1, col=1
        )
        fig.update_yaxes(
            gridcolor=self.theme['grid_color'],
            tickformat=".2f",
            row=2, col=1
        )
        fig.update_yaxes(
            gridcolor=self.theme['grid_color'],
            tickformat=".2f",
            row=3, col=1
        )
        
        # レイアウト調整
        fig.update_layout(
            **self._create_layout('バックテスト資産推移', height=800)
        )
        
        return fig
    
    def _create_trade_analysis(self):
        """取引分析グラフを作成"""
        if self.trades_df.empty:
            # 取引データが空の場合は空のグラフを返す
            return None, None, None
            
        # 勝ち負けの円グラフ
        win_count = len(self.trades_df[self.trades_df['pnl'] > 0])
        loss_count = len(self.trades_df[self.trades_df['pnl'] <= 0])
        
        # データが0件の場合のチェック
        if win_count == 0 and loss_count == 0:
            # ダミーデータで空のグラフを作成
            pie_fig = go.Figure(go.Pie(labels=['データなし'], values=[1]))
            pie_fig.update_layout(**self._create_layout('取引タイプ (データなし)', height=300))
            hist_fig = go.Figure()
            hist_fig.update_layout(**self._create_layout('損益分布 (データなし)', height=300))
            scatter_fig = go.Figure()
            scatter_fig.update_layout(**self._create_layout('保有期間別損益 (データなし)', height=300))
            return pie_fig, hist_fig, scatter_fig
        
        # 円グラフデータ
        if 'type' in self.trades_df.columns:
            # 買い/売りで集計
            trade_counts = self.trades_df['type'].value_counts()
            pie_labels = trade_counts.index
            pie_values = trade_counts.values
            title = '取引タイプ別集計'
            colors = [self.theme['primary_color'], self.theme['loss_color']]
        else:
            # 勝ち/負けで集計
            pie_labels = ['勝ち', '負け']
            pie_values = [win_count, loss_count]
            title = '取引結果 (勝ち/負け)'
            colors = [self.theme['profit_color'], self.theme['loss_color']]
        
        # 円グラフの作成
        pie_fig = go.Figure()
        pie_fig.add_trace(go.Pie(
            labels=pie_labels, 
            values=pie_values,
            hole=.4,
            textinfo='label+percent',
            marker=dict(colors=colors),
            hovertemplate='%{label}: %{value}件<br>割合: %{percent}<extra></extra>'
        ))
        
        # 取引件数と勝率を追加
        win_rate = win_count / (win_count + loss_count) * 100 if (win_count + loss_count) > 0 else 0
        pie_fig.add_annotation(
            text=f"総取引数: {win_count + loss_count}件<br>勝率: {win_rate:.1f}%",
            x=0.5, y=0.5,
            font=dict(size=14, color=self.theme['font_color']),
            showarrow=False
        )
        
        pie_fig.update_layout(
            **self._create_layout(title, height=350)
        )
        
        # 損益分布ヒストグラム
        bin_size = max(1, int(len(self.trades_df) / 20))  # ビンサイズの動的計算
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=self.trades_df['pnl'],
            nbinsx=bin_size,
            marker=dict(
                color=self.theme['profit_color'],
                line=dict(color='white', width=1)
            ),
            opacity=0.75,
            name='利益',
            hovertemplate='利益: %{x:.2f}<br>頻度: %{y}件<extra></extra>'
        ))
        
        # 損益0の垂直線
        hist_fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dash")
        )
        
        # 平均利益と平均損失の垂直線
        avg_profit = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] < 0]) > 0 else 0
        
        if avg_profit > 0:
            hist_fig.add_shape(
                type="line",
                x0=avg_profit, y0=0, x1=avg_profit, y1=1,
                yref="paper",
                line=dict(color=self.theme['profit_color'], width=2)
            )
            hist_fig.add_annotation(
                x=avg_profit, y=1,
                text=f"平均利益: {avg_profit:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=40, ay=-40,
                font=dict(color=self.theme['profit_color'])
            )
        
        if avg_loss < 0:
            hist_fig.add_shape(
                type="line",
                x0=avg_loss, y0=0, x1=avg_loss, y1=1,
                yref="paper",
                line=dict(color=self.theme['loss_color'], width=2)
            )
            hist_fig.add_annotation(
                x=avg_loss, y=1,
                text=f"平均損失: {avg_loss:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=-40, ay=-40,
                font=dict(color=self.theme['loss_color'])
            )
        
        hist_fig.update_layout(
            **self._create_layout('損益分布', height=400),
            xaxis=dict(
                title='損益',
                tickformat='.2f',
                gridcolor=self.theme['grid_color']
            ),
            yaxis=dict(
                title='頻度',
                gridcolor=self.theme['grid_color']
            ),
            bargap=0.1
        )
        
        # 保有期間と損益の散布図
        if 'holding_bars' in self.trades_df.columns:
            hold_col = 'holding_bars'
            xaxis_title = '保有期間 (バー数)'
        elif 'entry_time' in self.trades_df.columns and 'exit_time' in self.trades_df.columns:
            self.trades_df['holding_days'] = (self.trades_df['exit_time'] - self.trades_df['entry_time']).dt.total_seconds() / (24 * 3600)
            hold_col = 'holding_days'
            xaxis_title = '保有期間 (日)'
        else:
            # 保有期間データがない場合
            scatter_fig = go.Figure()
            scatter_fig.update_layout(**self._create_layout('保有期間別損益 (データなし)', height=400))
            return pie_fig, hist_fig, scatter_fig
            
        scatter_fig = go.Figure()
        
        # Long/Shortで色分け
        if 'type' in self.trades_df.columns:
            for trade_type, color in zip(['buy', 'sell'], [self.theme['primary_color'], self.theme['loss_color']]):
                df_filtered = self.trades_df[self.trades_df['type'] == trade_type]
                if not df_filtered.empty:
                    scatter_fig.add_trace(go.Scatter(
                        x=df_filtered[hold_col],
                        y=df_filtered['pnl'],
                        mode='markers',
                        name=trade_type.capitalize(),
                        marker=dict(
                            size=10,
                            color=color,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'保有期間: %{{x:.1f}}<br>損益: %{{y:.2f}}<br>取引タイプ: {trade_type}<extra></extra>'
                    ))
        else:
            # 損益で色分け
            scatter_fig.add_trace(go.Scatter(
                x=self.trades_df[hold_col],
                y=self.trades_df['pnl'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.trades_df['pnl'],
                    colorscale=self.theme['colorscale'],
                    colorbar=dict(title='損益'),
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='保有期間: %{x:.1f}<br>損益: %{y:.2f}<extra></extra>',
                name='取引'
            ))
        
        # トレンドライン
        if len(self.trades_df) >= 5:  # 最低5件あれば傾向線を表示
            z = np.polyfit(self.trades_df[hold_col], self.trades_df['pnl'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(self.trades_df[hold_col].min(), self.trades_df[hold_col].max(), 100)
            
            scatter_fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='傾向線',
                hoverinfo='skip'
            ))
        
        scatter_fig.update_layout(
            **self._create_layout('保有期間別損益', height=450),
            xaxis=dict(
                title=xaxis_title,
                gridcolor=self.theme['grid_color']
            ),
            yaxis=dict(
                title='損益',
                tickformat='.2f',
                gridcolor=self.theme['grid_color'],
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=1
            )
        )
        
        return pie_fig, hist_fig, scatter_fig
    
    def _create_monthly_performance(self):
        """月次パフォーマンスヒートマップを作成"""
        if self.results_df.empty:
            # データが空の場合
            monthly_fig = go.Figure()
            monthly_fig.update_layout(**self._create_layout('月次パフォーマンス (データなし)', height=300))
            return monthly_fig
            
        try:
            # 日付列の処理
            # インデックスが日付型の場合はそのまま使用
            if isinstance(self.results_df.index, pd.DatetimeIndex):
                # 日次リターンを計算
                if 'returns' not in self.results_df.columns:
                    self.results_df['returns'] = self.results_df['equity'].pct_change() * 100
                
                # 年と月を抽出
                years = [d.year for d in self.results_df.index]
                months = [d.month for d in self.results_df.index]
                
                # 月次リターンを計算
                monthly_returns = self.results_df.groupby([pd.Series(years), pd.Series(months)])['returns'].sum().unstack()
                
                # 月名に変換
                month_names = {
                    1: '1月', 2: '2月', 3: '3月', 4: '4月', 5: '5月', 6: '6月',
                    7: '7月', 8: '8月', 9: '9月', 10: '10月', 11: '11月', 12: '12月'
                }
                
                # 列の並び替え（1月から12月）
                if not monthly_returns.empty:
                    monthly_returns = monthly_returns.reindex(columns=sorted(monthly_returns.columns))
                    monthly_returns.columns = [month_names[m] for m in monthly_returns.columns]
                    
                    # 年次合計の追加
                    monthly_returns['年間'] = monthly_returns.sum(axis=1)
                    
                    # ヒートマップ用のデータ準備
                    z_data = monthly_returns.values
                    x_data = monthly_returns.columns
                    y_data = monthly_returns.index
                    
                    # 月次リターン評価の色スケール設定
                    max_abs_val = max(abs(np.nanmin(z_data)), abs(np.nanmax(z_data)))
                    max_range = max(max_abs_val, 5)  # 最小でも±5%のレンジ
                    
                    # ヒートマップ作成
                    fig = go.Figure(data=go.Heatmap(
                        z=z_data,
                        x=x_data,
                        y=y_data,
                        colorscale=self.theme['colorscale'],
                        zmid=0,  # 0をスケールの中央に
                        zmin=-max_range,
                        zmax=max_range,
                        colorbar=dict(
                            title='リターン(%)',
                            titleside='right',
                            titlefont=dict(size=14),
                            tickformat='.1f'
                        ),
                        hoverongaps=False,
                        hovertemplate='%{y}年 %{x}<br>リターン: %{z:.2f}%<extra></extra>',
                        text=[[f"{val:.2f}%" if not np.isnan(val) else "" for val in row] for row in z_data],
                        texttemplate="%{text}",
                        textfont=dict(color="black")
                    ))
                    
                    # 年間合計列の強調
                    fig.update_layout(
                        **self._create_layout('月次・年次パフォーマンス (%)', height=400),
                        xaxis=dict(
                            side='top',
                            tickangle=-30
                        ),
                        yaxis=dict(
                            autorange='reversed'  # 新しい年を上に表示
                        )
                    )
                    
                    return fig
                else:
                    # データがない場合
                    monthly_fig = go.Figure()
                    monthly_fig.update_layout(**self._create_layout('月次パフォーマンス (データ不足)', height=300))
                    return monthly_fig
            else:
                # 日付列がインデックスでない場合
                monthly_fig = go.Figure()
                monthly_fig.update_layout(**self._create_layout('月次パフォーマンス (日付データなし)', height=300))
                return monthly_fig
                
        except Exception as e:
            print(f"月次パフォーマンス計算エラー: {e}")
            monthly_fig = go.Figure()
            monthly_fig.update_layout(**self._create_layout('月次パフォーマンス (計算エラー)', height=300))
            return monthly_fig
    
    def _create_metrics_table(self):
        """メトリクス表を作成"""
        # メトリクスが空の場合は計算
        if not self.metrics:
            self._calculate_metrics()
            
        # 主要メトリクスを表示
        key_metrics = [
            {'メトリクス': '総リターン (%)', '値': self.metrics.get('total_return', 0), 'カテゴリ': 'リターン'},
            {'メトリクス': '年率リターン (%)', '値': self.metrics.get('annual_return', 0), 'カテゴリ': 'リターン'},
            {'メトリクス': 'シャープレシオ', '値': self.metrics.get('sharpe_ratio', 0), 'カテゴリ': 'リスク'},
            {'メトリクス': '最大ドローダウン (%)', '値': self.metrics.get('max_drawdown', 0), 'カテゴリ': 'リスク'},
            {'メトリクス': '勝率 (%)', '値': self.metrics.get('win_rate', 0), 'カテゴリ': '取引'},
            {'メトリクス': '損益率', '値': self.metrics.get('profit_factor', 0), 'カテゴリ': '取引'},
            {'メトリクス': '平均利益', '値': self.metrics.get('avg_profit', 0), 'カテゴリ': '取引'},
            {'メトリクス': '平均損失', '値': self.metrics.get('avg_loss', 0), 'カテゴリ': '取引'},
            {'メトリクス': '平均保有日数', '値': self.metrics.get('avg_hold_days', 0), 'カテゴリ': '取引'},
            {'メトリクス': '最大連勝', '値': self.metrics.get('max_win_streak', 0), 'カテゴリ': '取引'},
            {'メトリクス': '最大連敗', '値': self.metrics.get('max_loss_streak', 0), 'カテゴリ': '取引'},
            {'メトリクス': '取引回数', '値': len(self.trades_df) if self.trades_df is not None else 0, 'カテゴリ': '取引'}
        ]
        
        # データフレームに変換
        metrics_df = pd.DataFrame(key_metrics)
        
        # グループごとに色を設定
        category_colors = {
            'リターン': self.theme['primary_color'],
            'リスク': self.theme['loss_color'],
            '取引': self.theme['accent_color']
        }
        
        # 値のフォーマット
        def format_value(row):
            val = row['値']
            metric = row['メトリクス']
            
            if '(%)' in metric:
                return f"{val:.2f}%"
            elif metric in ['シャープレシオ', '損益率']:
                return f"{val:.2f}"
            elif metric in ['平均利益', '平均損失']:
                return f"{val:.2f}"
            elif metric == '取引回数':
                return f"{int(val):,}回"
            elif metric == '平均保有日数':
                return f"{val:.1f}日"
            else:
                return f"{val:.2f}"
        
        metrics_df['表示値'] = metrics_df.apply(format_value, axis=1)
        
        # カテゴリごとの色を設定
        cell_colors = []
        for cat in metrics_df['カテゴリ']:
            color = category_colors.get(cat, 'white')
            # 薄い色にする
            rgb = px.colors.hex_to_rgb(color)
            rgba = [rgb[0], rgb[1], rgb[2], 0.2]
            cell_colors.append(f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})")
        
        # テーブル作成
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>カテゴリ</b>', '<b>メトリクス</b>', '<b>値</b>'],
                fill_color=self.theme['table_header_color'],
                align='left',
                font=dict(size=14, color=self.theme['font_color'])
            ),
            cells=dict(
                values=[metrics_df['カテゴリ'], metrics_df['メトリクス'], metrics_df['表示値']],
                fill_color=[cell_colors, 'white', 'white'],
                align='left',
                font=dict(size=13, color=self.theme['font_color']),
                height=30
            ),
            columnwidth=[0.25, 0.45, 0.3]
        )])
        
        layout = self._create_layout('バックテスト主要指標', height=430)
        # _create_layoutに既にmarginが含まれているので、ここでは追加しない
        fig.update_layout(**layout)
        
        return fig
    
    def generate_html_report(self, output_filename=None, backtest_params=None):
        """
        バックテスト結果をHTMLレポートとして生成
        
        Args:
            output_filename (str): 出力ファイル名
            backtest_params (dict): バックテストパラメータ（symbol, timeframe, years, initial_capital）
            
        Returns:
            str: 生成されたHTMLファイルのパス
        """
        
        # メトリクスの計算
        self._calculate_metrics()
        
        # データがない場合はエラー
        if self.results_df is None or self.results_df.empty:
            print("データが読み込まれていません。先にload_dataを呼び出すか、結果データを直接渡してください。")
            return None
            
        # レポートに表示するグラフを作成
        equity_curve = self._create_equity_curve()
        trade_pie, trade_hist, trade_scatter = self._create_trade_analysis()
        monthly_fig = self._create_monthly_performance()
        metrics_table = self._create_metrics_table()
        
        # 各グラフをクリーニング
        equity_curve = self._clean_figure_data(equity_curve)
        trade_pie = self._clean_figure_data(trade_pie)
        trade_hist = self._clean_figure_data(trade_hist)
        trade_scatter = self._clean_figure_data(trade_scatter)
        monthly_fig = self._clean_figure_data(monthly_fig)
        metrics_table = self._clean_figure_data(metrics_table)
        
        # CSS スタイル
        css_styles = """
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: #fff;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        h2 {
            margin-top: 30px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .graph-container {
            margin-bottom: 20px;
            background: #fff;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .section {
            margin-bottom: 40px;
        }
        .info-box {
            background-color: #f8f9fa;
            border-left: 4px solid #17a2b8;
            padding: 10px 15px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #555;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #777;
            font-size: 0.9em;
        }
        .params-section {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .params-list {
            list-style-type: none;
            padding: 0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        .params-list li {
            padding: 5px 0;
        }
        """
        
        # HTMLレポートの構築
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>バックテスト結果レポート</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
            {css_styles}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>バックテスト結果レポート</h1>
                <p style="text-align: center;">レポート生成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        # バックテストパラメータセクションの追加
        if backtest_params:
            params_items = []
            for key, value in backtest_params.items():
                params_items.append(f"<li><strong>{key}:</strong> {value}</li>")
            
            html_content += f"""
            <div class="params-section">
                <h2>バックテスト設定</h2>
                <ul class="params-list">
                    {"".join(params_items)}
                </ul>
            </div>
            """
                
        # 資産推移グラフ
        if equity_curve:
            html_content += """
                <div id="equity" class="section">
                    <h2>資産推移</h2>
                    <div class="info-box">
                        資産の推移、日次リターン、ドローダウンを時系列で表示しています。グラフは拡大・縮小可能です。
                    </div>
                    <div class="graph-container">
                        <div id="equity-curve"></div>
                    </div>
                </div>
            """
        
        # 取引分析
        if trade_pie and trade_hist and trade_scatter:
            html_content += """
                <div id="trades" class="section">
                    <h2>取引分析</h2>
                    <div class="info-box">
                        取引タイプの分布、損益分布、保有期間と損益の関係を分析しています。各グラフは相互に関連しており、パターンを見つけるのに役立ちます。
                    </div>
                    
                    <div class="grid-2">
                        <div class="graph-container">
                            <h3>取引タイプ</h3>
                            <div id="trade-pie"></div>
                        </div>
                        
                        <div class="graph-container">
                            <h3>損益分布</h3>
                            <div id="trade-hist"></div>
                        </div>
                    </div>
                    
                    <div class="graph-container">
                        <h3>保有期間別損益</h3>
                        <div id="trade-scatter"></div>
                    </div>
                </div>
            """
        
        # 月次パフォーマンス
        if monthly_fig:
            html_content += """
                <div id="monthly" class="section">
                    <h2>月次パフォーマンス</h2>
                    <div class="info-box">
                        各月のパフォーマンスをヒートマップで表示しています。色が緑いほど良好なパフォーマンスを示します。
                        月ごとの傾向やシーズナリティを分析するのに役立ちます。
                    </div>
                    <div class="graph-container">
                        <div id="monthly-performance"></div>
                    </div>
                </div>
            """
        
        # メトリクステーブル
        if metrics_table:
            html_content += """
                <div id="metrics" class="section">
                    <h2>詳細指標</h2>
                    <div class="info-box">
                        バックテストの詳細な指標を表示しています。リターン、リスク、取引に関する様々な指標を確認できます。
                    </div>
                    <div class="graph-container">
                        <div id="metrics-table"></div>
                    </div>
                </div>
            """
        
        # フッター
        html_content += f"""
                <div class="footer">
                    Auto-Trade バックテストレポート &copy; {datetime.datetime.now().year}
                </div>
            </div>
            
            <script>
                // エラーキャッチ用のラッパー関数
                function safeNewPlot(divId, data, layout, config) {{
                    try {{
                        Plotly.newPlot(divId, data, layout, config);
                    }} catch (error) {{
                        console.error('プロット作成エラー (' + divId + '):', error);
                        // エラー時に簡易メッセージを表示
                        document.getElementById(divId).innerHTML = 
                            '<div style="text-align:center;padding:20px;color:#d62728">グラフの表示中にエラーが発生しました。</div>';
                    }}
                }}
        """
        
        # 各グラフのJSONデータを埋め込む
        if equity_curve:
            html_content += f"var equityCurveData = {self._safe_json_serialize(equity_curve)};\n"
            html_content += "safeNewPlot('equity-curve', equityCurveData.data, equityCurveData.layout, {responsive: true});\n"
        
        if trade_pie:
            html_content += f"var tradePieData = {self._safe_json_serialize(trade_pie)};\n"
            html_content += "safeNewPlot('trade-pie', tradePieData.data, tradePieData.layout, {responsive: true});\n"
        
        if trade_hist:
            html_content += f"var tradeHistData = {self._safe_json_serialize(trade_hist)};\n"
            html_content += "safeNewPlot('trade-hist', tradeHistData.data, tradeHistData.layout, {responsive: true});\n"
        
        if trade_scatter:
            html_content += f"var tradeScatterData = {self._safe_json_serialize(trade_scatter)};\n"
            html_content += "safeNewPlot('trade-scatter', tradeScatterData.data, tradeScatterData.layout, {responsive: true});\n"
        
        if monthly_fig:
            html_content += f"var monthlyPerfData = {self._safe_json_serialize(monthly_fig)};\n"
            html_content += "safeNewPlot('monthly-performance', monthlyPerfData.data, monthlyPerfData.layout, {responsive: true});\n"
        
        if metrics_table:
            html_content += f"var metricsTableData = {self._safe_json_serialize(metrics_table)};\n"
            html_content += "safeNewPlot('metrics-table', metricsTableData.data, metricsTableData.layout, {responsive: true});\n"
        
        html_content += """
                // レスポンシブ対応
                window.addEventListener('resize', function() {
                    const graphDivs = document.querySelectorAll('[id^="equity-"], [id^="trade-"], [id^="monthly-"], [id^="metrics-"]');
                    graphDivs.forEach(function(div) {
                        try {
                            Plotly.relayout(div, {
                                'width': div.offsetWidth,
                                'height': div.offsetHeight
                            });
                        } catch (error) {
                            console.error('リサイズエラー:', error);
                        }
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # HTMLファイル出力
        if output_filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'backtest_report_{timestamp}.html'
            
        output_path = os.path.join(self.html_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTMLレポートを生成しました: {output_path}")
        
        return output_path

    def plot_backtest_results(self, output_filename):
        """
        バックテスト結果をグラフとして画像に保存
        
        Args:
            output_filename (str): 出力ファイル名
            
        Returns:
            bool: 成功したかどうか
        """
        if self.results_df is None or self.results_df.empty:
            print("データが読み込まれていません。先にload_dataを呼び出すか、結果データを直接渡してください。")
            return False
            
        try:
            # グラフサイズの設定
            plt.figure(figsize=(16, 12))
            
            # ------------ 資産推移グラフ ------------
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(self.results_df.index, self.results_df['equity'], 'b-', linewidth=1.5)
            
            # 取引ポイントのプロット
            if self.trades_df is not None and len(self.trades_df) > 0:
                # 取引データフォーマットの確認
                entry_time_col = 'entry_time' if 'entry_time' in self.trades_df.columns else 'open_time'
                exit_time_col = 'exit_time' if 'exit_time' in self.trades_df.columns else 'close_time'
                direction_col = 'direction' if 'direction' in self.trades_df.columns else 'type'
                
                if entry_time_col in self.trades_df.columns and exit_time_col in self.trades_df.columns:
                    buy_indices = []
                    sell_indices = []
                    
                    for _, trade in self.trades_df.iterrows():
                        # 時間をインデックスとして使用するため、最も近い結果のインデックスを探す
                        entry_idx = self.results_df.index.get_indexer([trade[entry_time_col]], method='nearest')[0]
                        exit_idx = self.results_df.index.get_indexer([trade[exit_time_col]], method='nearest')[0]
                        
                        if direction_col in self.trades_df.columns:
                            if trade[direction_col] == 'long':
                                # 買いポイント
                                buy_indices.append(entry_idx)
                                # 売りポイント（決済）
                                sell_indices.append(exit_idx)
                            else:
                                # 売りポイント
                                sell_indices.append(entry_idx)
                                # 買いポイント（決済）
                                buy_indices.append(exit_idx)
                        else:
                            # 方向情報がない場合はすべてlong扱い
                            buy_indices.append(entry_idx)
                            sell_indices.append(exit_idx)
                            
                    if buy_indices:
                        ax1.plot(self.results_df.index[buy_indices], self.results_df['equity'].iloc[buy_indices], 'g^', markersize=5)
                    if sell_indices:
                        ax1.plot(self.results_df.index[sell_indices], self.results_df['equity'].iloc[sell_indices], 'rv', markersize=5)
            
            ax1.set_title('資産推移', fontsize=14)
            ax1.set_ylabel('資産額', fontsize=12)
            ax1.grid(True)
            
            # X軸のフォーマット
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            
            # ------------ ドローダウングラフ ------------
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            
            # ドローダウンがない場合は計算
            if 'drawdown' not in self.results_df.columns:
                running_max = self.results_df['equity'].cummax()
                drawdown = (self.results_df['equity'] / running_max - 1) * 100
            else:
                drawdown = self.results_df['drawdown']
                
            ax2.fill_between(self.results_df.index, 0, drawdown, color='r', alpha=0.3)
            ax2.set_title('ドローダウン(%)', fontsize=14)
            ax2.set_ylabel('ドローダウン (%)', fontsize=12)
            ax2.grid(True)
            
            # ------------ 利益推移グラフ ------------
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            
            # 取引データが利用可能な場合は描画
            if self.trades_df is not None and len(self.trades_df) > 0:
                # 取引データフォーマットの確認
                exit_time_col = 'exit_time' if 'exit_time' in self.trades_df.columns else 'close_time'
                pnl_col = 'pnl' if 'pnl' in self.trades_df.columns else 'profit'
                
                if exit_time_col in self.trades_df.columns and pnl_col in self.trades_df.columns:
                    # 日付でソート
                    trade_df_sorted = self.trades_df.sort_values(exit_time_col)
                    
                    # 累積利益の計算と描画
                    trade_df_sorted['cumulative_pnl'] = trade_df_sorted[pnl_col].cumsum()
                    
                    # 利益の取引をマーク
                    profitable_trades = trade_df_sorted[trade_df_sorted[pnl_col] > 0]
                    losing_trades = trade_df_sorted[trade_df_sorted[pnl_col] < 0]
                    
                    ax3.plot(trade_df_sorted[exit_time_col], trade_df_sorted['cumulative_pnl'], 'b-', linewidth=1.5)
                    
                    if len(profitable_trades) > 0:
                        ax3.scatter(profitable_trades[exit_time_col], profitable_trades[pnl_col], color='g', s=30, alpha=0.5)
                    if len(losing_trades) > 0:
                        ax3.scatter(losing_trades[exit_time_col], losing_trades[pnl_col], color='r', s=30, alpha=0.5)
            else:
                # 取引データがない場合は資産の変化を表示
                equity_diff = self.results_df['equity'].diff()
                ax3.plot(self.results_df.index, equity_diff.cumsum(), 'b-', linewidth=1.5)
                
                # 日々の変化をマーク
                positive_changes = self.results_df[equity_diff > 0]
                negative_changes = self.results_df[equity_diff < 0]
                
                if len(positive_changes) > 0:
                    ax3.scatter(positive_changes.index, equity_diff[equity_diff > 0], color='g', s=30, alpha=0.5)
                if len(negative_changes) > 0:
                    ax3.scatter(negative_changes.index, equity_diff[equity_diff < 0], color='r', s=30, alpha=0.5)
                
            ax3.set_title('累積利益と取引', fontsize=14)
            ax3.set_ylabel('利益額', fontsize=12)
            ax3.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"バックテスト結果のグラフを保存しました: {output_filename}")
            return True
        
        except Exception as e:
            print(f"グラフ生成エラー: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='バックテスト結果の可視化')
    parser.add_argument('--results', type=str, default='advanced_backtest_results.csv',
                        help='バックテスト結果のCSVファイル名')
    parser.add_argument('--trades', type=str, default='advanced_backtest_trades.csv',
                        help='取引履歴のCSVファイル名')
    parser.add_argument('--output', type=str, default=None,
                        help='出力HTMLファイル名 (デフォルト: backtest_report_YYYYMMDD_HHMMSS.html)')
    
    args = parser.parse_args()
    
    visualizer = BacktestVisualizer()
    if visualizer.load_data(args.results, args.trades):
        output_path = visualizer.generate_html_report(args.output)
        print(f"レポートを以下に保存しました: {output_path}")
    else:
        print("バックテストデータの読み込みに失敗しました。")

if __name__ == "__main__":
    main() 