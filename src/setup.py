"""
セットアップスクリプト
初期設定と依存関係のインストール
"""
import os
import sys
import subprocess
import shutil
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Setup')

def check_python_version():
    """Pythonバージョンチェック"""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        logger.error(f"Python {required_version[0]}.{required_version[1]} 以上が必要です。現在のバージョン: {current_version[0]}.{current_version[1]}")
        return False
    
    logger.info(f"Python バージョン確認: {current_version[0]}.{current_version[1]}")
    return True

def install_dependencies():
    """依存パッケージのインストール"""
    try:
        logger.info("依存パッケージをインストールしています...")
        
        # requirements.txtのパスを取得
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(setup_dir)
        req_path = os.path.join(root_dir, 'requirements.txt')
        
        # インストール実行
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_path])
        
        logger.info("依存パッケージのインストールが完了しました")
        return True
    except Exception as e:
        logger.error(f"依存パッケージのインストールに失敗しました: {e}")
        return False

def create_env_file():
    """環境変数ファイルの作成"""
    try:
        # .envファイルのパスを取得
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(setup_dir, '.env')
        
        # .envファイルが存在しない場合のみ作成
        if not os.path.exists(env_path):
            logger.info(".envファイルを作成しています...")
            
            with open(env_path, 'w') as f:
                f.write("""# 取引所API設定
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
BITFLYER_API_KEY=
BITFLYER_SECRET_KEY=
ALPACA_API_KEY=
ALPACA_SECRET_KEY=

# 取引設定
MAX_POSITION_SIZE=0.01
RISK_PER_TRADE=0.01
STOP_LOSS_PERCENTAGE=0.01
TAKE_PROFIT_PERCENTAGE=0.02

# ニュース設定
NEWS_API_KEY=
""")
            
            logger.info(".envファイルを作成しました。APIキーを設定してください。")
        else:
            logger.info(".envファイルはすでに存在します")
        
        return True
    except Exception as e:
        logger.error(f".envファイルの作成に失敗しました: {e}")
        return False

def create_data_directories():
    """データディレクトリの作成"""
    try:
        # ディレクトリパスを取得
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(setup_dir, 'models', 'saved_models')
        
        # ディレクトリが存在しない場合のみ作成
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info("データディレクトリを作成しました")
        return True
    except Exception as e:
        logger.error(f"データディレクトリの作成に失敗しました: {e}")
        return False

def main():
    """メイン関数"""
    logger.info("セットアップを開始します...")
    
    # Pythonバージョンチェック
    if not check_python_version():
        return
    
    # 依存パッケージインストール
    install_dependencies()
    
    # 環境変数ファイル作成
    create_env_file()
    
    # データディレクトリ作成
    create_data_directories()
    
    logger.info("セットアップが完了しました")
    logger.info("使用方法については USAGE.md を参照してください")

if __name__ == "__main__":
    main() 