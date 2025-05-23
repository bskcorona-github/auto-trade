@echo off
echo Gitリポジトリを初期化しています...
git init

echo リモートリポジトリを追加しています...
git remote add origin https://github.com/bskcorona-github/auto-trade.git

echo ファイルをステージングエリアに追加しています...
git add .

echo 変更をコミットしています...
git commit -m "Initial commit with .gitignore and project files"

echo GitHubにプッシュしています...
git push -u origin master

echo 完了しました！
pause 