# いいねカウンター（Thumb_Up）— 使い方ガイド

軽量な **MediaPipe Gesture Recognizer** を使って「Thumb_Up（親指グッ）」を検出し、
**カウンター表示＋紙吹雪＋ハート＋「+1」エフェクト＋ランダム効果音（WAV）** を出します。
CPU負荷を抑えるため 640×480 / 約15FPS で動作します（Windows想定）。

---

## 必要環境

- Python 3.10（推奨。仮想環境でセットアップ）
- Webカメラ
- モデルファイル：`assets/gesture_recognizer.task`

---

## セットアップ手順

```powershell
# 1) 仮想環境を作成＆有効化
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) 依存ライブラリをインストール
python -m pip install -U pip
python -m pip install numpy==1.26.4 opencv-python==4.8.1.78 mediapipe==0.10.14

# 3) モデルを配置
# 下記URLから gesture_recognizer.task をダウンロードして assets 配下へ
# https://storage.googleapis.com/mediapipe-assets/gesture_recognizer.task
mkdir assets
# （手動でファイルを入れてください）

# 4) オーディオフォルダ（任意）を作成して WAV を複数置く
mkdir audio
# 例: audio\like1.wav, audio\like2.wav, ...
```

## 実行方法

```
# 音あり（audio/ 内の .wav からランダム再生）
python .\gesture_count.py --model assets\gesture_recognizer.task

# 音なし
python .\gesture_count.py --model assets\gesture_recognizer.task --no-sound

# FPS 表示を出す
python .\gesture_count.py --model assets\gesture_recognizer.task --show-fps

# カメラ/解像度の指定例
python .\gesture_count.py --model assets\gesture_recognizer.task --camera 1 --width 1280 --height 720

```

## 操作方法

```
q … 終了
r … カウンターを 0 にリセット
```

## 機能概要

* 検出対象：Thumb_Up（スコアしきい値 0.6）
* エフェクト：紙吹雪、赤い ハート、浮き上がる 「+1」
* 効果音：audio/ の .wav を ランダムに1つ再生（Windows 標準 winsound 使用）
* クールダウン：3 秒（直後の連続検出はカウントしない）
* 軽量化：640×480 / 約15FPS（処理タイミング以外のフレームはスキップ）

## コマンドラインオプション

| オプション                 |                             既定値 | 説明                                  |
| -------------------------- | ---------------------------------: | ------------------------------------- |
| `--model`                | `assets/gesture_recognizer.task` | MediaPipe の `.task` モデルへのパス |
| `--camera`               |                              `0` | カメラインデックス                    |
| `--width` / `--height` |                  `640` / `480` | キャプチャ解像度                      |
| `--audio-dir`            |                          `audio` | ランダム再生する WAV を入れるフォルダ |
| `--no-sound`             |                                  - | 効果音なし                            |
| `--show-fps`             |                                  - | ウィンドウ左上に瞬間FPSを表示         |
