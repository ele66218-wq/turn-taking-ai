# turn-taking-ai

**リアルタイム音声対話におけるユーザー割り込み意図推定**
（「AIが話している最中に、相手がいつ発話権を欲しがっているかを声色から当てる」プロジェクト）

AIが話している途中でユーザーが「うーん」「えっと」と発話を始めたとき、それが単なる相槌
（続けてよい）なのか、発話権を取りに来ている合図（止まるべき）なのかを、音声の
音響特徴（F0・音量・スペクトル・時間構造）から推定します。テキストや意味内容には頼らず、
**声色とタイミングだけ**で判断するのがこのプロジェクトの核心です。

```
直近0.5〜1秒の音響特徴 → P(ユーザーが発話権を欲しがっている) → CONTINUE / PAUSE / YIELD
```

## できること

- `data/raw/labels.csv` ベースのアノテーション管理（continue / backchannel /
  hesitation / interrupt / noise / unsure の6ラベル）
- numpy/scipyのみで実装した音響特徴量抽出（F0・RMS・スペクトル重心・MFCC・jitter/shimmer等、
  93次元）— `librosa`等の重い依存なし
- scikit-learnによるベースラインモデル学習（RandomForest / LogisticRegression）
- 話者グループ化cross-validationと、誤停止率・割り込み見逃し率という
  ドメイン固有の評価指標
- CONTINUE / PAUSE / YIELD の3値コントローラ（ヒステリシス・AI発話直後のエコーガード付き）
- マイク入力 or WAVファイル再生によるリアルタイム制御ループ
- マイクなしでも全パイプラインを試せる合成データ生成器

## セットアップ

```bash
cd turn-taking-ai
uv venv --python 3.12 .venv
uv pip install -e .            # コア機能のみ
uv pip install -e '.[mic]'     # マイク録音も使う場合
uv pip install -e '.[dev]'     # テストも回す場合
```

> **メモ（このリポジトリの開発環境固有の注意）:** 環境によっては
> `uv pip install -e .` が作る `__editable__*.pth` が site.py に読み込まれず
> `ModuleNotFoundError: No module named 'turn_taking'` になることがあります
> （原因未特定・要確認）。その場合は編集可能インストールを使わず、
> `uv pip install .`（通常インストール）でコードを反映するか、
> `PYTHONPATH=src python -m turn_taking.cli ...` で直接実行してください。

## 使い方

### 1. まず合成データで動作確認（マイク不要）

```bash
turn-taking synth --out data/raw --n-per-label 60
turn-taking dataset-info --labels data/raw/labels.csv
turn-taking train --labels data/raw/labels.csv --audio-root data/raw
turn-taking predict data/raw/synthetic/syn_interrupt_0001.wav
turn-taking realtime --simulate data/raw/synthetic/syn_interrupt_0001.wav
```

合成音声はF0・音量・持続時間などをラベルごとに単純化して再現したものに過ぎず、
**実音声の代わりにはなりません。** モデルの学習はできても、性能評価は無意味です。
必ずパイプラインの疎通確認用と考えてください。

### 2. 実際に自分の声を録音してラベル付け

```bash
python scripts/record_clip.py --speaker あなたの名前 --session 2026-09-13 --loop
```

1クリップずつ録音→再生確認→ラベル選択、を繰り返して
`data/raw/labels.csv` と `data/raw/manual/*.wav` に追記されます。
最初の実験としては、以下を50個ずつくらい集めるのがおすすめです（設計時の議論より）。

- 相槌の「うーん」（backchannel）
- 考え始めの「うーん……えっと」（hesitation）
- 明確な割り込み「いや、ちょっと待って」（interrupt）
- 無関係な雑音・小声（noise）
- 何も起きていない区間（continue）

### 3. 実データで学習・評価

```bash
turn-taking train --labels data/raw/labels.csv --audio-root data/raw \
    --model-out models/baseline.joblib
```

話者(speaker)が2人以上いれば、話者グループ化k-fold cross-validationが自動で走ります。
出力される `false_stop_rate`（誤って止まる率）と `missed_interrupt_rate`
（割り込みを見逃す率）が、単純な精度より重要な評価指標です。

### 4. リアルタイムで試す

```bash
turn-taking realtime --model models/baseline.joblib          # マイク入力
turn-taking realtime --model models/baseline.joblib --simulate some_clip.wav  # WAV再生
```

## 設定

`config/default.yaml` に音声フレーミング・特徴量・コントローラ閾値・学習パラメータを
まとめています。`--config path/to/your.yaml` で上書き可能（キーは
`src/turn_taking/config.py` のdataclassと1:1対応、未知キーはエラーになります）。

## プロジェクト構成

```
turn-taking-ai/
├── config/default.yaml      # デフォルト設定
├── data/raw/                # labels.csv + 録音済みwav（wav本体はgit管理外）
├── models/                  # 学習済みモデル(.joblib) — git管理外
├── scripts/record_clip.py   # 対話式の録音・ラベル付けヘルパー
├── src/turn_taking/
│   ├── config.py            # 設定dataclass + YAML読み込み
│   ├── labels.py            # アノテーションスキーマ + labels.csv I/O
│   ├── audio_io.py          # WAV I/O・マイク入力・リサンプリング
│   ├── features.py          # 音響特徴量抽出(numpy/scipyのみ)
│   ├── synthetic.py         # 合成データ生成器(マイクなし動作確認用)
│   ├── dataset.py           # labels.csv → 特徴量行列
│   ├── metrics.py           # 誤停止率/割り込み見逃し率などの評価指標
│   ├── train.py             # scikit-learnベースライン学習
│   ├── infer.py             # 1ウィンドウ推論
│   ├── controller.py        # CONTINUE/PAUSE/YIELDステートマシン
│   ├── realtime.py          # マイク/WAV再生の実行ループ
│   └── cli.py                # `turn-taking` CLI
├── web/                     # ブラウザ版デモ(GitHub Pages, JS完結)
└── tests/                   # pytestユニットテスト
```

## 開発ロードマップ（将来の拡張）

- **Stage 2**: 手設計特徴量 → wav2vec2/HuBERT埋め込みに置き換え、小さいMLPで比較実験
- ASRの部分認識結果（「はい」相槌 vs 「いや、それは…」）を追加特徴として組み込む
- 社内IVRへの限定導入前に、既存VAD方式 vs 提案手法の比較実験（誤停止率・見逃し率・
  停止までの遅延で評価）

## ライセンス

MIT
