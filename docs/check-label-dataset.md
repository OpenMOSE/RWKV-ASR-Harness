# Label データセットの検証

機密の音声データを `data_type=label` で学習する前に、**学習コードを起動せず**に
データの健全性を確認するためのツールです。モデルも GPU も不要なので、
データ管理側のマシン単体で実行できます。

- スクリプト本体: [`tools/check_label_dataset.py`](../tools/check_label_dataset.py)
- ランチャー: [`scripts/check_label.sh`](../scripts/check_label.sh)

検証が緑（PASS）であれば、実際の dataloader も同じデータを問題なく読み込めます
（`tools/check_label_dataset.py` は学習側 `lg_train/dataset.py` のインデックス構築・
パス解決・`speech_token_len` のロジックを忠実に再現しています）。

---

## `.label` ファイルの形式

`--root` 配下を再帰的に走査し、拡張子 `*.label` のファイルを読み込みます。
各行は **「音声パス」＋空白（スペース or TAB）＋「書き起こし」** です。

```
ja/WAVE/utt_0001.wav<TAB>こんにちは、世界。
ja/WAVE/utt_0002.wav おはようございます
```

- 音声パスは相対パス（`--root` または `--root` の親から解決）または絶対パス。
- 1ファイルに複数行を書けます。空行・カラムが1つだけの行はスキップされます。
- 音声パスにスペースが含まれると `split` が誤るため、警告（`warn`）として検出します。

### パス解決（混在・1階層ズレに対応）

`.label` 内の音声パスは現実にはバラつきます（絶対 / `.label` 基準の相対 / データ
セットルート基準 / 1階層余分・不足など）。チェッカーは次の候補を**優先順に総当り**
し、各 `.label` ディレクトリごとに成功した方式をキャッシュします:

1. `AUDIO_ROOT`（指定時。最優先）
2. `.label` ファイル自身のディレクトリ
3. その親を `MAX_UP` 階層まで遡る（不足ズレ対策）
4. データセットルート / その親
   - 各 base について「そのまま」と「先頭1階層を除去（余分ズレ対策）」を試行

SUMMARY に **どの方式で解決できたかの内訳**と解決率が出ます:

```
path resolution: {'label_dir-1:asis': 41000, 'abs': 1200, 'MISSING': 30}
               resolved 42200/42230 (99.9%) | MISSING 30
```

`missing` の行には実際に試した候補パスが併記されるので、ベースディレクトリの
指定ミス（`AUDIO_ROOT` / `MAX_UP` で調整）か、本当に存在しないのかを判別できます。

> ✅ **学習側も同じ解決ロジックに更新済み:** `WorldDataset._resolve_audio_path`
> はこのチェッカーと同じ多方式解決（`.label` 基準・親遡り・1階層ズレ・`audio_root`）
> になりました。`train.py` の `--audio_root` / `--resolve_max_up`、または
> `train_asr_stage1.sh` の `AUDIO_ROOT` / `RESOLVE_MAX_UP` で同様に調整できます。
> さらに学習中はスキップ率を `[label] skip rate XX% (...)` として定期表示するので、
> サイレントなデータ欠落（＝過学習の原因）にすぐ気づけます。

---

## 実行方法

### bash ランチャー（推奨）

```bash
# クイックスキャン（全件ヘッダ確認 + ランダム300件をフルデコード）
ROOT=/share/voice-dataset ./scripts/check_label.sh

# 【大規模・推奨】各 .label ファイルから 5% を層化抽出してフルチェック
#   → 残り95%には一切触れない（3TB級はこれが現実的）
ROOT=/share/voice-dataset FRAC=0.05 ./scripts/check_label.sh

# 全件をフルデコード（網羅的・低速）
ROOT=/share/voice-dataset CHECK_ALL=1 ./scripts/check_label.sh

# misc / noise を含むパスを除外して検証
ROOT=/share/voice-dataset LABEL_EXCLUDE=misc,noise ./scripts/check_label.sh

# CI 用: 問題があれば exit code != 0
ROOT=/share/voice-dataset STRICT=1 ./scripts/check_label.sh
```

#### 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `ROOT` | （必須） | `*.label` を含むルートフォルダ |
| `CTX_LEN` | `1024` | 学習と同じ context 長（token_len 上限の判定に使用） |
| `LABEL_EXCLUDE` | 空 | スキップするキーワード（カンマ区切り、例 `misc,noise`） |
| `MAX_CHECK` | `300` | フルデコードする件数（ランダム抽出） |
| `FRAC` | off | 各 `.label` ファイルから抽出する割合（例 `0.05`=5%）。残りは未処理。抽出分はフルデコード |
| `CHECK_ALL` | `0` | `1` で全件フルデコード（低速・網羅的） |
| `WORKERS` | `8` | デコードのスレッド数 |
| `SHOW` | `3` | トークン化詳細を表示するサンプル数 |
| `EXAMPLES` | `8` | カテゴリごとに表示する問題例の数 |
| `REPORT` | `label_check_report.jsonl` | 問題行の**全リスト**を書き出す JSONL パス |
| `NO_REPORT` | `0` | `1` でレポートファイルを書かない |
| `DECODE_ERRORS` | `skip` | UTF-8 として不正な `.label` ファイルの扱い。`skip`=警告してファイルごとスキップ / `ignore`=不正バイトのみ捨てて有効行は採用 |
| `STRICT` | `0` | `1` で問題検出時に exit code 1 |

### 3TB 級データのおすすめワークフロー

全件（`CHECK_ALL=1`）はファイルヘッダを全部開くため I/O が膨大です。代わりに
**`FRAC` で各 `.label` ファイルから一定割合だけを層化抽出**してください。抽出
されなかったエントリはパス解決・ヘッダ読み込み・デコードを**一切行いません**。

```bash
# まず 1% で全体傾向を素早く把握 → 次に 5% で精査
ROOT=/share/voice-dataset FRAC=0.01 ./scripts/check_label.sh
ROOT=/share/voice-dataset FRAC=0.05 WORKERS=16 ./scripts/check_label.sh
```

- 抽出は **各 `.label` ファイル単位**（層化）なので、フォルダ/コーパスごとの
  偏りなく全体を代表します。`SEED`（既定42）で再現可能。
- 抽出された分は**フルデコード**され、音声整合性・token_len・ラベル整合まで検査
  されます（`FRAC` 指定時は `CHECK_ALL` 相当の精査を抽出集合に適用）。
- 1ファイルあたり最低1件は必ず残します（空サンプル防止）。

> **大規模データの扱い:** `CHECK_ALL=1` / `FRAC=...` のどちらでも、ターミナルには
> ストリーミング集計したカウンタとヒストグラムのみを表示し（メモリは件数に
> 依存しない O(1)）、個々の問題行は `$REPORT`（JSONL）に逐次書き出します。
> 進捗は `処理数/総数 (％) | entries/s | elapsed` 形式でライブ表示されます。
> 後から問題だけを確認するには:
> ```bash
> # too_long のものだけ抽出
> grep '"status": "too_long"' label_check_report.jsonl | head
> # ステータス別の件数
> python -c "import json,collections;print(collections.Counter(json.loads(l)['status'] for l in open('label_check_report.jsonl')))"
> ```

### 直接呼び出し

```bash
python tools/check_label_dataset.py --root /share/voice-dataset \
    --ctx_len 1024 --label_exclude misc,noise --check_all --show 5 --strict
```

`python tools/check_label_dataset.py --help` で全オプションを確認できます。

---

## 検証する内容

1. **INDEX** — `.label` 走査数、抽出発話数、除外数、不正/空行数。
   - **「utterances indexed」が 0** の場合、dataloader は 0 サンプルになります
     （学習時の `ValueError: num_samples should be a positive integer value,
     but got num_samples=0` の直接原因）。
2. **パス解決＋存在確認** — `missing`（ファイル無し）を集計。
3. **音声整合性** — soundfile/librosa でデコード（mp3 等は librosa フォールバック）、
   サンプルレート・チャンネル・長さ・フォーマットの分布。`decode_fail` を分類。
4. **token_len ガード** — 16kHz リサンプル後の長さから算出し、`ctx_len-32` を超える
   クリップ（NaN-loss 回避のため学習時にドロップされる）を `too_long` として検出。
5. **ラベル抽出（deep check）** — `<|image_pad|>` が単一トークン `65532` に
   エンコードされるか、プレースホルダ数 == `token_len` の整合、教師あり位置を
   デコードして書き起こしが復元されるか（例: `Assistant:こんにちは世界`）。

---

## 出力例

```
========== INDEX ==========
.label files scanned : 1 (excluded by keyword: 0)
utterances indexed   : 5
entries excluded     : 1
malformed/empty lines: 1

========== AUDIO CHECK ==========
header check: ALL 5 entries | full decode: 5 entries (all)
problem report -> label_check_report.jsonl
  5/5 (100.0%) |    511 entries/s | elapsed     0s

========== SUMMARY ==========
status counts: {'missing': 1, 'ok': 3, 'too_long': 1}
sample rate  : {16000: 4}
duration (s) : n=4 min=0.50 mean=31.12 max=120.00
  duration histogram:
    0.5-1s |  1  25.0%  ##########
      1-2s |  1  25.0%  ##########
      3-5s |  1  25.0%  ##########
    >= 60s |  1  25.0%  ##########
token_len    : n=4 min=13 mean=779.0 max=3001 | ctx_len-32=992 (1 over limit)
  token_len histogram:
      < 248 |  3  75.0%  ##############################
     >= 992 |  1  25.0%  ##########

---- missing: 1 ----
  ja.label:5 | .../missing.wav -> audio file not found
---- too_long: 1 ----
  ja.label:4 | .../big.wav -> token_len=3001 exceeds ctx_len-32 (992) ...

========== DEEP TOKENIZATION CHECK ==========
pipeline.encode('<|image_pad|>') -> [65532]
[ok] placeholder -> single token id 65532
--- sample [ok] a.wav ---
  transcript      : こんにちは世界
  token_len(audio): 26  | placeholder tokens in input: 26
  supervised toks : 12  | seq len: 1024 / ctx 1024
  decoded labels  : Assistant:こんにちは世界
[deep] alignment mismatches: 0 / 3 buildable samples checked

========== VERDICT ==========
PASS — index loads, audio decodes, and placeholder/label extraction aligns.
```

### ステータスの意味

| status | 意味 | 対処 |
|---|---|---|
| `ok` | 問題なし | — |
| `missing` | 音声ファイルが見つからない | パス・`--root` を確認 |
| `decode_fail` | デコード失敗（破損・非対応形式） | 該当ファイルを除外/修復 |
| `too_long` | `token_len > ctx_len-32`（学習でドロップ） | `CTX_LEN` を上げるか短いクリップに分割 |
| `warn` | 音声拡張子が無い等のパース疑い | `.label` の区切り（パス中のスペース）を確認 |

---

## 文字エンコーディング異常への耐性

`.label` が UTF-8 として不正なバイト（例: `0xf1`）を含む場合でも処理は止まりません。

- 既定（`DECODE_ERRORS=skip`）: そのファイルを **警告付きでスキップ**し、
  `[warn] skip .label (not valid UTF-8: ...)` と表示。INDEX サマリに
  `.label files skipped : N (not valid UTF-8)` として件数が出ます。
- `DECODE_ERRORS=ignore`: 不正バイトだけを捨て、そのファイルの**有効行は採用**します。

また、巨大コーパスでインデックス構築が無言にならないよう、
`[index] scanned N .label files | M entries kept | skipped(bad-utf8)=K | 秒数`
の進捗を 5000 ファイルごとに表示します（「警告の後フリーズ」に見える問題を解消）。

## 依存パッケージ

コア検査: `numpy`, `soundfile`, `librosa`（学習環境に含まれています）。
deep check 用の RWKV tokenizer が import できない環境でも、それ以外の検査は実行されます。
