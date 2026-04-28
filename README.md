# texteller4chemical

Fine-tune/domain-adapt pretrained TexTeller cho bài toán nhận dạng công thức hoặc cấu trúc hóa học viết tay trên EDU-CHEMC / EDU-CHMEC_MM23.

Project này cố ý **không** yêu cầu Tex80M và **không** train TexTeller từ đầu theo mặc định. Luồng chuẩn là:

```text
handwritten chemical image -> pretrained TexTeller base -> fine-tune -> chemical markup sequence
```

## Vì Sao Không Cần Tex80M

Tex80M là dữ liệu pretraining gốc của TexTeller. Ở đây mục tiêu là fine-tuning/domain adaptation: tận dụng encoder-decoder pretrained đã học OCR/HMER tổng quát, rồi thích nghi sang ảnh công thức hóa học viết tay. Bạn chỉ cần EDU-CHEMC_MM23 và một checkpoint/model id TexTeller pretrained hợp lệ.

Nếu bạn bật `--from_scratch`, script sẽ dừng với lỗi rõ ràng vì đó không phải pipeline mặc định của project này.

Model pretrained mặc định hiện dùng:

```text
OleehyO/TexTeller
```

Repo HuggingFace này có `config.json` kiểu `vision-encoder-decoder`, encoder `vit`, decoder `trocr`, weight `model.safetensors`, tokenizer BPE và ONNX export. Encoder của checkpoint nhận ảnh grayscale `1 x 448 x 448`, nên config mặc định của project cũng dùng resize/pad về `448 x 448`, `channels: 1`.

## Math HMER Và EDU-CHEMC Khác Nhau

- Math HMER thường là `image -> LaTeX toán học`.
- EDU-CHEMC là `image -> chemical markup`, có bond, atom, branch, reconnection mark, góc liên kết và cú pháp ChemFig-like.
- Vì vậy không nên coi toàn bộ target là LaTeX toán học thông thường.

Các target thường gặp:

- `chemfig`: chuỗi ChemFig / LaTeX-based chemical markup, ví dụ `\chemfig{...}`.
- `ssml_sd`: sequence target cho encoder-decoder thường, mặc định của project này.
- `ssml_rcgd`: target có tuple/list cho random conditional guided decoder, không phù hợp với TexTeller decoder gốc ở baseline.
- `ssml_normed`: normalized SSML, thường dùng cho evaluation/submission.

Baseline nên dùng `ssml_sd`. Sau khi có model tốt, mới thử `chemfig` hoặc một decoder riêng cho `ssml_rcgd`.

## Cài Đặt

```bash
uv sync
```

Nếu cần package TexTeller được publish dưới tên `texteller`:

```bash
uv add texteller
```

Nếu muốn lấy thêm extras training do repo TexTeller cung cấp:

```bash
uv add "texteller[train]"
```

Nếu package không expose training API, clone repo tham khảo để tự map loader:

```bash
git clone https://github.com/OleehyO/TexTeller external/TexTeller
```

Sau đó truyền:

```bash
--texteller_repo_path external/TexTeller
```

`src/chemtexteller/model_loader.py` hiện thử các đường nạp sau:

1. `AutoModelForVision2Seq`
2. `VisionEncoderDecoderModel`
3. package `texteller` nếu có API kiểu `from_pretrained`

Nếu cả ba thất bại, lỗi sẽ hướng dẫn bạn truyền đúng model id/checkpoint hoặc chỉnh adapter loader theo API thật của TexTeller.

## Chuẩn Bị Dataset

Dataset của bạn đang ở:

```text
F:\dataset\EDU-CHMEC_MM23
```

Nếu ảnh và JSON nằm trực tiếp trong folder đó:

```bash
uv run python scripts/prepare_edu_chemc.py \
  --src_dir F:/dataset/EDU-CHMEC_MM23 \
  --out_dir data/processed/edu_chemc \
  --target_field ssml_sd \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

Output:

```text
data/processed/edu_chemc/
  train/
    sample_000001.png
    metadata.jsonl
  validation/
    metadata.jsonl
  test/
    metadata.jsonl
  dataset_stats.json
```

Mỗi dòng `metadata.jsonl` có dạng:

```json
{"file_name": "sample_000001.png", "target": "\\chemfig { ... }"}
```

Không dùng `ssml_rcgd` cho baseline. Script sẽ chặn target này trừ khi bạn truyền `--allow_rcgd`.

## Phân Tích Target

```bash
uv run python scripts/analyze_targets.py \
  --metadata data/processed/edu_chemc/train/metadata.jsonl
```

Report được lưu ở:

```text
outputs/reports/target_analysis.md
outputs/reports/target_analysis.json
```

Script đếm token, độ dài sequence, và các pattern chemical markup như `\chemfig`, `\Chemabove`, `branch`, `?[a]`, bond angle, `\circle`.

## Kiểm Tra Tokenizer Coverage

Trước khi train, kiểm tra tokenizer TexTeller pretrained có xử lý tốt token hóa học không:

```bash
uv run python scripts/analyze_tokenizer_coverage.py \
  --metadata data/processed/edu_chemc/train/metadata.jsonl \
  --pretrained_model_name_or_path OleehyO/TexTeller
```

Nếu OOV/unknown cao, có thể extend tokenizer:

```bash
uv run python scripts/analyze_tokenizer_coverage.py \
  --metadata data/processed/edu_chemc/train/metadata.jsonl \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --extend_tokenizer \
  --vocab_file F:/dataset/EDU-CHMEC_MM23/EDU-CHEMC.vocab \
  --output_tokenizer_dir outputs/tokenizer_edu_chemc
```

Khi dùng tokenizer đã extend, train với:

```bash
--tokenizer_path outputs/tokenizer_edu_chemc
```

Nếu báo mismatch vocab size, nghĩa là tokenizer đã thêm token nhưng decoder embedding/lm head của model chưa resize được. Khi đó cần chỉnh `resize_token_embeddings_if_needed()` theo API decoder thật của TexTeller.

## Fine-Tune

```bash
uv run accelerate launch scripts/train.py \
  --config configs/train_edu_chemc.yaml \
  --dataset_dir data/processed/edu_chemc \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --output_dir outputs/runs/edu_chemc_texteller
```

## Kaggle Quick Start

Có thể chạy project này trên Kaggle. Cách nhanh nhất:

1. Upload dataset EDU-CHEMC_MM23 lên Kaggle dưới dạng private Dataset.
2. Upload repo này lên GitHub, hoặc upload repo zip thành Kaggle Dataset riêng.
3. Tạo Kaggle Notebook, bật accelerator GPU.
4. Không cài lại `torch/torchvision` nếu Kaggle đã có CUDA-enabled PyTorch.

Ví dụ notebook nếu clone từ GitHub:

```bash
!git clone <YOUR_REPO_URL> /kaggle/working/texteller4chemical
%cd /kaggle/working/texteller4chemical
!pip install -r requirements-kaggle.txt
!pip install -e . --no-deps
```

Kiểm tra GPU:

```bash
!python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Chuẩn bị dataset, thay `<YOUR_KAGGLE_DATASET_DIR>` bằng path trong `/kaggle/input`:

```bash
!python scripts/prepare_edu_chemc.py \
  --src_dir /kaggle/input/<YOUR_KAGGLE_DATASET_DIR> \
  --out_dir /kaggle/working/data/processed/edu_chemc \
  --target_field ssml_sd \
  --copy_mode reference
```

`--copy_mode reference` is recommended on Kaggle because it keeps images in `/kaggle/input` and writes only metadata/splits to `/kaggle/working`. Avoid `copy` on Kaggle unless you intentionally want to duplicate the full image dataset into the notebook output.

Fine-tune:

```bash
!accelerate launch scripts/train.py \
  --config configs/train_edu_chemc_kaggle.yaml \
  --dataset_dir /kaggle/working/data/processed/edu_chemc \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --output_dir /kaggle/working/outputs/runs/edu_chemc_texteller
```

Trên Kaggle, nếu bị OOM thì giảm `per_device_train_batch_size` về `1`, tăng `gradient_accumulation_steps`, hoặc thêm `--freeze_encoder`.

Với tokenizer đã extend:

```bash
uv run accelerate launch scripts/train.py \
  --config configs/train_edu_chemc.yaml \
  --dataset_dir data/processed/edu_chemc \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --tokenizer_path outputs/tokenizer_edu_chemc \
  --output_dir outputs/runs/edu_chemc_texteller
```

Nếu dùng local clone TexTeller:

```bash
uv run accelerate launch scripts/train.py \
  --config configs/train_edu_chemc.yaml \
  --dataset_dir data/processed/edu_chemc \
  --pretrained_model_name_or_path <LOCAL_CHECKPOINT_OR_MODEL_DIR> \
  --texteller_repo_path external/TexTeller \
  --output_dir outputs/runs/edu_chemc_texteller
```

Checkpoint được lưu ở:

```text
outputs/runs/edu_chemc_texteller/last
outputs/runs/edu_chemc_texteller/best
```

## Evaluate

```bash
uv run python scripts/evaluate.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --dataset_dir data/processed/edu_chemc \
  --split test \
  --output_csv outputs/eval_predictions.csv
```

Metrics hiện có:

- exact match raw
- exact match sau normalize whitespace
- token-level edit distance
- normalized token edit distance
- character-level edit distance
- average target/prediction length

Không dùng render-based metric mặc định vì baseline target là SSML/ChemFig-like sequence, không phải LaTeX toán học render đơn giản.

## Predict Một Ảnh

```bash
uv run python scripts/predict.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --image_path /path/to/image.png
```

Beam search:

```bash
uv run python scripts/predict.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --image_path /path/to/image.png \
  --num_beams 4 \
  --max_new_tokens 512
```

## Config Mặc Định

Config chính nằm ở:

```text
configs/train_edu_chemc.yaml
```

Mặc định khá tiết kiệm VRAM:

- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 16`
- `max_target_length: 512`
- ảnh grayscale resize/pad `448 x 448`, khớp encoder ViT của `OleehyO/TexTeller`
- LR `1e-5`
- cosine scheduler

Nếu GPU 16GB bị OOM:

- giữ batch size = 1
- tăng `gradient_accumulation_steps`
- bật `--gradient_checkpointing` nếu model hỗ trợ
- bật `--freeze_encoder`
- thử LoRA bằng `--use_lora`
- giảm `image_size`
- giảm `max_target_length`

Nếu loss NaN:

- tắt fp16/bf16
- giảm learning rate
- kiểm tra target rỗng/lỗi bằng `analyze_targets.py`
- kiểm tra tokenizer coverage
- kiểm tra embedding resize sau khi extend tokenizer

Nếu model sinh token lạ:

- kiểm tra `outputs/reports/tokenizer_coverage.json`
- dùng tokenizer extended nếu unknown/OOV cao
- giảm LR
- train lâu hơn
- tăng dữ liệu augmentation nhẹ nhưng không làm biến dạng bond/atom

## Điểm Cần Sửa Khi Dùng Checkpoint TexTeller Gốc

TexTeller repo/package có thể không tương thích trực tiếp với HuggingFace `VisionEncoderDecoderModel`. Khi đó cần sửa `src/chemtexteller/model_loader.py`:

- import class model thật từ clone TexTeller
- gọi đúng hàm load checkpoint
- map image processor/preprocess đúng với checkpoint
- set `pad_token_id`, `bos_token_id`, `eos_token_id`, `decoder_start_token_id`
- nếu extend tokenizer, resize decoder token embedding và output head
- đảm bảo forward nhận `pixel_values` và `labels`, hoặc viết custom training loop thay vì `Seq2SeqTrainer`

Không nên thay decoder bằng vocab mới rồi train từ đầu toàn bộ model. Với chemical domain adaptation, hướng hợp lý là giữ pretrained encoder/decoder càng nhiều càng tốt, chỉ resize/extend tokenizer khi thật sự cần.

## Lỗi Thường Gặp Khi Chuyển Từ Math HMER Sang Chemical Markup

- Tokenizer LaTeX toán tách sai token chemical như `branch`, `?[a]`, `-[:30]`, `<:[:270]`.
- Augmentation quá mạnh làm đổi topology bond hoặc làm mất atom nhỏ như `H`, `O`, `N`.
- Dùng `ssml_rcgd` với decoder sequence thường, dẫn tới target JSON/list dài và không học đúng cơ chế condition/reconnection.
- Normalize whitespace không nhất quán, làm exact match thấp dù sequence gần đúng.
- `chemfig` render được bằng LaTeX nhưng `ssml_sd` là target sequence cho học trực tiếp, không nên áp metric render mặc định.
- Max target length quá thấp làm truncate công thức vòng/nhánh dài.
- Resize ảnh làm bond mảnh biến mất; nên giữ aspect ratio và pad thay vì stretch.
- Extend tokenizer nhưng quên resize embedding/lm head.
