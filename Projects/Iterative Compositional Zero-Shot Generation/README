# CCDT

這個專案是以 DiT (Diffusion Transformer) 為基礎修改的條件式影像生成模型。模型使用兩個條件輸入：

- `c1`: attribute condition
- `c2`: object condition

## Environment

建議使用 Conda 建立環境：

```bash
conda env create -f environment.yml
conda activate DiT
```

如果你是手動安裝依賴，請先補裝以下前置套件：

```bash
pip install torchsampler timm diffusers datasets accelerate
```

主要會用到：

- PyTorch
- torchvision
- diffusers
- timm
- torchsampler
- tqdm

訓練需要 GPU，且目前 `train.py` 使用 PyTorch DDP。

## Dataset Format

訓練資料需依照類別資料夾排列，每張圖片放在對應的條件資料夾底下。資料夾名稱格式為：

```text
<attribute> <object>
```

範例：

```text
Phison/
├── Good Group1/
│   ├── xxx.jpg
│   └── ...
├── Good Group2/
├── Shift Group1/
└── Shift Group2/
```

`dataset.py` 會從圖片上一層資料夾名稱解析 label：

```text
Good Group1 -> c1 = Good, c2 = Group1
```

不同資料集的 attribute / object 對應表定義在 `config.py`。

## Train CCDT

使用 2 張 GPU 訓練：

```bash
torchrun --nnodes=1 --nproc_per_node=2 train.py --model DiT-XL/2 --num_condition 4 2
```

如果要指定資料集路徑：

```bash
torchrun --nnodes=1 --nproc_per_node=2 train.py \
  --data_path /path/to/dataset \
  --model DiT-XL/2 \
  --num_condition 4 2
```

常用參數：

- `--data_path`: 訓練資料路徑，預設為 `/workspace/DiT/CelebA`
- `--model`: DiT model size，例如 `DiT-XL/2`
- `--image-size`: 輸入圖片解析度，預設 `128`
- `--num_condition`: 兩個條件的類別數，格式為 `<num_c1> <num_c2>`
- `--epochs`: 訓練 epoch 數，預設 `200`
- `--global-batch-size`: 全域 batch size，預設 `64`
- `--ckpt-every`: 每幾個 epoch 存一次 checkpoint，預設 `50`
- `--results-dir`: checkpoint 和 log 輸出資料夾，預設 `results`

checkpoint 會儲存在：

```text
results/<experiment-name>/checkpoints/
```

## Generate CCDT Images

使用訓練好的 checkpoint 生成圖片：

```bash
python sample.py --num_condition 4 2
```

如果要指定 checkpoint：

```bash
python sample.py \
  --model DiT-XL/2 \
  --num_condition 4 2 \
  --ckpt /path/to/checkpoint.pt
```

常用參數：

- `--model`: 使用的 DiT 架構
- `--image-size`: 生成圖片解析度，需和訓練時一致
- `--num_condition`: 兩個條件的類別數，需和訓練時一致
- `--ckpt`: checkpoint 路徑
- `--cfg-scale`: classifier-free guidance scale
- `--num-sampling-steps`: diffusion sampling steps
- `--seed`: random seed

## Condition Settings

`--num_condition 4 2` 代表：

- 第一個條件 `c1` 有 4 個類別
- 第二個條件 `c2` 有 2 個類別

請確認這個數字和 `config.py` 中對應資料集的類別數一致，否則 label index 可能超出 embedding table 範圍。

## Project Structure

```text
models.py          # DiT / CCDT model definition
train.py           # DDP training script
sample.py          # sampling script
dataset.py         # custom image dataset loader
config.py          # dataset condition mappings
diffusion/         # diffusion process utilities
LPIPS/             # LPIPS evaluation code
results/           # training outputs and checkpoints
Sample/            # generated samples
```

## Notes

- `train.py` 會根據 `--data_path` 的資料夾名稱選擇 `config.py` 裡的設定，例如 `CelebA`、`Phison`、`Mnist`。
- 圖片會先透過 Stable Diffusion VAE encode 到 latent space，再送進 DiT 訓練。
- sampling 時需使用和訓練相同的 `--image-size`、`--model`、`--num_condition`。
