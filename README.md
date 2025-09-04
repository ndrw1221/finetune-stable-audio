# LoRA / Personalization of Stable Audio Open

# 訓練資料

請將下載下來的訓練資料解壓縮，並放在與 config 同層、名為的 dataset 的資料夾中。

正確檔案結構應如下：

```text
finetune-stable-audio/
├── ...
├── config/
└── dataset/
    ├── personalization/
    │   └── Yoasobi-Idol/
    │       ├── audio/
    │       ├── latent/
    │       └── Yoasobi-Idol.json
    └── lora/
        └── pili_dataset/
            ├── audio/
            ├── latent/
            └── pili_dataset.json
```

# 環境設置

## Requirements

- Linux Machine
- GPU with VRAM >= 24G
- CUDA (Development done in CUDA 12.6)
- Python (Development done in Python 3.10)
- PyTorch >= 2.5 for Flash Attention and Flex Attention support

## Install

```bash
$ pip install -r requirements.txt
```

# 執行程式

## A. LoRA 微調

```bash
$ accelerate launch --config_file config/accelerate_config.yaml train.py --config config/lora.yaml
```

## B. Personalization 微調

```bash
$ accelerate launch --config_file config/accelerate_config.yaml train.py --config config/personalization.yaml
```