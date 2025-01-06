# Robust Single-Image Dehazing

This repo is a ...

## Testing a model:

### 1. Base model:
```bash
python test.py --model <model_name> 
```

or

```bash
python test_adversarial.py --model <model_name> --attack_config <path_to_attack_config>
```
where `model_name` is the checkpoint name placed in `./saved_models/base`. All attack configs are placed in `./configs/attacks`.


### 2. Fine-tuned model:
```bash
python test.py --model <model_name> --fine_tuned
```

or 

```bash
python test.py --model <model_name> --fine_tuned --attack_config
```
where `model_name` is the folder name placed in `./saved_models/fine_tuned`.

**Note**: the fine-tuned checkpoint in the corresponding folder should be named `fine_tuned.ckpt`.

## Fine-tuning a model:
```bash
python tune.py <fine_tuning_config>
```
where `fine_tuning_config` is the name of a `.json` configuration file from `./configs/finetune`.

---

## Acknowledgments 

- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [DehazeFormer](https://github.com/IDKiro/DehazeFormer?tab=readme-ov-file), for open-sourcing their project and dataset
- [ICCV-2023-MB-TaylorFormer](https://github.com/FVL2020/ICCV-2023-MB-TaylorFormer/tree/main), for open-sourcing their project