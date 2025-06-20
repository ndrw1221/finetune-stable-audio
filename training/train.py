# train.py

import os
import gc
import math
import yaml
import pytz
import logging
import argparse
import itertools
import warnings
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from tqdm.auto import tqdm
from peft import get_peft_model, LoraConfig
from diffusers import StableAudioPipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from diffusers.utils import is_wandb_available

# Import from our refactored modules
from dataset import AudioInversionDataset, collate_fn
from utils import (
    setup_logging,
    print_trainable_parameters,
    log_validation,
    get_alphas_sigmas,
)

# Set up logger
logger = logging.getLogger(__name__)


def main(config_path):
    # --- 1. Configuration and Setup ---
    setup_logging()
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load config from YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create timestamped output directory
    tz = pytz.timezone("Asia/Taipei")
    now_str = datetime.now(tz).strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"{config['output']['run_name_prefix']}_{now_str}"
    output_dir = os.path.join(config["output"]["output_dir_root"], run_name)
    config["output"][
        "output_dir"
    ] = output_dir  # Store dynamic path back into config for easy access

    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_with="wandb",
    )

    logger.info(
        f"Process rank: {accelerator.process_index}, device: {accelerator.device}"
    )
    logger.info(f"Output directory: {output_dir}")

    # --- 2. Model Loading and Preparation ---
    logger.info("Loading Stable Audio pipeline...")
    pipeline = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
    )
    pipeline.to(accelerator.device)

    # --- Textual Inversion Setup ---
    if config.get("textual_inversion", {}).get("use_textual_inversion"):
        logger.info("Setting up Textual Inversion...")
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder

        ti_config = config["textual_inversion"]
        placeholder_token = ti_config["placeholder_token"]

        # Add new token to tokenizer
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        # Get original embeddings before resizing to calculate stats
        with torch.no_grad():
            original_embeds = text_encoder.get_input_embeddings().weight.clone()
            mean = original_embeds.mean(dim=0)
            std = original_embeds.std(dim=0)

        # Resize token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialize new token embedding
        token_embeds = text_encoder.get_input_embeddings().weight.data
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

        # Initialize with random noise based on existing embeddings' stats
        token_embeds[placeholder_token_id] = torch.normal(mean, std).to(
            token_embeds.dtype
        )

        # Freeze all text_encoder parameters except the new token embeddings
        text_encoder.requires_grad_(False)
        text_encoder.get_input_embeddings().weight.requires_grad = True
        logger.info(
            f"Added new token '{placeholder_token}' and initialized it randomly with the mean and std of existing embeddings."
        )

    # Freeze non-trainable components
    pipeline.vae.requires_grad_(False)
    # Keep text_encoder trainable status as set above for TI
    if not config.get("textual_inversion", {}).get("use_textual_inversion"):
        pipeline.text_encoder.requires_grad_(False)
    pipeline.projection_model.requires_grad_(False)

    transformer = pipeline.transformer

    # Apply LoRA if configured
    if config["lora"]["use_lora"]:
        logger.info("Applying LoRA to the transformer...")
        lora_conf = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["lora_dropout"],
            use_dora=False,
        )
        transformer = get_peft_model(transformer, lora_conf)
    elif config["peft"]["cross_kv"]:
        logger.info(
            "Applying PEFT that only trains W^k and W^v in cross-attention layers..."
        )
        transformer.requires_grad_(False)
        for block in transformer.transformer_blocks:
            block.attn2.to_k.weight.requires_grad = True
            block.attn2.to_v.weight.requires_grad = True
            block.attn2.to_k.weight.data = block.attn2.to_k.weight.data.float()
            block.attn2.to_v.weight.data = block.attn2.to_v.weight.data.float()
    else:
        logger.info("Fine-tuning the full transformer model.")
        transformer.requires_grad_(True)

    print_trainable_parameters(transformer)

    # --- 3. Data Loading ---
    logger.info("Loading and preparing dataset...")
    dataset = AudioInversionDataset(
        meta_path=config["data"]["meta_data_path"],
        audio_latent_root=config["data"]["audio_latent_root"],
        audio_data_root=config["data"]["audio_data_root"],
    )

    val_size = int(config["training"]["val_ratio"] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["dataloader"]["num_workers"],
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["dataloader"]["val_batch_size"],
        collate_fn=collate_fn,
        num_workers=config["dataloader"]["num_workers"],
    )

    # --- 4. Optimizer and LR Scheduler ---
    params_to_optimize = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    if config.get("textual_inversion", {}).get("use_textual_inversion"):
        ti_params = filter(
            lambda p: p.requires_grad, pipeline.text_encoder.parameters()
        )
        params_to_optimize.extend(list(ti_params))
        logger.info("Added textual inversion embeddings to the optimizer.")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config["training"]["gradient_accumulation_steps"]
    )
    max_train_steps = (
        config["training"]["max_train_steps"]
        or config["training"]["num_train_epochs"] * num_update_steps_per_epoch
    )

    lr_scheduler = get_scheduler(
        name=config["lr_scheduler"]["type"],
        optimizer=optimizer,
        num_warmup_steps=config["lr_scheduler"]["num_warmup_steps"]
        * accelerator.num_processes,
        power=config["lr_scheduler"]["power"],
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=config["lr_scheduler"]["num_cycles"],
    )

    # --- 5. Accelerator Preparation ---
    if config.get("textual_inversion", {}).get("use_textual_inversion"):
        (
            transformer,
            pipeline.text_encoder,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            pipeline.text_encoder,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        )
    else:
        (transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler) = (
            accelerator.prepare(
                transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
        )

    # --- 6. Tracking and Resuming ---
    global_step = 0
    if config["training"]["resume_from_checkpoint"]:
        logger.info(
            f"Resuming from checkpoint: {config['training']['resume_from_checkpoint']}"
        )
        accelerator.load_state(config["training"]["resume_from_checkpoint"])
        global_step = int(
            os.path.basename(config["training"]["resume_from_checkpoint"]).split("-")[
                -1
            ]
        )
        logger.info(f"Resumed from step {global_step}")

    if accelerator.is_main_process:
        if is_wandb_available() and config["output"]["use_wandb"]:
            accelerator.init_trackers(
                project_name=config["output"]["wandb_project_name"],
                config=config,
                init_kwargs={"wandb": {"name": run_name}},
            )
        os.makedirs(output_dir, exist_ok=True)

    # --- 7. Training Loop ---
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Training Steps")

    for epoch in range(config["training"]["num_train_epochs"]):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                latents = batch["latents"].to(torch.float16)
                bsz = latents.shape[0]

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    pipeline.scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to latents
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                # Calculate target for v-prediction
                alphas, sigmas = get_alphas_sigmas(
                    timesteps.float()
                    / (pipeline.scheduler.config.num_train_timesteps - 1)
                )
                alphas = alphas[:, None, None]
                sigmas = sigmas[:, None, None]
                target = alphas * noise - sigmas * latents

                # Encode text and duration prompts
                with torch.no_grad():
                    prompt_embeds = pipeline.encode_prompt(
                        prompt=batch["prompt_texts"],
                        device=accelerator.device,
                        do_classifier_free_guidance=False,
                    )
                    seconds_start_hs, seconds_end_hs = pipeline.encode_duration(
                        batch["seconds_start"],
                        batch["seconds_total"],
                        accelerator.device,
                        False,
                        bsz,
                    )
                audio_duration_embeds = torch.cat(
                    [seconds_start_hs, seconds_end_hs], dim=2
                )
                encoder_hidden_states = torch.cat(
                    [prompt_embeds, seconds_start_hs, seconds_end_hs], dim=1
                )

                # Forward pass
                model_pred = transformer(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    global_hidden_states=audio_duration_embeds,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred, target, reduction="mean")

                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # --- 8. Logging and Checkpointing ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    if global_step % config["output"]["checkpointing_steps"] == 0:
                        save_path = os.path.join(
                            output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)

                        # Rename model files for clarity
                        transformer_path = os.path.join(save_path, "model.safetensors")
                        if os.path.exists(transformer_path):
                            os.rename(transformer_path, os.path.join(save_path, "transformer.safetensors"))

                        if config.get("textual_inversion", {}).get("use_textual_inversion"):
                            text_encoder_path = os.path.join(save_path, "model_1.safetensors")
                            if os.path.exists(text_encoder_path):
                                os.rename(text_encoder_path, os.path.join(save_path, "text_encoder.safetensors"))
                        
                        logger.info(f"Saved checkpoint to {save_path}")

                    if global_step % config["output"]["validation_steps"] == 0:
                        log_validation(
                            val_dataloader, pipeline, config, accelerator, global_step
                        )

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune the Stable Audio Open model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
