# utils.py

import os
import gc
import math
import random
import shutil
import logging
import torch
import soundfile as sf
from tqdm import tqdm


def setup_logging(log_level=logging.INFO):
    """Configures the root logger."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_trainable_parameters(model):
    """Logs the number and percentage of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100

    logging.info("=" * 50)
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Trainable %: {trainable_percentage:.2f}%")
    logging.info("=" * 50)


def get_alphas_sigmas(t):
    """
    Returns the scaling factors for the clean image (alpha) and noise (sigma)
    for a given timestep t, used in v-prediction.
    """
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def log_validation(val_dataloader, pipeline, config, accelerator, global_step):
    """
    Runs the validation loop, generating audio samples and logging them to a directory.
    If config.training.val_ratio is 0, uses predefined prompts from config.validation.prompts.

    Args:
        val_dataloader (DataLoader): The validation dataloader. Can be empty.
        pipeline (StableAudioPipeline): The diffusion pipeline.
        config (dict): The configuration dictionary.
        accelerator (Accelerator): The Accelerate object.
        global_step (int): The current training step.
    """
    if not accelerator.is_main_process:
        return

    logging.info(f"Running validation for global step {global_step}...")
    pipeline.transformer.eval()

    val_config = config["validation"]
    output_dir = config["output"]["output_dir"]  # This path is set in train.py

    val_audio_dir = os.path.join(output_dir, f"validation_step_{global_step}")
    os.makedirs(val_audio_dir, exist_ok=True)
    description_path = os.path.join(val_audio_dir, "prompts.txt")

    generator = torch.Generator(device=accelerator.device).manual_seed(42)

    # If val_ratio is 0, use predefined prompts from config
    if config["training"]["val_ratio"] == 0:
        logging.info("Using predefined prompts for validation as val_ratio is 0.")
        prompt_texts = val_config.get("prompts")
        if not prompt_texts:
            logging.warning(
                "`validation.prompts` is not defined in the config. Skipping validation."
            )
            os.rmdir(val_audio_dir)  # Clean up empty dir
            return

        seconds_total = val_config.get("seconds_total")
        if not seconds_total:
            logging.warning(
                "`validation.seconds_total` is not defined in the config. Skipping validation."
            )
            os.rmdir(val_audio_dir)  # Clean up empty dir
            return

        with open(description_path, "w") as desc_file:
            for i, prompt in tqdm(
                enumerate(prompt_texts),
                total=len(prompt_texts),
                desc="Generating validation samples from prompts",
            ):
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.float16):
                        audio = pipeline(
                            prompt=prompt,
                            num_inference_steps=val_config["denoise_steps"],
                            audio_end_in_s=seconds_total,
                            num_waveforms_per_prompt=1,
                            generator=generator,
                        ).audios

                # Save generated audio
                gen_output = audio[0].T.float().cpu().numpy()
                gen_file = os.path.join(val_audio_dir, f"generated_{i}.wav")
                sf.write(gen_file, gen_output, pipeline.vae.sampling_rate)

                desc_file.write(f"[{i}] {prompt}\n")

    # Else, use the validation dataloader
    else:
        val_batches = list(val_dataloader)
        if not val_batches:
            logging.warning(
                "Validation set is empty, but val_ratio > 0. Skipping validation."
            )
            os.rmdir(val_audio_dir)  # Clean up empty dir
            return

        # Get a fixed number of random batches for consistent validation
        random.shuffle(val_batches)
        val_batches_to_log = val_batches[: val_config["num_samples_to_log_per_batch"]]

        with open(description_path, "w") as desc_file:
            for i, batch in tqdm(
                enumerate(val_batches_to_log),
                total=len(val_batches_to_log),
                desc="Generating validation samples from dataloader",
            ):
                prompt_texts = batch["prompt_texts"]
                original_audio_paths = batch["original_audio_paths"]
                bsz = len(prompt_texts)

                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.float16):
                        audio = pipeline(
                            prompt=prompt_texts,
                            num_inference_steps=val_config["denoise_steps"],
                            audio_end_in_s=batch["seconds_total"][
                                0
                            ],  # Assumes all in batch have same duration
                            num_waveforms_per_prompt=1,
                            generator=generator,
                        ).audios

                for j in range(bsz):
                    idx = i * bsz + j
                    # Save generated audio
                    gen_output = audio[j].T.float().cpu().numpy()
                    gen_file = os.path.join(val_audio_dir, f"generated_{idx}.wav")
                    sf.write(gen_file, gen_output, pipeline.vae.sampling_rate)

                    # Copy original audio for comparison
                    original_file = os.path.join(val_audio_dir, f"original_{idx}.wav")
                    shutil.copy(original_audio_paths[j], original_file)

                    desc_file.write(f"[{idx}] {prompt_texts[j]}\n")

    logging.info(f"Saved validation samples to {val_audio_dir}")
    torch.cuda.empty_cache()
    gc.collect()
