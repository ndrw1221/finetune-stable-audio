import os
import json
import math
import torch.nn.functional as F
import itertools
import torch
import gc
import soundfile as sf
import shutil
import warnings
import pytz
import random
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from diffusers import StableAudioPipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from diffusers.utils import is_wandb_available

tz = pytz.timezone("Asia/Taipei")
now = datetime.now(tz)
now_str = now.strftime("%Y-%m-%d %H:%M:%S")

config = {
    "debug": False,
    # File paths and directories
    # "meta_data_path": "pili_no_vocals_47s.json",
    # "meta_data_path": "pili_no_vocals_47s_qwen_pili.json",
    "meta_data_path": "pili_no_vocals_47s_pili.json",
    "audio_data_root": "/mnt/gestalt/home/ndrw1221/datasets/pili/pili_no_vocals_47s",
    "audio_latent_root": "/mnt/gestalt/home/ndrw1221/datasets/pili/pili_no_vocals_47s_latent",
    "output_dir": f"/mnt/gestalt/home/ndrw1221/sao_pili-output/output/run_pili_only_{now_str}",
    # "output_dir": f"/mnt/gestalt/home/ndrw1221/sao_pili-output/output/qwen_pili_{now_str}",
    # "output_dir": f"/mnt/gestalt/home/ndrw1221/sao_pili-output/output/full_finetune_{now_str}",
    "wand_run_name": f"run_pili_only_{now_str}",
    # "wand_run_name": f"qwen_pili_{now_str}",
    # "wand_run_name": f"full_finetune_{now_str}",
    # Model and training parameters
    "val_ratio": 0.1,
    "train_batch_size": 6,
    "val_batch_size": 5,
    "dataloader_num_workers": 4,
    "learning_rate": 5e-6,
    "weight_decay": 1e-3,
    "lr_scheduler": "polynomial",
    "num_cycles": 10,
    "num_warmup_steps": 0,
    "gradient_accumulation_steps": 16,
    "max_train_steps": 10000,
    "num_train_epochs": 20,
    "checkpointing_steps": 1000,
    "validation_steps": 1000,
    "use_lora": True,
    "lora_r": 32,
    "lora_alpha": 96,
    # Pipeline parameters
    "sigma_min": 0.3,
    "sigma_max": 500,
    # Validation parameters
    "denoise_step": 100,
    #     "resume_from_checkpoint": "/home/ndrw1221/lab-projects/finetune-stable-audio/output/run_2025-05-28 09:21:10/checkpoint-4000",
}


class AudioInversionDataset(Dataset):
    def __init__(
        self,
        config,
        audio_latent_root,
        audio_data_root,
        device,
    ):
        self.audio_data_root = audio_data_root
        self.audio_latent_root = audio_latent_root
        self.device = device
        self.meta_path = config["meta_data_path"]
        with open(self.meta_path) as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        meta_entry = self.meta[i]
        audio_path = meta_entry.get("path")

        # Load audio tokens, they are encoded with the Stable-audio VAE and saved, skipping the the VAE encoding process saves memory when training MuseControlLite
        audio_full_path = os.path.join(self.audio_data_root, audio_path)
        audio_token_path = os.path.join(
            self.audio_latent_root, audio_path.replace("wav", "pth")
        )
        audio = torch.load(audio_token_path, map_location=torch.device("cpu"))

        example = {
            "text": meta_entry["caption"],
            "audio_full_path": audio_full_path,
            "audio": audio,
            "seconds_start": 0,
            "seconds_end": 2097152 / 44100,
        }
        return example


def collate_fn(examples):
    audio = [example["audio"] for example in examples]
    prompt_texts = [example["text"] for example in examples]
    audio_full_path = [example["audio_full_path"] for example in examples]
    seconds_start = [example["seconds_start"] for example in examples]
    seconds_end = [example["seconds_end"] for example in examples]
    audio = torch.stack(audio)
    batch = {
        "audio_full_path": audio_full_path,
        "audio": audio,
        "prompt_texts": prompt_texts,
        "seconds_start": seconds_start,
        "seconds_end": seconds_end,
    }
    return batch


def print_trainable_parameters(model, verbose=True):
    """
    Print the number and percentage of trainable parameters in the model.
    """

    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}, shape={param.shape}, dtype={param.dtype}")
            else:
                print(f"Frozen: {name}, shape={param.shape}, dtype={param.dtype}")
        print("==" * 50)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    print(
        f"Total parameters: {total_params} || Trainable parameters: {trainable_params} || Trainable %: {trainable_percentage:.2f}%"
    )
    print("==" * 50)


def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def log_validation(val_dataloader, pipeline, config, global_step):
    val_audio_dir = os.path.join(config["output_dir"], f"val_audio_{global_step}")
    os.makedirs(val_audio_dir, exist_ok=True)
    description_path = os.path.join(val_audio_dir, "description.txt")

    # Only generate from 4 random batches
    val_examples = list(val_dataloader)
    random.shuffle(val_examples)
    val_examples = val_examples[:4]

    pipeline.transformer.eval()
    with open(description_path, "a") as desc_file:
        for step, batch in enumerate(val_examples):
            prompt_texts = batch["prompt_texts"]
            audio_full_path = batch["audio_full_path"]
            generator = torch.Generator("cuda").manual_seed(0)
            bsz = len(prompt_texts)

            with torch.no_grad():
                audio = pipeline(
                    prompt=prompt_texts,
                    negative_prompt=[""] * bsz,
                    num_inference_steps=config["denoise_step"],
                    audio_end_in_s=2097152 / 44100,
                    num_waveforms_per_prompt=1,
                    generator=generator,
                ).audios

            for i in range(bsz):
                idx = step * bsz + i
                output = audio[i].T.float().cpu().numpy()
                gen_file = os.path.join(val_audio_dir, f"validation_{idx}.wav")
                original_file = os.path.join(val_audio_dir, f"original_{idx}.wav")
                sf.write(gen_file, output, pipeline.vae.sampling_rate)
                shutil.copy(audio_full_path[i], original_file)
                desc_file.write(f"[{idx}] {prompt_texts[i]}\n")

            print(f"[Step {step}] Saved {bsz} samples to {val_audio_dir}")

    torch.cuda.empty_cache()
    gc.collect()


def main():
    ####### Set up the accelerator and device ########
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        log_with="wandb",
    )

    if not is_wandb_available():
        raise ImportError(
            "Make sure to install wandb if you want to use it for logging during training."
        )

    ################# Load the model and prepare it for training ################
    pipeline = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16
    )
    pipeline = pipeline.to(device)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.projection_model.requires_grad_(False)

    noise_scheduler = pipeline.scheduler
    # noise_scheduler.config.sigma_max = config["sigma_max"]
    # noise_scheduler.config.sigma_min = config["sigma_min"]
    transformer = pipeline.transformer

    if config["use_lora"]:
        print("Using LoRA for training")
        transformer_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )

        transformer = get_peft_model(transformer, transformer_config)
    else:
        print("Not using LoRA, training the full transformer model")
        transformer.requires_grad_(True)

    ######### Initialize the optimizer, dataset, dataloader, and lr_scheduler #########
    optimizer = AdamW(
        itertools.chain(transformer.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    dataset = AudioInversionDataset(
        config=config,
        audio_data_root=config["audio_data_root"],
        audio_latent_root=config["audio_latent_root"],
        device=device,
    )

    val_size = int(config["val_ratio"] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
    )

    lr_scheduler = get_scheduler(
        config["lr_scheduler"],
        optimizer=optimizer,
        step_rules=None,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=config["max_train_steps"],
        num_cycles=config["num_cycles"],
    )

    ############### Prepare everything with accelerator ################
    transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            transformer, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    )

    ############### Recalculate number of epochs ################
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config["gradient_accumulation_steps"]
    )

    if config["max_train_steps"] is None:
        config["max_train_steps"] = (
            config["num_train_epochs"] * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True
    else:
        overrode_max_train_steps = False

    if overrode_max_train_steps:
        config["max_train_steps"] = (
            config["num_train_epochs"] * num_update_steps_per_epoch
        )
    config["num_train_epochs"] = math.ceil(
        config["max_train_steps"] / num_update_steps_per_epoch
    )

    ######################### Resume from checkpoint logic ########################
    resume_checkpoint = config.get("resume_from_checkpoint")
    if resume_checkpoint and os.path.isdir(resume_checkpoint):
        ckpt_step = int(os.path.basename(resume_checkpoint).split("-")[-1])
        global_step = ckpt_step
        accelerator.load_state(resume_checkpoint)

        wandb_id_path = os.path.join(os.path.dirname(resume_checkpoint), "wandb_id.txt")
        wandb_id = open(wandb_id_path).read().strip()

        # config["output_dir"] = os.path.dirname(resume_checkpoint)
        # config["wand_run_name"] = os.path.basename(config["output_dir"])
        print(f"Resuming from checkpoint: {resume_checkpoint} at step {global_step}")
    else:
        global_step = 0
        wandb_id = None
        wandb_id_dir = config["output_dir"]

    ######################### Initialize trackers #########################
    if accelerator.is_main_process:
        if wandb_id is None:
            import wandb, uuid

            wandb_id = uuid.uuid4().hex

        accelerator.init_trackers(
            project_name="stable-audio-pili",
            config=config,
            init_kwargs={
                "wandb": {
                    "name": config["wand_run_name"],
                    # "resume": "allow" if global_step == 0 else "must",
                    # "id": wandb_id,
                }
            },
        )

        if global_step == 0:
            os.makedirs(wandb_id_dir, exist_ok=True)
            with open(os.path.join(wandb_id_dir, "wandb_id.txt"), "w") as f:
                f.write(wandb_id)

    progress_bar = tqdm(
        range(global_step, config["max_train_steps"]),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    ################# Log First #################
    if not config["debug"] and not resume_checkpoint:
        if accelerator.is_main_process:
            log_validation(
                val_dataloader,
                pipeline,
                config,
                global_step,
            )
        accelerator.wait_for_everyone()

    ################# Training loop #################
    for epoch in range(config["num_train_epochs"]):
        # Only run the first epoch
        for step, batch in enumerate(train_dataloader):
            transformer.train()
            with accelerator.accumulate(transformer):
                latents = batch["audio"].to(device)
                prompt_texts = batch["prompt_texts"]
                audio_start_in_s = batch["seconds_start"]
                audio_end_in_s = batch["seconds_end"]
                bsz = latents.shape[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    # print("Using epsilon prediction")
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    # print("Using v prediction")
                    # target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    alphas, sigmas = get_alphas_sigmas(timesteps)
                    alphas = alphas[:, None, None]  # Shape to match latents
                    sigmas = sigmas[:, None, None]
                    target = alphas * noise - sigmas * latents

                with torch.no_grad():
                    prompt_embeds = pipeline.encode_prompt(
                        prompt=prompt_texts,
                        device="cuda",
                        do_classifier_free_guidance=False,
                    )
                    # Encode duration
                    seconds_start_hidden_states, seconds_end_hidden_states = (
                        pipeline.encode_duration(
                            audio_start_in_s,
                            audio_end_in_s,
                            device="cuda",
                            do_classifier_free_guidance=False,
                            batch_size=bsz,
                        )
                    )
                audio_duration_embeds = torch.cat(
                    [seconds_start_hidden_states, seconds_end_hidden_states], dim=2
                )
                encoder_hidden_states = torch.cat(
                    [
                        prompt_embeds,
                        seconds_start_hidden_states,
                        seconds_end_hidden_states,
                    ],
                    dim=1,
                )
                with accelerator.autocast():
                    model_pred = pipeline.transformer(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        global_hidden_states=audio_duration_embeds,
                        return_dict=False,
                    )[0]
                    target = target.to(dtype=model_pred.dtype)
                    # Compute the loss
                    loss = F.mse_loss(model_pred, target, reduction="mean")

                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(
                        transformer.parameters(),
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    gc.collect()
                    torch.cuda.empty_cache()

            if accelerator.sync_gradients:
                # audios = []
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % config["checkpointing_steps"] == 0:
                        save_dir = os.path.join(
                            config["output_dir"], f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_dir)

                    if global_step % config["validation_steps"] == 0:
                        log_validation(
                            val_dataloader,
                            pipeline,
                            config,
                            global_step,
                        )

            logs = {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= config["max_train_steps"]:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    print("Training complete.")


if __name__ == "__main__":
    main()
