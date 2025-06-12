# dataset.py

import os
import json
import torch
from torch.utils.data import Dataset


class AudioInversionDataset(Dataset):
    """
    Dataset for loading pre-computed audio latents and their corresponding text captions.

    Args:
        meta_path (str): Path to the JSON metadata file.
        audio_latent_root (str): Root directory where audio latents (.pth files) are stored.
    """

    def __init__(self, meta_path: str, audio_latent_root: str, audio_data_root: str):
        self.audio_latent_root = audio_latent_root
        self.audio_data_root = audio_data_root

        try:
            with open(meta_path) as f:
                self.meta = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        meta_entry = self.meta[i]
        audio_path = meta_entry.get("path")

        # Construct path to the pre-computed latent tensor
        latent_filename = audio_path.replace(".wav", ".pth")
        latent_full_path = os.path.join(self.audio_latent_root, latent_filename)

        try:
            # Load latent tensor
            audio_latents = torch.load(
                latent_full_path, map_location=torch.device("cpu")
            )
        except FileNotFoundError:
            # Provide a helpful error message if a latent file is missing
            raise FileNotFoundError(
                f"Audio latent file not found for {audio_path} at {latent_full_path}"
            )

        # Path to the original audio for validation logging
        original_audio_path = os.path.join(self.audio_data_root, audio_path)

        example = {
            "text": meta_entry["caption"],
            "latents": audio_latents,
            "original_audio_path": original_audio_path,
            "seconds_start": 0,
            "seconds_total": 47.0,  # Corresponds to 2097152 / 44100
        }
        return example


def collate_fn(examples):
    """
    Custom collate function to batch examples from AudioInversionDataset.
    """
    latents = torch.stack([example["latents"] for example in examples])
    prompt_texts = [example["text"] for example in examples]
    original_audio_paths = [example["original_audio_path"] for example in examples]
    seconds_start = [example["seconds_start"] for example in examples]
    seconds_total = [example["seconds_total"] for example in examples]

    return {
        "latents": latents,
        "prompt_texts": prompt_texts,
        "original_audio_paths": original_audio_paths,
        "seconds_start": seconds_start,
        "seconds_total": seconds_total,
    }
