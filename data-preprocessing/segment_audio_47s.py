import os
import glob
import torchaudio
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


def split_audio_file(
    input_path,
    output_folder,
    chunk_shape=(2, 2097152),
    target_sr=44100,
    discard_remainder=False,
):
    """
    Split a WAV file into fixed-size chunks.

    Parameters:
    - input_path (str): Path to the input .wav file.
    - output_folder (str): Base output folder to save chunks.
    - chunk_shape (tuple): Target shape per chunk. Default is (2, 2097152).
    - target_sr (int): Target sampling rate (default: 44100 Hz).
    - discard_remainder (bool): Whether to discard the last incomplete chunk instead of padding.
    """
    try:
        # Load and resample if needed
        waveform, sample_rate = torchaudio.load(input_path)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )
            waveform = resampler(waveform)

        # Ensure stereo (2 channels)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        total_samples = waveform.shape[1]
        chunk_size = chunk_shape[1]
        num_full_chunks = total_samples // chunk_size
        remainder = total_samples % chunk_size

        # Prepare full output waveform
        if discard_remainder:
            padded_waveform = waveform[:, : num_full_chunks * chunk_size]
        else:
            padded_len = (num_full_chunks + (1 if remainder > 0 else 0)) * chunk_size
            padded_waveform = torch.zeros((2, padded_len))
            padded_waveform[:, :total_samples] = waveform

        # Create output dir
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.join(output_folder, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save chunks
        num_chunks = padded_waveform.shape[1] // chunk_size
        for i in range(num_chunks):
            chunk = padded_waveform[:, i * chunk_size : (i + 1) * chunk_size]
            chunk_path = os.path.join(output_dir, f"{base_name}_chunk{i+1}.wav")
            torchaudio.save(chunk_path, chunk, sample_rate=target_sr)

        return f"[✓] {base_name}: {num_chunks} chunk(s)"
    except Exception as e:
        return f"[✗] Failed processing {input_path}: {e}"


def process_folder(
    input_folder, output_folder, discard_remainder=False, num_workers=-1
):
    """
    Process all .wav files in a folder using multiprocessing with a progress bar.

    Parameters:
    - input_folder (str): Directory containing .wav files.
    - output_folder (str): Directory to store chunked files.
    - discard_remainder (bool): Whether to discard final incomplete chunks.
    - num_workers (int): Number of parallel jobs. Default -1 uses all CPUs.
    """
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    if not wav_files:
        print("No .wav files found in the input folder.")
        return

    print(
        f"Found {len(wav_files)} .wav files. Processing with {num_workers if num_workers > 0 else 'all'} workers..."
    )

    # tqdm + joblib trick: wrap delayed calls and print result after collection
    results = Parallel(n_jobs=num_workers)(
        delayed(split_audio_file)(
            wav_file, output_folder, (2, 2097152), 44100, discard_remainder
        )
        for wav_file in tqdm(wav_files, desc="Processing", unit="file")
    )

    # Output summary
    for r in results:
        print(r)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split WAV files into fixed chunks of shape (2, 2097152) ~47s."
    )
    parser.add_argument(
        "input_folder", type=str, help="Folder containing input .wav files"
    )
    parser.add_argument("output_folder", type=str, help="Folder to store output chunks")
    parser.add_argument(
        "--discard_remainder",
        action="store_true",
        help="Discard final incomplete chunk instead of padding with zeros",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="Number of parallel workers (default: use all CPUs)",
    )

    args = parser.parse_args()
    process_folder(
        args.input_folder,
        args.output_folder,
        discard_remainder=args.discard_remainder,
        num_workers=args.num_workers,
    )
