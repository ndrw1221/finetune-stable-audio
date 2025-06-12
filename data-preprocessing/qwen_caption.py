import librosa
import torch
import json
import os

from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm


# -------------------------
# Function to construct the chat-style input for the model
# -------------------------
def build_conversation(audio_path, prompt):
    """
    Construct a chat conversation format required by Qwen2 Audio model.

    Args:
        audio_path (str): Path to the audio file.
        prompt (str): Text prompt for the model.

    Returns:
        list: Chat-style conversation dictionary.
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant for music captioning.",
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]


# -------------------------
# Main Inference Pipeline
# -------------------------
def main():
    # Load model and processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16
    )

    # Set paths
    audio_dir = "/mnt/gestalt/home/ndrw1221/datasets/pili/pili_no_vocals_47s"
    meta_json = "../pili_no_vocals_47s.json"
    output_json = "../pili_no_vocals_47s_qwen_pili.json"
    batch_size = 6

    # Prompt used for captioning
    basic_prompt = "Generate a detailed caption for this music piece, primarily focusing on instrumentation, genre, mood, rhythm, and scenario."
    improved_prompt = "Generate a detailed caption in English for this music piece in the Pili style, with focus on Chinese instrumentation, genre, mood, rhythm, and scenario. Mention Pili style in the caption."

    # Load metadata JSON
    with open(meta_json, "r") as f:
        metas = json.load(f)

    # Prepare audio file paths and chat conversations
    audio_paths = [os.path.join(audio_dir, meta["path"]) for meta in metas]
    conversations = [
        build_conversation(audio_path, improved_prompt) for audio_path in audio_paths
    ]

    results = []

    # Process in batches
    for i in tqdm(range(0, len(conversations), batch_size), desc="Processing batches"):
        batch = conversations[i : i + batch_size]

        # Format input conversations into text template
        text_inputs = [
            processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            for conversation in batch
        ]

        # Load and process audio files
        audios = []
        for conversation in batch:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audio_path = ele["audio_url"]
                            audio, _ = librosa.load(
                                audio_path, sr=processor.feature_extractor.sampling_rate
                            )
                            audios.append(audio)

        # Tokenize and move inputs to GPU
        inputs = processor(
            text=text_inputs,
            audio=audios,
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        # Generate responses
        generate_ids = model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]  # skip input tokens

        # Decode generated output
        responses = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Store results
        for audio_path, response in zip(audio_paths[i : i + batch_size], responses):
            audio_path = os.path.relpath(audio_path, audio_dir)
            results.append({"path": audio_path, "caption": response.strip()})

    # Write results to output JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
