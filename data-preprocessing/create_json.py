import os
import glob
import json

if __name__ == "__main__":
    jsonl_dir = "/home/ndrw1221/nas/datasets/pili/pili_data"
    wav_dir = "/home/ndrw1221/nas/datasets/pili/pili_no_vocals_47s"
    wav_files = glob.glob(os.path.join(wav_dir, "**/*.wav"), recursive=True)

    print(f"Total number of wav files: {len(wav_files)}")

    json_file = []

    for wav_file in wav_files:
        audio_path = os.path.join(*wav_file.split("/")[-2:])
        json_file_name = wav_file.split("/")[-2] + ".jsonl"
        json_file_path = os.path.join(jsonl_dir, json_file_name)

        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                line = f.readline()
                json_data = json.loads(line)
                caption = json_data["description"]

        # Create the JSON object
        json_object = {
            "path": audio_path,
            "caption": caption,
        }

        # Append the JSON object to the list
        json_file.append(json_object)

    # Write the list of JSON objects to a file
    output_json_path = os.path.join(wav_dir, "pili_no_vocals_47s.json")
    with open(output_json_path, "w") as f:
        json.dump(json_file, f, indent=4)
    print(f"JSON file created at: {output_json_path}")
    print(f"Total number of JSON objects: {len(json_file)}")
