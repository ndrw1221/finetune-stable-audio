import json
import os


def main():
    datasets = [
        "佾雲_no_vocals",
        "夜神武戲-闇夜降臨(高音質版本) Taiwan Pili glove puppetry soundtrack_no_vocals",
        "陰陽師_no_vocals",
        "霹靂布袋戲-厲神伐天罡 (天之厲氣勢曲)_no_vocals",
        "醫邪 (天不孤角色曲)_no_vocals",
        "傲笑紅塵_no_vocals",
        "九絕陣_no_vocals",
    ]

    metadata_path = "../dset/pili_no_vocals_47s_v*.json"
    output_dir = "../dset"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for dataset_name in datasets:
        filtered_data = [item for item in metadata if dataset_name in item["path"]]

        if filtered_data:
            output_path = os.path.join(output_dir, f"{dataset_name}.json")
            with open(output_path, "w", encoding="utf-8") as out_file:
                json.dump(filtered_data, out_file, ensure_ascii=True, indent=4)
            print(f"Created dataset file: {output_path}")

        else:
            print(f"No data found for dataset: {dataset_name}")


if __name__ == "__main__":
    main()
