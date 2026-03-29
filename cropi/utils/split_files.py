import json
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description="Split a JSONL file into N shards.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL file")
    parser.add_argument("--split_num", required=True, type=int, help="Number of output shards")
    return parser.parse_args()


def split_jsonl(input_file: Path, split_num: int) -> list[Path]:
    if split_num <= 0:
        raise ValueError("--split_num must be a positive integer")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    split_size = len(data) // split_num
    output_files: list[Path] = []
    for i in range(split_num):
        if i == split_num - 1:
            chunk = data[i * split_size :]
        else:
            chunk = data[i * split_size : (i + 1) * split_size]

        output_path = Path(f"{input_file}.{i}")
        with output_path.open("w", encoding="utf-8") as f:
            for record in chunk:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_files.append(output_path)

    return output_files


def main():
    args = parse_args()
    output_files = split_jsonl(args.input, args.split_num)
    print(f"Wrote {len(output_files)} shard(s) for {args.input}.")


if __name__ == "__main__":
    main()
