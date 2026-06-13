import json
import random
import shutil
from pathlib import Path
import re

# =========================
# 可修改参数
# =========================
RATIOS = [1, 5, 10]   # 表示 1:1, 1:5, 1:10
SEED = 42

# 脚本位于 data_balance/ 下
BASE_DIR = Path(__file__).resolve().parent
SRC_ROOT = BASE_DIR.parent / "data_balanced"   # 输入目录
DST_ROOT = BASE_DIR                            # 输出目录（就是 data_balance）


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def split_pos_neg(samples):
    pos = [x for x in samples if str(x.get("output")) == "1"]
    neg = [x for x in samples if str(x.get("output")) == "0"]
    return pos, neg


def subsample_train(samples, ratio, seed=42):
    """
    ratio=5 表示正负样本比例 1:5
    """
    rng = random.Random(seed)

    pos, neg = split_pos_neg(samples)

    if len(pos) == 0:
        raise ValueError("未找到正样本（output == '1'）。")

    target_neg_num = len(pos) * ratio
    if target_neg_num > len(neg):
        raise ValueError(
            f"负样本不足：需要 {target_neg_num} 个，但只有 {len(neg)} 个。"
        )

    sampled_neg = rng.sample(neg, target_neg_num)
    new_samples = pos + sampled_neg
    rng.shuffle(new_samples)
    return new_samples, len(pos), len(neg), target_neg_num


def parse_filename(filename: str):
    """
    解析类似：
    bigvul_0-512_train.json
    bigvul_512-1024_validate.json

    返回：
    dataset_name, length_part, split
    """
    m = re.match(r"^(.*?)_(\d+-\d+)_(train|validate|test)\.json$", filename)
    if not m:
        return None
    dataset_name = m.group(1)
    length_part = m.group(2)
    split = m.group(3)
    return dataset_name, length_part, split


def collect_alpaca_groups(alpaca_dir: Path):
    """
    把一个 alpaca 目录中的文件按 (dataset_name, length_part) 分组
    """
    groups = {}

    for file_path in alpaca_dir.glob("*.json"):
        parsed = parse_filename(file_path.name)
        if parsed is None:
            print(f"[跳过] 文件名不符合规则: {file_path}")
            continue

        dataset_name, length_part, split = parsed
        key = (dataset_name, length_part)

        if key not in groups:
            groups[key] = {}

        groups[key][split] = file_path

    return groups


def process_one_group(dataset_dir_name: str, alpaca_dir: Path, dataset_name: str, length_part: str, files: dict):
    """
    处理一组 train/validate/test
    """
    required = ["train", "validate", "test"]
    for s in required:
        if s not in files:
            print(f"[跳过] {dataset_name}_{length_part} 缺少 {s} 文件")
            return

    train_path = files["train"]
    validate_path = files["validate"]
    test_path = files["test"]

    print(f"\n处理数据集: {dataset_name}, 长度: {length_part}")
    print(f"  train: {train_path.name}")
    print(f"  validate: {validate_path.name}")
    print(f"  test: {test_path.name}")

    train_samples = load_json(train_path)

    for ratio in RATIOS:
        try:
            new_train, pos_num, neg_num, used_neg_num = subsample_train(
                train_samples, ratio, seed=SEED
            )
        except ValueError as e:
            print(f"  [跳过 1:{ratio}] {e}")
            continue

        # 输出目录：data_balance/数据集名/alpaca/
        out_dir = DST_ROOT / dataset_dir_name / "alpaca"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 保留长度，避免不同长度文件互相覆盖
        train_out = out_dir / f"{dataset_name}_{length_part}_{ratio}_train.json"
        validate_out = out_dir / f"{dataset_name}_{length_part}_{ratio}_validate.json"
        test_out = out_dir / f"{dataset_name}_{length_part}_{ratio}_test.json"

        save_json(new_train, train_out)
        shutil.copy2(validate_path, validate_out)
        shutil.copy2(test_path, test_out)

        print(
            f"  [完成 1:{ratio}] "
            f"pos={pos_num}, original_neg={neg_num}, sampled_neg={used_neg_num}"
        )
        print(f"    -> {train_out.relative_to(DST_ROOT)}")
        print(f"    -> {validate_out.relative_to(DST_ROOT)}")
        print(f"    -> {test_out.relative_to(DST_ROOT)}")


def main():
    if not SRC_ROOT.exists():
        print(f"错误：源目录不存在：{SRC_ROOT}")
        return

    dataset_dirs = [p for p in SRC_ROOT.iterdir() if p.is_dir()]

    if not dataset_dirs:
        print(f"错误：在 {SRC_ROOT} 下没有找到任何数据集文件夹。")
        return

    print(f"源目录: {SRC_ROOT}")
    print(f"输出目录: {DST_ROOT}")
    print(f"比例列表: {RATIOS}")
    print(f"随机种子: {SEED}")

    for dataset_dir in dataset_dirs:
        alpaca_dir = dataset_dir / "alpaca"
        if not alpaca_dir.exists():
            print(f"\n[跳过] {dataset_dir.name} 下没有 alpaca 文件夹")
            continue

        print(f"\n========== 进入数据集目录: {dataset_dir.name} ==========")
        groups = collect_alpaca_groups(alpaca_dir)

        if not groups:
            print(f"[跳过] {alpaca_dir} 中没有符合命名规则的 json 文件")
            continue

        for (dataset_name, length_part), files in groups.items():
            process_one_group(
                dataset_dir_name=dataset_dir.name,
                alpaca_dir=alpaca_dir,
                dataset_name=dataset_name,
                length_part=length_part,
                files=files
            )

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()