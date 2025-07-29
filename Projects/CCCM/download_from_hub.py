import shutil
import argparse
import os
from huggingface_hub import hf_hub_download


def main(args):
    download_path = hf_hub_download(
        repo_id="bobolai/StepFusionCCCM",
        filename=args.file
    )

    dest_path = os.path.join(args.dest, args.file)
    shutil.copy(download_path, dest_path)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file", type=str, help="file you want to download from hg StepFusionCCCM repo")
    parser.add_argument('-d', "--dest", type=str, help="local path you want your downloaded file to be saved.")

    args = parser.parse_args()

    main(args)
