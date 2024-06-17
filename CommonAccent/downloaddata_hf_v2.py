import os
import argparse
import csv

from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")

def prepare_cv_from_hf(output_folder, language="en"):
    """Function to prepare the datasets in <output-folder>"""

    output_folder = os.path.join(output_folder, language)
    # Create the output folder if it is not present
    os.makedirs(output_folder, exist_ok=True)

    # Load the Common Voice dataset
    common_voice_ds = load_dataset("mozilla-foundation/common_voice_11_0", language, streaming=False)

    # Just select relevant splits: train/validation/test set
    splits = ["train", "validation", "test"]
    common_voice = {}

    # Load, prepare, and filter each split
    for split in splits:
        # Filter out samples without accent
        ds = common_voice_ds[split].filter(lambda x: x['accent'] is not None and x['accent'] != '')
        common_voice[split] = ds

    for split in common_voice:
        csv_lines = []
        # Starting index
        idx = 0
        for sample in common_voice[split]:
            # Get path and utt_id
            mp3_path = sample['path']
            utt_id = os.path.basename(mp3_path).split(".")[0]

            # Create a row with metadata + transcripts
            csv_line = [
                idx,  # ID
                utt_id,  # Utterance ID
                mp3_path,  # File name
                sample["locale"],
                sample["accent"],
                sample["age"],
                sample["gender"],
                sample["sentence"],  # transcript
            ]

            # Adding this line to the csv_lines list
            csv_lines.append(csv_line)
            # Increment index
            idx += 1

        # CSV column titles
        csv_header = ["idx", "utt_id", "mp3_path", "language", "accent", "age", "gender", "transcript"]
        # Add titles to the list at index 0
        csv_lines.insert(0, csv_header)

        # Writing the csv lines
        csv_file = os.path.join(output_folder, split + '.tsv')

        with open(csv_file, mode="w", encoding="utf-8") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_lines:
                csv_writer.writerow(line)
    print(f"Prepared CommonVoice dataset for {language} in {output_folder}")

def main():
    # Read input from CLI, you need to run it from the command line
    parser = argparse.ArgumentParser()

    # Reporting vars
    parser.add_argument(
        "--language",
        type=str,
        default='en',
        help="Language to load",
    )
    parser.add_argument(
        "output_folder",
        help="Path of the output folder to store the csv files for each split",
    )
    args = parser.parse_args()

    # Call the main function
    prepare_cv_from_hf(output_folder=args.output_folder, language=args.language)

if __name__ == "__main__":
    main()
