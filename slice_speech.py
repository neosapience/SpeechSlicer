import argparse
from SpeechSlicer import SpeechSlicer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Parent path where the clips are saved in.")
    parser.add_argument("--file_type", type=str, default="video", choices=["video", "audio"], help="The type of the files before processing, \"video\" or \"audio\". Must correspond to the `extension` option value. The default value is \"video\".")
    parser.add_argument("--extension", type=str, default="mkv", help="The extension of the files before processing, like \"mkv\" or \"wav\". Must correspond to the `file_type` option value. The default value is \"mkv\".")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for LLM merging")
    parser.add_argument("--llm_merging", default=False, action="store_true")
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--padding", type=float, default=0.5, help="Maximum length of the padding to be added")
    parser.add_argument("--recursive_depth", type=int, default=2, help="How many times whisper will run. 2 is recommended.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    slicer = SpeechSlicer(args)
    slicer.slice()
