# Speech Slicing

A repository for extracting setence-by-sentence speech files from audio files or video files. This repository utilizes [Whisper V3](https://huggingface.co/openai/whisper-large-v3) and GPT-4o-mini to extract speech files.

![Overview1](overview1.png)
![Overview2](overview2.png)
![Overview3](overview3.png)

## Usage

### Environment

Install torch 2.0+ compatible to your environment, e.g.:

```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

```
pip install -r requirements.txt
```

### OpenAI API Key

Fill `api_key.txt` with your own OpenAI API Key.

### Inference

```
python slice_speech.py --input_path ./YOUR_INPUT_PATH --extension YOUR_FILES_EXTENSION
```

For further details, check the help messages by `python slice_speech.py --help`.

### Implementations

[X] Audio Slicing with Sliding

[X] Recursive Whisper

[X] LLM Merging

[ ] Postprocessing with length and VAD 
