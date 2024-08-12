# Speech Slicing

A repository for extracting setence-by-sentence speech files from audio files or video files. This repository utilizes [Whisper V3](https://huggingface.co/openai/whisper-large-v3) and GPT-4o-mini to extract speech files.

-------------------------------------------------------------------------------------------

![Overview1](https://github.com/neosapience/SpeechSlicer/blob/main/overview_1.png?raw=true)
-------------------------------------------------------------------------------------------
![Overview2](https://github.com/neosapience/SpeechSlicer/blob/main/overview_2.png?raw=true)
-------------------------------------------------------------------------------------------
<p align="center">
  <img src="https://github.com/neosapience/SpeechSlicer/blob/main/overview_3.png?raw=true" alt="Overview3">
</p>


## Usage

### Environment

Install torch 2.0+ compatible to your environment, e.g.:

```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
or
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

Install `ffmpeg`. If you are using conda, run
```
conda install conda-forge::ffmpeg
```
Install the other dependencies:
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

## Implementations

- [X] Audio Slicing with Sliding
- [X] Recursive Whisper
- [X] LLM Merging
- [ ] Postprocessing with length and VAD 
