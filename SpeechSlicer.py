import os
import json
from natsort import natsorted
from utils import get_answer
from pydub import AudioSegment
from tqdm import tqdm
import torch
import shutil
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import ast
import fnmatch

class SpeechSlicer:
    def __init__(self, args):
        self.input_path = args.input_path
        self.file_type = args.file_type
        self.extension = args.extension
        self.overwrite = args.overwrite
        self.model = args.model
        self.llm_merging = args.llm_merging
        self.subtitle_available = args.subtitle_available
        self.subtitle_path = args.subtitle_path
        self.output_path = args.output_path
        self.padding = args.padding
        self.recursive_depth = args.recursive_depth

        with open("./api_key.txt", "r") as f:
            api_key = f.read().strip("\n\"\'")
        self.api_key = api_key

        self.input_files = []
        self.wav_to_asr = {}
        self.asr_to_slice = {}
        self.last_asr = []

    def set_input_files(self, input_path, extension):
        all_files = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith("." + extension):
                    all_files.append(os.path.join(root, file))
        self.input_files = natsorted(all_files)
    
    def save_last_asr(self, asr_path):
        all_files = []
        for root, dirs, files in os.walk(asr_path):
            for file in files:
                if file.endswith(".json"):
                    all_files.append(os.path.join(root, file))
        self.last_asr = natsorted(all_files)

    def convert_to_wav(self):
        self.set_input_files(self.input_path, self.extension)

        print("Converting videos to wav...")
        if not os.path.exists(os.path.join(self.output_path, "wav")):
            os.makedirs(os.path.join(self.output_path, "wav"))

        for f in tqdm(self.input_files):
            if not self.overwrite:
                if os.path.exists(os.path.join(self.output_path, "wav", '.'.join(os.path.basename(f).split('.')[:-1]) + '.wav')):
                    continue
            convert_cmd = f"ffmpeg -i \"{f}\" -ab 160k -ac 2 -ar 44100 -vn \"{os.path.join(self.output_path, 'wav', '.'.join(os.path.basename(f).split('.')[:-1]) + '.wav')}\""
            os.system(convert_cmd)
        self.input_path = os.path.join(self.output_path, "wav")
        self.extension = "wav"

    def check_single_sentence(self, text):
        end_suffix = [".", "?", "!"]
        for es in end_suffix:
            if text.endswith(es) and text.count(es) == 1:
                return True
        return False

    def perform_asr(self, level):
        print()
        print(f"Performing ASR for the depth {level}")
        if level > 0:
            self.set_input_files(os.path.join(self.output_path, "cut_wav", str(level-1)), self.extension)
        elif self.file_type == "audio":
            self.set_input_files(self.input_path, self.extension)
        else:
            self.set_input_files(os.path.join(self.output_path, "wav"), self.extension)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.wav_to_asr = {}

        for f in tqdm(self.input_files):
            clip_name = ".".join(os.path.basename(f).split(".")[:-1])
            if level > 0:
                movie_name = "_".join(os.path.basename(f).split("_")[:-1])
            else:
                movie_name = clip_name

            self.wav_to_asr[f] = os.path.join(self.output_path, "asr", str(level), movie_name, clip_name + ".json")
            if not self.overwrite:
                if os.path.exists(os.path.join(self.output_path, "asr", str(level), movie_name, clip_name + ".json")):
                    continue

            if not os.path.exists(os.path.join(self.output_path, "asr", str(level), movie_name)):
                os.makedirs(os.path.join(self.output_path, "asr", str(level), movie_name))

            if level > 0:
                
                last_asr = fnmatch.filter(self.last_asr, f"{os.path.join(self.output_path, 'asr', str(level-1), movie_name)}*.json")
                assert len(last_asr) == 1
                index = int(clip_name.split("_")[-1])
                with open(last_asr[0], "r") as l:
                    jl = json.load(l)
                l = jl[index]

                is_single = self.check_single_sentence(l["text"])
                if is_single:
                    if not os.path.exists(os.path.join(self.output_path, "cut_wav", str(level), movie_name)):
                        os.makedirs(os.path.join(self.output_path, "cut_wav", str(level), movie_name))
                    shutil.copy(os.path.join(self.output_path, "cut_wav", str(level-1), movie_name, clip_name + ".wav"), os.path.join(self.output_path, "cut_wav", str(level), movie_name, clip_name + f"_000.{self.extension}"))
                    with open(os.path.join(self.output_path, "asr", str(level), movie_name, clip_name + ".json"), "w") as o:
                        json.dump([l], o, indent=4)

            if level < 1:
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=32,
                    return_timestamps=True,
                    torch_dtype=torch_dtype,
                    device=device,
                )
            else:
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=1,
                    return_timestamps=True,
                    torch_dtype=torch_dtype,
                    device=device,
                )

            result = pipe(f)
            duplicate = [None]
            final = []
            for chunk in result["chunks"]:
                if chunk["text"] == duplicate[-1]:
                    continue
                duplicate.append(chunk["text"])
                final.append(chunk)

            with open(os.path.join(self.output_path, "asr", str(level), movie_name, clip_name + ".json"), "w") as j:
                json.dump(final, j, indent=4)
        self.save_last_asr(os.path.join(self.output_path, "asr", str(level)))

    def cut_wav(self, level, llm_merged=False):
        print()
        print(f"Slicing wav files for depth {level}")
        if level == 0:
            self.set_input_files(self.input_path, "wav")
        else:
            self.set_input_files(os.path.join(self.output_path, "cut_wav", str(level-1)), "wav")
        self.asr_to_slice = {}
        if not os.path.exists(os.path.join(os.path.join(self.output_path, "cut_wav", str(level)))):
            os.makedirs(os.path.join(os.path.join(self.output_path, "cut_wav", str(level))))

        for f in tqdm(self.input_files):
            file_name = os.path.basename(f)
            clip_name = ".".join(file_name.split(".")[:-1])
            if level > 0:
                movie_name = "_".join(os.path.basename(f).split("_")[:-1])
            else:
                movie_name = clip_name
            extension = "." + file_name.split(".")[-1]
            

            if llm_merged:
                if os.path.exists(os.path.join(os.path.join(self.output_path, "cut_wav", str(level), movie_name, clip_name + "_000.wav"))):
                    continue
                
            else:
                if not os.path.exists(os.path.join(os.path.join(self.output_path, "cut_wav", str(level), movie_name))):
                    os.mkdir(os.path.join(os.path.join(self.output_path, "cut_wav", str(level), movie_name)))

            
            self.asr_to_slice[self.wav_to_asr[f]] = []

            if llm_merged:
                with open(self.wav_to_asr[f].replace(f"/asr/{level}/", "/llm_merged/"), "r") as g:
                    j = json.load(g)
            else:
                with open(self.wav_to_asr[f], "r") as g:
                    j = json.load(g)
                
            if len(j) == 1:
                shutil.copy(f, os.path.join(self.output_path, "cut_wav", str(level), movie_name, clip_name + "_000" + extension))
                self.asr_to_slice[self.wav_to_asr[f]].append(os.path.join(self.output_path, "cut_wav", str(level), movie_name, clip_name + "_000" + extension))
                continue
            audio = AudioSegment.from_file(f)
            for tidx, _ in enumerate(j):
                if j[tidx]["timestamp"][0]:
                    j[tidx]["timestamp"][0] += 0.25
                if j[tidx]["timestamp"][1]:
                    j[tidx]["timestamp"][1] += 0.25
            for idx, a in enumerate(j):
                if not self.overwrite and os.path.exists(os.path.join(os.path.join(self.output_path, "cut_wav", str(level), movie_name, clip_name + f"_{idx:03d}.wav"))):
                    continue

                start = a["timestamp"][0]
                end = a["timestamp"][1]
                text = a["text"]
                if start != 0 and not start:
                    start = j[idx-1]["timestamp"][1] + self.padding
                if not end:
                    end = len(audio)-1

                if idx+1 != len(j):
                    if not j[idx+1]["timestamp"][0]:
                        end = end + self.padding
                    else:
                        if j[idx+1]["timestamp"][0] - end < self.padding:
                            end = end + (j[idx+1]["timestamp"][0] - end) / 2
                        else:
                            end = end + self.padding

                if idx != 0:
                    if start - j[idx-1]["timestamp"][1] < self.padding:
                        start = start - (start - j[idx-1]["timestamp"][1]) / 2
                    else:
                        start = start - self.padding

                if (end*1000 - 1) >= len(audio):
                    end = (len(audio)-1)/1000
                if (start*1000) < 0:
                    start = 0

                sliced_audio = audio[start*1000:end*1000]
                sliced_audio.export(os.path.join(self.output_path, "cut_wav", str(level), movie_name, clip_name + f"_{idx:03d}" + extension))
                self.asr_to_slice[self.wav_to_asr[f]].append(os.path.join(self.output_path, "cut_wav", str(level), movie_name, clip_name + f"_{idx:03d}" + extension))

    def llm_merge(self):
        print()
        print("Performing LLM merge...")

        if not os.path.exists(os.path.join(self.output_path, "llm_merged")):
            os.mkdir(os.path.join(self.output_path, "llm_merged"))

        for f in tqdm(self.last_asr):
            new_j = []
            file_name = os.path.basename(f)
            clip_name = ".".join(file_name.split(".")[:-1])
            if self.recursive_depth > 1:
                movie_name = "_".join(os.path.basename(f).split("_")[:-1])
            else:
                movie_name = clip_name
            extension = "." + file_name.split(".")[-1]

            if not self.overwrite:
                if os.path.exists(os.path.join(self.output_path, "llm_merged", movie_name, file_name)):
                    continue

            if not os.path.exists(os.path.join(self.output_path, "llm_merged", movie_name)):
                os.mkdir(os.path.join(os.path.join(self.output_path, "llm_merged", movie_name)))
            
            if os.path.exists(os.path.join(self.output_path, "cut_wav", str(self.recursive_depth-1), movie_name, clip_name + "_000.wav")):
                shutil.copy(f, os.path.join(self.output_path, "llm_merged", movie_name, clip_name + ".json"))
                continue

            with open(f, "r") as g:
                j = json.load(g)
            
            if len(j) == 1:
                new_j = j
            else:
                numbered_text = ""
                for aidx, a in enumerate(j):
                    numbered_text += f"{aidx+1}. {a['text'].strip()}\n"
                system_prompt = "You are a professional movie scenario editor. You will be given with the utterances of the characters. Some of the utterances are done by a single character, but splitted into multiple lines. Merge the sentences if they seem to be splitted unnecessarily. Your answer must be a list of lists, which includes utterances to merge, like [[1], [2], ..., [19, 20, 21], [22, 23]]. The utterances that need not to be merged must be in a single-element list. Show only the list of lists as your output."
                prompt = numbered_text

                response = get_answer(prompt, system_prompt=system_prompt, api_key=self.api_key).strip()

                while True:
                    try:
                        response_list = ast.literal_eval(response)
                        break
                    except:
                        print("Formatting error...")
                        continue

                idx_cnt = 0
                for r in response_list:
                    if len(r) < 2:
                        new_j.append(j[idx_cnt])
                        idx_cnt += 1
                    else:
                        newr = {"timestamp":[], "text": ""}
                        newr["timestamp"] = [j[r[0]-1]["timestamp"][0], j[r[-1]-1]["timestamp"][-1]]
                        for k in r:
                            newr["text"] += j[k-1]["text"]
                            
                        new_j.append(newr)
                        idx_cnt += len(r)

            with open(os.path.join(self.output_path, "llm_merged", movie_name, file_name), "w") as f:
                json.dump(new_j, f, indent=4)

    def slice(self):
        if self.file_type == "video":
            self.convert_to_wav()
        
        for i in range(0, self.recursive_depth):
            self.perform_asr(i)

            if i + 1 != self.recursive_depth:
                self.cut_wav(i)

        self.llm_merge()
        self.cut_wav(i, llm_merged=True)

        
