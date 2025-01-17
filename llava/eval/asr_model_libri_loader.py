import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import glob
from pathlib import Path

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    AUDIO_TOKEN_INDEX,
    DEFAULT_AUDIO_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, load_pretrained_model_asr
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    tokenizer_audio_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader

import librosa
import math
from jiwer import wer


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, audio_folder, tokenizer, audio_processor, model_config):
        self.audio_folder = audio_folder
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.model_config = model_config
        self.questions = self.get_asr_questions(self.audio_folder)

    def __getitem__(self, index):
        line = self.questions[index]
        audio_path = line["audio_path"]
        ground_truth = line["text"]

        instruction = "Transcribe the following speech into text."
        qs = DEFAULT_AUDIO_TOKEN + "\n" + instruction

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt().strip()

        audio = librosa.load(audio_path, sr=16000)[0]
        audio_tensor = self.audio_processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)

        input_ids = tokenizer_audio_token(
            prompt, self.tokenizer, AUDIO_TOKEN_INDEX, return_tensors="pt"
        )

        return input_ids, audio_tensor, ground_truth

    def get_asr_questions(self, root_audios):
        assert Path(root_audios).exists(), f"root_audios {root_audios} does not exist"
        search_trans = glob.glob(
            os.path.join(root_audios, "**", "*.trans.txt"), recursive=True
        )

        total_lines = []
        for trans in search_trans:
            with open(trans, "r") as f:
                lines = f.readlines()
            total_lines.extend(lines)

        with open("asr_questions.txt", "w") as f:
            f.writelines(total_lines)

        questions = []
        for line in total_lines:
            line = line.strip()
            audio_file = line.split()[0]
            audio_file_parts = audio_file.split("-")
            text = line.split()[1:]
            text = " ".join(text)
            audio_path = os.path.join(
                self.audio_folder,
                audio_file_parts[0],
                audio_file_parts[1],
                audio_file + ".flac",
            )
            assert Path(audio_path).exists(), f"audio_path {audio_path} does not exist"
            questions.append(
                {"audio_file": audio_file, "audio_path": audio_path, "text": text}
            )

        return questions

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, audio_tensors, grounds = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    audio_tensors = torch.stack(audio_tensors, dim=0)
    return input_ids, audio_tensors, grounds


# DataLoader
def create_data_loader(
    audio_folder, tokenizer, audio_processor, model_config, batch_size=1, num_workers=0
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(audio_folder, tokenizer, audio_processor, model_config)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader, dataset


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, audio_processor, context_len = (
        load_pretrained_model_asr(model_path, args.model_base, model_name)
    )

    data_loader, dataset = create_data_loader(
        args.audio_folder, tokenizer, audio_processor, model.config
    )

    src_list = []
    tgt_list = []
    for input_ids, audio_tensor, audio_text in tqdm(data_loader, total=len(dataset)):
        input_ids = input_ids.to(device="cuda", non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                audios=audio_tensor.to(
                    dtype=torch.float16, device="cuda", non_blocking=True
                ),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        src_list.append(audio_text[0])
        tgt_list.append(outputs)
    
    # Computes the WER
    wer_result = wer(src_list, tgt_list)
    print(f"WER: {wer_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--audio-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    eval_model(args)    
