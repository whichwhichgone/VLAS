from flask import Flask, jsonify, request, Response
from llava.model.builder import load_pretrained_model_asr
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    tokenizer_image_audio_token,
    process_images,
    get_model_name_from_path,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN
from llava.action_tokenizer_calvin import ActionTokenizer, encode_robot_obs
from llava import conversation as conversation_lib

import argparse
import os
import socket
import io
import json
import yaml
import numpy as np
import librosa
import torch
from PIL import Image
from functools import partial


TARGET_IMG_SIZE = 334  # NOTE need to be consistent with that in calvin2json.py


class LLMRobotServer:
    def __init__(self, args):
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = args.model_base
        (
            self.tokenizer,
            self.llm_robot,
            self.image_processor,
            self.audio_processor,
            self.context_len,
        ) = load_pretrained_model_asr(model_path, model_base, model_name)
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        self.action_stat = args.action_stat

        with open(args.eval_instructions, "r") as f:
            eval_instructions = yaml.safe_load(f)
        self.eval_instruct2id = {
            item_value[0]: idx
            for idx, item_value in enumerate(list(eval_instructions.values()))
        }
        self.eval_audio = args.eval_audio

    def compose_robot_input(
        self,
        img_static,
        img_gripper,
        instruction,
        robot_obs,
        debug=True,
    ):
        img_static = img_static.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_gripper = img_gripper.resize(
            (TARGET_IMG_SIZE, TARGET_IMG_SIZE // 2), Image.LANCZOS
        )
        img_concat = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        img_concat.paste(img_static, (0, 0))
        img_concat.paste(img_gripper, (0, TARGET_IMG_SIZE // 2))

        if debug:
            img_concat.save("./debug_img.png", "PNG")

        # The image height is equal to the width, thus no pad or square
        image_tensor = self.image_processor.preprocess(img_concat, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_tensor = image_tensor[None, :]
        robot_obs = [str(elem) for elem in robot_obs]
        robot_obs = " ".join(robot_obs)
        robot_obs = encode_robot_obs(robot_obs, self.action_tokenizer, self.action_stat)

        if isinstance(instruction, list) and len(instruction) == 2:
            # This is audio mode
            instruction_updated = (
                DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_AUDIO_TOKEN + "\n" + robot_obs
            )
            spk_id = instruction[1]
            audio_id = self.eval_instruct2id[instruction[0]]
            eval_audio = os.path.join(
                self.eval_audio, spk_id, f"{audio_id:04}" + ".wav"
            )
            eval_audio = librosa.load(eval_audio, sr=16000)[0]
            audio_tensor = self.audio_processor(
                eval_audio, sampling_rate=16000, return_tensors="pt"
            ).input_features
        elif isinstance(instruction, list) and len(instruction) == 3:
            # This is voice mode
            task_name = instruction[0]
            spk_id = instruction[1]
            rag_prompt = instruction[2]
            '''
            instruction_updated = (
                DEFAULT_IMAGE_TOKEN + "\n" + rag_prompt + "\n" + DEFAULT_AUDIO_TOKEN + "\n" + robot_obs
            )
            
            audio_id = self.eval_instruct2id[task_name]
            eval_audio = os.path.join(
                self.eval_audio, spk_id, f"{audio_id:04}" + ".wav"
            )
            eval_audio = librosa.load(eval_audio, sr=16000)[0]
            audio_tensor = self.audio_processor(
                eval_audio, sampling_rate=16000, return_tensors="pt"
            ).input_features
            '''

            instruction_updated = DEFAULT_IMAGE_TOKEN + "\n" + rag_prompt + "\n" + task_name + "\n" + robot_obs
            audio_tensor = None
        else:
            # This is normal mode
            instruction_updated = (
                DEFAULT_IMAGE_TOKEN + "\n" + instruction + "\n" + robot_obs
            )
            audio_tensor = None

        conv = conversation_lib.default_conversation.copy()
        conv.system = "A chat between a curious user and an artificial intelligence robot. The robot provides actions to follow out the user's instructions."
        conv.append_message(conv.roles[0], instruction_updated)
        conv.append_message(conv.roles[1], None)
        instruction_updated = conv.get_prompt()

        if audio_tensor is None:
            input_ids = torch.stack(
                [
                    tokenizer_image_token(
                        instruction_updated, self.tokenizer, return_tensors="pt"
                    )
                ],
                dim=0,
            )
        else:
            input_ids = torch.stack(
                [
                    tokenizer_image_audio_token(
                        instruction_updated, self.tokenizer, return_tensors="pt"
                    )
                ],
                dim=0,
            )
        return input_ids, image_tensor, audio_tensor

    def robot_action_generate(self, input_ids, images, audios):
        """_summary_

        Args:
            input_ids : shape of (1, L)
            images : shape of (1, C, H, W)
            audios : shape of (1, C, T)

        Returns:
            _type_: _description_
        """
        with torch.inference_mode():
            output_ids = self.llm_robot.generate(
                input_ids.cuda(),
                images=images.to(dtype=torch.float16, device="cuda", non_blocking=True),
                audios=(
                    audios.to(dtype=torch.float16, device="cuda", non_blocking=True)
                    if audios is not None
                    else audios
                ),
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        # skip the <s> and </s> special token
        output_ids = output_ids[0].cpu().numpy().tolist()[2:-1]
        actions = []
        for elem in output_ids:
            actions.append(self.action_tokenizer.decode_token_ids_to_actions(elem))
        actions = np.array(actions)
        return actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/zhaowei/workspace/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-audio-v1-abc2d",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--action_stat",
        type=str,
        default="/wangdonglin/calvin/task_ABC_D/training/statistics.yaml",
    )
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument(
        "--eval_instructions",
        type=str,
        default="/zhaowei/workspace/LLaVA/playground/calvin_data_audio/eval/new_custom_validation.yaml",
    )
    parser.add_argument(
        "--eval_audio", type=str, default="/zhaowei/data/calvin_eval_audio_real_custom_copy"
    )
    args = parser.parse_args()

    flask_app = Flask(__name__)
    llm_robot = LLMRobotServer(args)

    @flask_app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            img_static = np.frombuffer(
                request.files["img_static"].read(), dtype=np.uint8
            )
            img_static = img_static.reshape((200, 200, 3))
            img_gripper = np.frombuffer(
                request.files["img_gripper"].read(), dtype=np.uint8
            )
            img_gripper = img_gripper.reshape((84, 84, 3))

            content = request.files["json"].read()
            content = json.loads(content)
            instruction = content["instruction"]
            robot_obs = content["robot_obs"]

            img_static = Image.fromarray(img_static)
            img_gripper = Image.fromarray(img_gripper)

            input_ids, images, audios = llm_robot.compose_robot_input(
                img_static, img_gripper, instruction, robot_obs
            )
            action = llm_robot.robot_action_generate(input_ids, images, audios)
            return jsonify(action.tolist())

    flask_app.run(host="0.0.0.0", port=args.port)
