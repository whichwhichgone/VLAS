"""
Flask-based Demo Server for VLA (Vision-Language-Action) Model Deployment

This file implements a Flask-based server that deploys a pre-trained VLA model for robot action prediction.
The server receives visual inputs (static and gripper camera images), text instructions, and robot states
from clients, processes them through the VLA model, and returns predicted robot actions.

Main Components:
- LLMRobotServer: Core class that handles model loading and inference
- Flask endpoints: HTTP API for client-server communication
- Image processing: Handles static and gripper camera images
- Action generation: Converts model outputs to robot actions
"""

from flask import Flask, jsonify, request, Response
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_AUDIO_TOKEN
from llava.mm_utils import tokenizer_image_token
from llava.action_tokenizer import ActionTokenizer, encode_robot_obs
from llava import conversation as conversation_lib

import argparse
import os
import socket
import io
import json
import numpy as np
import torch
from PIL import Image
from functools import partial


TARGET_IMG_SIZE = 336  # NOTE need to be consistent with that in calvin2json.py


class LLMRobotServer:
    def __init__(self, args):
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = args.model_base
        self.tokenizer, self.llm_robot, self.image_processor, self.context_len = (
            load_pretrained_model(model_path, model_base, model_name)
        )
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.num_beams = args.num_beams
        self.max_new_tokens = args.max_new_tokens
        self.action_tokenizer = ActionTokenizer(self.tokenizer)
        self.action_stat = args.action_stat

    def compose_robot_input(
        self, img_static, img_gripper, instruction, robot_obs, debug=True
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

        image_tensor = self.image_processor.preprocess(img_concat, return_tensors="pt")[
            "pixel_values"
        ][0]
        image_tensor = image_tensor[None, :]
        robot_obs = [str(elem) for elem in robot_obs]
        robot_obs = " ".join(robot_obs)
        robot_obs = encode_robot_obs(robot_obs, self.action_tokenizer, self.action_stat)

        instruction = DEFAULT_IMAGE_TOKEN + "\n" + instruction + "\n" + robot_obs
        conv = conversation_lib.default_conversation.copy()
        conv.system = (
            "A chat between a curious user and an artificial intelligence robot. "
            "The robot provides actions to follow out the user's instructions."
        )
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        instruction = conv.get_prompt()

        input_ids = torch.stack(
            [tokenizer_image_token(instruction, self.tokenizer, return_tensors="pt")],
            dim=0,
        )
        return input_ids, image_tensor

    def robot_action_generate(self, input_ids, images):
        """_summary_

        Args:
            input_ids : shape of (1, L)
            images : shape of (1, C, H, W)

        Returns:
            _type_: _description_
        """
        with torch.inference_mode():
            output_ids = self.llm_robot.generate(
                input_ids.cuda(),
                images=images.to(dtype=torch.float16, device="cuda", non_blocking=True),
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
        default="llava-v1.5-7b-calvin-rel-obs-reduce5-v2-abcd2d",
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
        default="/wangdonglin/calvin/task_ABCD_D/training/statistics.yaml",
    )
    parser.add_argument("--port", type=int, default=9002)
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

            input_ids, images = llm_robot.compose_robot_input(
                img_static, img_gripper, instruction, robot_obs
            )
            action = llm_robot.robot_action_generate(input_ids, images)
            return jsonify(action.tolist())

    flask_app.run(host="0.0.0.0", port=args.port)
