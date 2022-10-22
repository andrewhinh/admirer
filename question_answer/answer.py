# Imports
import argparse
import itertools
import json
import os
from pathlib import Path
import random
from typing import Union

import numpy as np
from onnxruntime import InferenceSession
import openai
from PIL import Image
import timm
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageSegmentation,
    AutoTokenizer,
    CLIPProcessor,
    pipeline,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
)


# Variables
# Artifact path
artifact_path = Path(__file__).resolve().parent / "artifacts"

# PICa formatting/config
img_id = 100  # Random
question_id = 1005  # Random
n_shot = 16
n_ensemble = 5
similarity_metric = "imagequestion"
coco_path = artifact_path / "coco_annotations"
similarity_path = artifact_path / "coco_clip_new"

# Model setup
transformers_path = artifact_path / "transformers"
onnx_path = artifact_path / "onnx"

# Segmentation model config
resnet_path = transformers_path / "resnet" / "resnet50_a1_0-14fe96d1.pth"
tag_model = transformers_path / "facebook" / "detr-resnet-50-panoptic"
max_length = 16
num_beams = 4

# Caption model config
caption_model = transformers_path / "nlpconnect" / "vit-gpt2-image-captioning"
engine = "text-davinci-002"
caption_type = "vinvl_tag"

# CLIP Encoders config
clip_processor = transformers_path / "openai" / "clip-vit-base-patch16"
clip_onnx = onnx_path / "clip.onnx"


# Helper/main classes
class PICa_OKVQA:
    """
    Question Answering Class
    """

    def __init__(self, caption_info, tag_info, questions, context_idxs, question_features, image_features):
        self.tag_info = tag_info
        self.questions = questions
        # load cached image representation (Coco caption & Tags)
        self.inputtext_dict = self.load_cachetext(caption_info)

        (
            self.traincontext_caption_dict,
            self.traincontext_answer_dict,
            self.traincontext_question_dict,
        ) = self.load_anno(
            "%s/captions_train2014.json" % coco_path,
            "%s/mscoco_train2014_annotations.json" % coco_path,
            "%s/OpenEnded_mscoco_train2014_questions.json" % coco_path,
            None,
        )
        self.train_keys = list(self.traincontext_answer_dict.keys())
        self.load_similarity(context_idxs, question_features, image_features)

    def answer_gen(self):
        _, _, question_dict = self.load_anno(None, None, None, self.questions)

        key = list(question_dict.keys())[0]
        img_key = int(key.split("<->")[0])
        question, caption = (
            question_dict[key],
            self.inputtext_dict[img_key],
        )

        caption_i = caption[
            random.randint(0, len(caption) - 1)
        ]  # select one caption if exists multiple, not true except COCO GT (5)

        pred_answer_list, pred_prob_list = [], []
        context_key_list = self.get_context_keys(
            key,
            similarity_metric,
            n_shot * n_ensemble,
        )

        for repeat in range(n_ensemble):
            # prompt format following GPT-3 QA API
            prompt = "Please answer the question according to the above context.\n===\n"
            for ni in range(n_shot):
                if context_key_list is None:
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                else:
                    context_key = context_key_list[ni + n_shot * repeat]
                img_context_key = int(context_key.split("<->")[0])
                while True:  # make sure get context with valid question and answer
                    if (
                        len(self.traincontext_question_dict[context_key]) != 0
                        and len(self.traincontext_answer_dict[context_key][0]) != 0
                    ):
                        break
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                prompt += (
                    "Context: %s\n===\n"
                    % self.traincontext_caption_dict[img_context_key][
                        random.randint(
                            0,
                            len(self.traincontext_caption_dict[img_context_key]) - 1,
                        )
                    ]
                )
                prompt += "Q: %s\nA: %s\n\n===\n" % (
                    self.traincontext_question_dict[context_key],
                    self.traincontext_answer_dict[context_key][0],
                )
            prompt += "Context: %s\n===\n" % caption_i
            prompt += "Q: %s\nA:" % question
            response = None
            try:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=5,
                    logprobs=1,
                    temperature=0.0,
                    stream=False,
                    stop=["\n", "<|endoftext|>"],
                )
            except Exception as e:
                print(e)
                exit(0)

            plist = []
            for ii in range(len(response["choices"][0]["logprobs"]["tokens"])):
                if response["choices"][0]["logprobs"]["tokens"][ii] == "\n":
                    break
                plist.append(response["choices"][0]["logprobs"]["token_logprobs"][ii])
            pred_answer_list.append(self.process_answer(response["choices"][0]["text"]))
            pred_prob_list.append(sum(plist))
        maxval = -999.0
        for ii in range(len(pred_prob_list)):
            if pred_prob_list[ii] > maxval:
                maxval, pred_answer = pred_prob_list[ii], pred_answer_list[ii]
        return pred_answer

    def rationale(self, answer):
        _, _, question_dict = self.load_anno(None, None, None, self.questions)

        key = list(question_dict.keys())[0]
        img_key = int(key.split("<->")[0])
        question, _ = (
            question_dict[key],
            self.inputtext_dict[img_key],
        )

        pred_answer_list, pred_prob_list = [], []

        for _ in range(n_ensemble):
            # prompt format following GPT-3 QA API
            prompt = ""
            prompt += "Q: %s\n" % question
            prompt += "A: %s\nThis is because" % answer

            response = None
            try:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=10,
                    logprobs=1,
                    temperature=0.0,
                    stream=False,
                    stop=["\n", "<|endoftext|>"],
                )
            except Exception as e:
                print(e)
                exit(0)

            plist = []
            for ii in range(len(response["choices"][0]["logprobs"]["tokens"])):
                if response["choices"][0]["logprobs"]["tokens"][ii] == "\n":
                    break
                plist.append(response["choices"][0]["logprobs"]["token_logprobs"][ii])
            pred_answer_list.append(self.process_answer(response["choices"][0]["text"]))
            pred_prob_list.append(sum(plist))
        maxval = -999.0
        for ii in range(len(pred_prob_list)):
            if pred_prob_list[ii] > maxval:
                maxval, pred_answer = pred_prob_list[ii], pred_answer_list[ii]
        return pred_answer

    def get_context_keys(self, key, metric, n):
        if metric == "question":
            lineid = self.valkey2idx[key]
            similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            index = similarity.argsort()[-n:][::-1]
            return [self.train_idx[str(x)] for x in index]
        elif metric == "imagequestion":
            # combined with Q-similairty (image+question)
            lineid = self.valkey2idx[key]
            question_similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            # end of Q-similairty
            similarity = question_similarity + np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
            index = similarity.argsort()[-n:][::-1]
            return [self.train_idx[str(x)] for x in index]
        else:
            return None

    def load_similarity(self, context_idxs, question_features, image_features):
        val_idx = context_idxs
        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)
        if similarity_metric == "question":
            self.train_feature = np.load("%s/coco_clip_vitb16_train2014_okvqa_question.npy" % similarity_path)
            self.val_feature = question_features
            self.train_idx = json.load(
                open(
                    "%s/okvqa_qa_line2sample_idx_train2014.json" % similarity_path,
                    "r",
                )
            )
        elif similarity_metric == "imagequestion":
            self.train_feature = np.load("%s/coco_clip_vitb16_train2014_okvqa_question.npy" % similarity_path)
            self.val_feature = question_features
            self.train_idx = json.load(
                open(
                    "%s/okvqa_qa_line2sample_idx_train2014.json" % similarity_path,
                    "r",
                )
            )
            self.image_train_feature = np.load(
                "%s/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy" % similarity_path
            )
            self.image_val_feature = image_features

    def load_tags(self):
        tags_dict = {}
        image_id, tags = self.tag_info[0], self.tag_info[1]
        tag_str = ", ".join([x for x in tags])
        tags_dict[image_id] = tag_str
        return tags_dict

    def load_cachetext(self, caption_info):
        if "tag" in caption_type:
            tags_dict = self.load_tags()
        idx = caption_info[0]
        caption = caption_info[1]
        caption_dict = {idx: caption}
        if caption_type == "vinvl_tag":
            caption_dict[idx] += ". " + tags_dict[idx]
        return caption_dict

    def load_anno(self, coco_caption_file, answer_anno_file, question_anno_file, questions):
        if coco_caption_file is not None:
            coco_caption = json.load(open(coco_caption_file, "r"))
            if isinstance(coco_caption, dict):
                coco_caption = coco_caption["annotations"]
        if answer_anno_file is not None:
            answer_anno = json.load(open(answer_anno_file, "r"))
        if question_anno_file is not None:
            question_anno = json.load(open(question_anno_file, "r"))
        else:
            question_anno = questions

        caption_dict = {}
        if coco_caption_file is not None:
            for sample in coco_caption:
                if sample["image_id"] not in caption_dict:
                    caption_dict[sample["image_id"]] = [sample["caption"]]
                else:
                    caption_dict[sample["image_id"]].append(sample["caption"])
        answer_dict = {}
        if answer_anno_file is not None:
            for sample in answer_anno["annotations"]:
                if str(sample["image_id"]) + "<->" + str(sample["question_id"]) not in answer_dict:
                    answer_dict[str(sample["image_id"]) + "<->" + str(sample["question_id"])] = [
                        x["answer"] for x in sample["answers"]
                    ]
        question_dict = {}
        for sample in question_anno["questions"]:
            if str(sample["image_id"]) + "<->" + str(sample["question_id"]) not in question_dict:
                question_dict[str(sample["image_id"]) + "<->" + str(sample["question_id"])] = sample["question"]
        return caption_dict, answer_dict, question_dict

    def process_answer(self, answer):
        answer = answer.replace(".", "").replace(",", "").lower()
        to_be_removed = {"a", "an", "the", "to", ""}
        answer_list = answer.split(" ")
        answer_list = [item for item in answer_list if item not in to_be_removed]
        return " ".join(answer_list)


class Pipeline:
    """
    Main inference class
    """

    def __init__(self):
        # Tagging model setup
        segment_model = AutoModelForImageSegmentation.from_pretrained(tag_model, use_pretrained_backbone=False)
        resnet_model = timm.create_model(
            "resnet50", pretrained=False, features_only=True, out_indices=(1, 2, 3, 4), in_chans=3
        )
        resnet_weights = torch.load(resnet_path)
        resnet_weights_no_head = dict(itertools.islice(resnet_weights.items(), len(resnet_weights) - 2))
        resnet_model.load_state_dict(resnet_weights_no_head)
        segment_model.detr.model.backbone.conv_encoder.model = resnet_model
        self.segment = pipeline(
            "image-segmentation", model=segment_model, feature_extractor=AutoFeatureExtractor.from_pretrained(tag_model)
        )
        self.tags = []

        # Caption model setup
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model)
        self.caption_feature_extractor = ViTFeatureExtractor.from_pretrained(caption_model)
        self.caption_tokenizer = AutoTokenizer.from_pretrained(caption_model)
        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CLIP Setup
        self.clip_session = InferenceSession(str(clip_onnx))
        self.clip_processor = CLIPProcessor.from_pretrained(clip_processor)

        # PICa Setup
        openai.api_key_path = Path(__file__).resolve().parent / "key.txt"

    def predict_caption(self, image):
        pixel_values = self.caption_feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        output_ids = self.caption_model.generate(pixel_values, **gen_kwargs)

        preds = self.caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds[0]

    def predict(self, image: Union[str, Path, Image.Image], question: Union[str, Path]) -> str:
        if not isinstance(image, Image.Image):
            image_pil = Image.open(image)
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert(mode="RGB")
        else:
            image_pil = image
        if isinstance(question, Path) | os.path.exists(question):
            with open(question, "r") as f:
                question_str = f.readline()
        else:
            question_str = question

        # Generating image tag(s)
        for dic in self.segment(image_pil):
            self.tags.append(dic["label"])
        tag_info = [img_id, self.tags]

        # Generating image caption
        caption = self.predict_caption(image_pil)
        caption_info = [img_id, caption]

        # Generating image/question features
        inputs = self.clip_processor(text=[question_str], images=image_pil, return_tensors="np", padding=True)
        # for i in session.get_outputs(): print(i.name)
        outputs = self.clip_session.run(
            output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(inputs)
        )

        # Generating context idxs
        context_idxs = {"0": str(img_id) + "<->" + str(question_id)}

        # Answering question
        questions = {"questions": [{"image_id": img_id, "question": question_str, "question_id": question_id}]}
        okvqa = PICa_OKVQA(
            caption_info, tag_info, questions, context_idxs, outputs[2], outputs[3]
        )  # Have to initialize here because necessary objects need to be generated
        answer = okvqa.answer_gen()
        # rationale = okvqa.rationale(answer)

        return answer  # + " because " + rationale


# Running model
def main():
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)

    args = parser.parse_args()

    # Answering question
    pipeline = Pipeline()
    pred_str = pipeline.predict(args.image, args.question)

    print(pred_str)


if __name__ == "__main__":
    main()
