# Imports
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import random
from typing import Union

import numpy as np
import openai
from PIL import Image
import torch
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    pipeline,
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
)


# Variables
# File paths
ex_path = Path(__file__).resolve().parent / "tests" / "support"
caption_path = ex_path / "caption.tsv"
tag_path = ex_path / "tag.tsv"
question_path = ex_path / "question.json"
idx_path = ex_path / "idx.json"
question_feature_path = ex_path / "question_feature.npy"
image_feature_path = ex_path / "image_feature.npy"

# PICa formatting
img_id = 100  # Random
question_id = 1005  # Random

# Segmentation model config
tag_model = "facebook/detr-resnet-50-panoptic"
tag_revision = "fc15262"
max_length = 16
num_beams = 4

# Caption model config
caption_model = "nlpconnect/vit-gpt2-image-captioning"
engine = "davinci"
caption_type = "vinvl_tag"
n_shot = 16
n_ensemble = 5
similarity_metric = "imagequestion"
coco_path = Path(__file__).resolve().parent / "coco_annotations"
similarity_path = Path(__file__).resolve().parent / "coco_clip_new"


# Helper/main classes
class PICa_OKVQA:
    """
    Question Answering Class
    """

    def __init__(self):
        # load cached image representation (Coco caption & Tags)
        self.inputtext_dict = self.load_cachetext()

        (
            self.traincontext_caption_dict,
            self.traincontext_answer_dict,
            self.traincontext_question_dict,
        ) = self.load_anno(
            "%s/captions_train2014.json" % coco_path,
            "%s/mscoco_train2014_annotations.json" % coco_path,
            "%s/OpenEnded_mscoco_train2014_questions.json" % coco_path,
        )
        self.train_keys = list(self.traincontext_answer_dict.keys())
        self.load_similarity()

    def answer_gen(self):
        _, _, question_dict = self.load_anno(None, None, question_path)

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
        _, _, question_dict = self.load_anno(None, None, question_path)

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
            question_similarity = np.matmul(self.train_feature, self.val_feature.detach().numpy()[lineid, :])
            # end of Q-similairty
            similarity = question_similarity + np.matmul(
                self.image_train_feature, self.image_val_feature.detach().numpy()[lineid, :]
            )
            index = similarity.argsort()[-n:][::-1]
            return [self.train_idx[str(x)] for x in index]
        else:
            return None

    def load_similarity(self):
        val_idx = json.load(open(idx_path, "r"))
        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)
        if similarity_metric == "question":
            self.train_feature = np.load("%s/coco_clip_vitb16_train2014_okvqa_question.npy" % similarity_path)
            self.val_feature = torch.load(question_feature_path)
            self.train_idx = json.load(
                open(
                    "%s/okvqa_qa_line2sample_idx_train2014.json" % similarity_path,
                    "r",
                )
            )
        elif similarity_metric == "imagequestion":
            self.train_feature = np.load("%s/coco_clip_vitb16_train2014_okvqa_question.npy" % similarity_path)
            self.val_feature = torch.load(question_feature_path)
            self.train_idx = json.load(
                open(
                    "%s/okvqa_qa_line2sample_idx_train2014.json" % similarity_path,
                    "r",
                )
            )
            self.image_train_feature = np.load(
                "%s/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy" % similarity_path
            )
            self.image_val_feature = torch.load(image_feature_path)

    def load_tags(self):
        tags_dict = {}
        read_tsv = list(csv.reader(open(tag_path, "r"), delimiter="\t"))
        row = read_tsv[0]
        image_id, tags = int(row[0]), ast.literal_eval(row[1])
        tag_str = ", ".join([x["class"] if x["class"] != " " else "" for x in tags])
        tags_dict[image_id] = tag_str
        return tags_dict

    def load_cachetext(self):
        read_tsv = list(csv.reader(open(caption_path, "r"), delimiter="\t"))
        if "tag" in caption_type:
            tags_dict = self.load_tags()
        row = read_tsv[0]
        idx = int(row[0])
        caption = ast.literal_eval(row[1])[0]["caption"]
        if caption == " ":
            caption = ""
        caption_dict = {idx: caption}
        if caption_type == "vinvl_tag":
            caption_dict[idx] += ". " + tags_dict[idx]
        return caption_dict

    def load_anno(self, coco_caption_file, answer_anno_file, question_anno_file):
        if coco_caption_file is not None:
            coco_caption = json.load(open(coco_caption_file, "r"))
            if isinstance(coco_caption, dict):
                coco_caption = coco_caption["annotations"]
        if answer_anno_file is not None:
            answer_anno = json.load(open(answer_anno_file, "r"))
        question_anno = json.load(open(question_anno_file, "r"))

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
        self.segment = pipeline("image-segmentation", model=tag_model, revision=tag_revision)
        self.tags = []

        # Caption model setup
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(caption_model)
        self.caption_feature_extractor = ViTFeatureExtractor.from_pretrained(caption_model)
        self.caption_tokenizer = AutoTokenizer.from_pretrained(caption_model)
        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.caption_model.to(self.device)
        self.captions = []

        # CLIP Setup
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # PICa Setup
        openai.api_key_path = Path(__file__).resolve().parent / "key.txt"

    def predict_caption(self, image):
        images = []
        images.append(image)

        pixel_values = self.caption_feature_extractor(images=images, return_tensors="pt").pixel_values
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
            if not dic["label"]:
                self.tags.append({"class": " "})
            else:
                self.tags.append({"class": dic["label"]})
        with open(tag_path, "wt") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow([100, self.tags])

        # Generating image caption
        caption = self.predict_caption(image_pil)
        if not caption:
            self.captions.append({"caption": " "})
        else:
            self.captions.append({"caption": caption})
        with open(caption_path, "wt") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow([100, self.captions])

        # Generating image/question features
        inputs = self.clip_processor(text=[question_str], images=image_pil, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        torch.save(outputs.text_embeds, question_feature_path)
        torch.save(outputs.image_embeds, image_feature_path)

        # Generating context idxs
        context_idxs = {"0": str(img_id) + "<->" + str(question_id)}
        with open(idx_path, "w") as out_file:
            json.dump(context_idxs, out_file)

        # Answering question
        questions = {"questions": [{"image_id": img_id, "question": question_str, "question_id": question_id}]}
        with open(question_path, "w") as out_file:
            json.dump(questions, out_file)
        okvqa = PICa_OKVQA()  # Have to initialize here because necessary files need to be generated
        answer = okvqa.answer_gen()
        rationale = okvqa.rationale(answer)

        # Cleaning up generated files
        for path in [caption_path, tag_path, question_path, idx_path, question_feature_path, image_feature_path]:
            if os.path.exists(str(path)):
                os.remove(path)

        return answer + " because " + rationale


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
