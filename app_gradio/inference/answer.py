# Imports
import argparse
import ast
import csv
import json
import random

import numpy as np
import openai
from PIL import Image
import torch
from transformers import AutoTokenizer, pipeline, VisionEncoderDecoderModel, ViTFeatureExtractor


# Variables
parser = argparse.ArgumentParser()
parser.add_argument("--api_key", type=str, required=True, help="api key; https://openai.com/api/")
parser.add_argument(
    "--image",
    type=str,
    default="example/img.jpg",
)
parser.add_argument(
    "--question",
    type=str,
    default="What is the temperature?",
)
parser.add_argument(
    "--tag_model",
    type=str,
    default="facebook/detr-resnet-50-panoptic",
)
parser.add_argument(
    "--tag_revision",
    type=str,
    default="fc15262",
)
parser.add_argument(
    "--caption_model",
    type=str,
    default="nlpconnect/vit-gpt2-image-captioning",
)
parser.add_argument("--engine", type=str, default="davinci", help="api engine; https://openai.com/api/")
parser.add_argument("--caption_type", type=str, default="vinvl_tag", help="vinvl_tag, vinvl")
parser.add_argument("--n_shot", type=int, default=1, help="number of shots (up to 16)")
parser.add_argument("--n_ensemble", type=int, default=1, help="number of ensemble")
parser.add_argument(
    "--similarity_metric",
    default=None,
    help="random/question/imagequestion",
)
parser.add_argument("--coco_path", type=str, default="coco_annotations")
parser.add_argument("--similarity_path", type=str, default="coco_clip_new")
args = parser.parse_args()

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

question_path = "example/question.json"
caption_path = "example/caption.tsv"
tag_path = "example/tag.tsv"


# Functions/classes for model
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def process_answer(answer):
    answer = answer.replace(".", "").replace(",", "").lower()
    to_be_removed = {"a", "an", "the", "to", ""}
    answer_list = answer.split(" ")
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return " ".join(answer_list)


def load_anno(coco_caption_file, answer_anno_file, question_anno_file):
    if isinstance(coco_caption_file, None):
        coco_caption = json.load(open(coco_caption_file, "r"))
        if isinstance(coco_caption, {}):
            coco_caption = coco_caption["annotations"]
    if isinstance(answer_anno_file, not None):
        answer_anno = json.load(open(answer_anno_file, "r"))
    question_anno = json.load(open(question_anno_file, "r"))

    caption_dict = {}
    if isinstance(coco_caption_file, not None):
        for sample in coco_caption:
            if sample["image_id"] not in caption_dict:
                caption_dict[sample["image_id"]] = [sample["caption"]]
            else:
                caption_dict[sample["image_id"]].append(sample["caption"])
    answer_dict = {}
    if isinstance(answer_anno_file, not None):
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


class PICa_OKVQA:
    """
    Main inference class
    """

    def __init__(self, args):
        self.args = args
        # load cached image representation (Coco caption & Tags)
        self.inputtext_dict = self.load_cachetext()

        (self.traincontext_caption_dict, self.traincontext_answer_dict, self.traincontext_question_dict,) = load_anno(
            "%s/captions_train2014.json" % args.coco_path,
            "%s/mscoco_train2014_annotations.json" % args.coco_path,
            "%s/OpenEnded_mscoco_train2014_questions.json" % args.coco_path,
        )
        self.train_keys = list(self.traincontext_answer_dict.keys())
        self.load_similarity()

    def inference(self):
        _, _, question_dict = load_anno(None, None, question_path)

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
            self.args.similarity_metric,
            self.args.n_shot * self.args.n_ensemble,
        )

        for repeat in range(self.args.n_ensemble):
            # prompt format following GPT-3 QA API
            prompt = "Please answer the question according to the above context.\n===\n"
            for ni in range(self.args.n_shot):
                if isinstance(context_key_list, None):
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                else:
                    context_key = context_key_list[ni + self.args.n_shot * repeat]
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
                    engine=self.args.engine,
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
            pred_answer_list.append(process_answer(response["choices"][0]["text"]))
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

    def load_similarity(self):
        val_idx = json.load(
            open(
                "%s/okvqa_qa_line2sample_idx_val2014.json" % self.args.similarity_path,
                "r",
            )
        )
        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)
        if self.args.similarity_metric == "question":
            self.train_feature = np.load("%s/coco_clip_vitb16_train2014_okvqa_question.npy" % self.args.similarity_path)
            self.val_feature = np.load("%s/coco_clip_vitb16_val2014_okvqa_question.npy" % self.args.similarity_path)
            self.train_idx = json.load(
                open(
                    "%s/okvqa_qa_line2sample_idx_train2014.json" % self.args.similarity_path,
                    "r",
                )
            )
        elif self.args.similarity_metric == "imagequestion":
            self.train_feature = np.load("%s/coco_clip_vitb16_train2014_okvqa_question.npy" % self.args.similarity_path)
            self.val_feature = np.load("%s/coco_clip_vitb16_val2014_okvqa_question.npy" % self.args.similarity_path)
            self.train_idx = json.load(
                open(
                    "%s/okvqa_qa_line2sample_idx_train2014.json" % self.args.similarity_path,
                    "r",
                )
            )
            self.image_train_feature = np.load(
                "%s/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy" % self.args.similarity_path
            )
            self.image_val_feature = np.load(
                "%s/coco_clip_vitb16_val2014_okvqa_convertedidx_image.npy" % self.args.similarity_path
            )

    def load_tags(self):
        tags_dict = {}
        read_tsv = csv.reader(open(tag_path, "r"), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]), ast.literal_eval(row[1])
            tag_str = ", ".join([x["class"] for x in tags])
            tags_dict[image_id] = tag_str
        return tags_dict

    def load_cachetext(self):
        read_tsv = csv.reader(open(caption_path, "r"), delimiter="\t")
        caption_dict = {}
        if "tag" in self.args.caption_type:
            tags_dict = self.load_tags()
        if self.args.caption_type == "vinvl_tag":
            for row in read_tsv:
                if int(row[0]) not in caption_dict:
                    caption_dict[int(row[0])] = [ast.literal_eval(row[1])[0]["caption"] + ". " + tags_dict[int(row[0])]]
                else:
                    caption_dict[int(row[0])].append(
                        ast.literal_eval(row[1])[0]["caption"] + ". " + tags_dict[int(row[0])]
                    )
        else:
            for row in read_tsv:
                if int(row[0]) not in caption_dict:
                    caption_dict[int(row[0])] = [ast.literal_eval(row[1])[0]["caption"]]
                else:
                    caption_dict[int(row[0])].append(ast.literal_eval(row[1])[0]["caption"])
        return caption_dict


# Running model
with open(question_path, "r") as file:
    data = json.load(file)
    data["questions"][0]["question"] = args.question
with open(question_path, "w") as file:
    json.dump(data, file)


model = pipeline("image-segmentation", model=args.tag_model, revision=args.tag_revision)
tags = []
for dic in model(args.image):
    tags.append({"class": dic["label"], "conf": dic["score"]})
with open(tag_path, "wt") as out_file:
    tsv_writer = csv.writer(out_file, delimiter="\t")
    tsv_writer.writerow([0, tags])

model = VisionEncoderDecoderModel.from_pretrained(args.caption_model)
feature_extractor = ViTFeatureExtractor.from_pretrained(args.caption_model)
tokenizer = AutoTokenizer.from_pretrained(args.caption_model)
device = torch.device("cpu")
model.to(device)
temp = []
val = predict_step([args.image])
temp.append({"caption": val[0], "conf": 0.0})
with open(caption_path, "wt") as out_file:
    tsv_writer = csv.writer(out_file, delimiter="\t")
    tsv_writer.writerow([0, temp])

openai.api_key = args.api_key
okvqa = PICa_OKVQA(args)
answer = okvqa.inference()
print(answer)
