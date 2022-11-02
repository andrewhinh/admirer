# Imports
import argparse
import itertools
import json
import os
from pathlib import Path
import random
from typing import Union, Optional, List, Any, Dict, Tuple
from collections import defaultdict

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
from dotenv import load_dotenv


# Variables
# Artifact path
artifact_path = Path(__file__).resolve().parent / "artifacts"

# PICa formatting/config
img_id = 100  # Random
question_id = 1005  # Random
n_shot = 32
n_ensemble = 5
similarity_metrics = ["question", "imagequestion"]
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

# Load environment variables
load_dotenv()


# Helper/main classes
class PICa_OKVQA:
    """
    Question Answering Class
    """

    def __init__(
        self, 
        caption_info: Tuple[int, str], 
        tag_info: Tuple[int, List[str]], 
        questions: Dict[str, List[Dict[str, Any]]], 
        context_idxs: Dict[str, str], 
        question_features, 
        image_features
    ):
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
        (
            self.traincontext_caption_dict,
            self.traincontext_answer_dict,
            self.traincontext_question_dict,
        ) = self.add_anno(
            "%s/admirer-pica.json" % coco_path,
            self.traincontext_caption_dict,
            self.traincontext_answer_dict,
            self.traincontext_question_dict,
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
        if not pred_answer or pred_answer == " ":
            pred_answer = "?"
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


    def get_context_keys(self, key: str, metric: str, n: int) -> List[str]:
        """Get context keys based on similarity scores"""
        # Throw error with an invalid metric
        if metric not in similarity_metrics:
            raise ValueError("Invalid similarity metric")
        
        lineid = self.valkey2idx[key]
        similarity: np.ndarray = np.matmul(self.train_feature, self.val_feature[lineid, :])
        
        if metric == "imagequestion":
            similarity = similarity + np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
        
        index: np.ndarray = similarity.argsort()[-n:][::-1] # Get n indices with highest similarity scores
        return [self.train_idx[str(x)] for x in index]


    def load_similarity(self, context_idxs: Dict[str, str], question_features, image_features) -> None:
        self.valkey2idx: Dict[str, int] = {}
        for idx in context_idxs:
            self.valkey2idx[context_idxs[idx]] = int(idx)
        
        # Raise exception if the metric is not valid
        if similarity_metric not in similarity_metrics:
            raise ValueError("Invalid similarity metric")
        
        # Add train feature, val feature and train idx for all valid metrics
        self.train_feature = np.load("%s/coco_clip_vitb16_train2014_okvqa_question.npy" % similarity_path)
        self.val_feature = question_features
        self.train_idx: Dict[str, str] = json.load(
            open(
                "%s/okvqa_qa_line2sample_idx_train2014.json" % similarity_path,
                "r",
            )
        )
        
        # Add image features for image questions
        if similarity_metric == "imagequestion":
            self.image_train_feature = np.load(
                "%s/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy" % similarity_path
            )
            self.image_val_feature = image_features


    def load_tags(self) -> Dict[int, str]:
        """Loads tags for an image"""
        tags_dict = {}
        image_id, tags = self.tag_info
        
        # Concatenate tags into a single string
        tag_str = ", ".join([x for x in tags])
        
        tags_dict[image_id] = tag_str
        return tags_dict


    def load_cachetext(self, caption_info: Tuple[int, str]) -> Dict[int, str]:
        """Loads and adds cachetect to the caption"""
        if "tag" in caption_type: 
            tags_dict = self.load_tags()
        idx, caption = caption_info
        caption_dict = {idx: caption}
        if caption_type == "vinvl_tag":
            caption_dict[idx] += ". " + tags_dict[idx]
        return caption_dict


    def load_anno(
        self, 
        coco_caption_file: Optional[Path], 
        answer_anno_file: Optional[Path], 
        question_anno_file: Optional[Path], 
        questions
    ) -> Tuple[Dict[int, List[str]], Dict[str, List[str]], Dict[str, str]]:
        """Loads annotation from a caption file"""
        
        # Define default dictionaries
        caption_dict: defaultdict[int, List[str]] = defaultdict(list)
        answer_dict: defaultdict[str, List[str]] = defaultdict(list)
        question_dict: defaultdict[str, str] = defaultdict(list)
        
        # Create caption dictionary
        if coco_caption_file is not None:
            coco_caption = json.load(open(coco_caption_file, "r"))
            if isinstance(coco_caption, dict):
                coco_caption: List[Dict[str, Union[str, int]]] = coco_caption["annotations"]
            for sample in coco_caption:
                caption_dict[sample["image_id"]].append(sample["caption"]) # int -> sample[image_id] 
                
        # Create answer dictionary
        if answer_anno_file is not None:
            answer_data = json.load(open(answer_anno_file, "r"))
            answer_annotations: List[Dict[str, Any]] = answer_data["annotations"]
            for sample in answer_annotations:
                id = str(sample["image_id"]) + "<->" + str(sample["question_id"])
                if id not in answer_dict:
                    answer_dict[id] = [
                        x["answer"] for x in sample["answers"]
                    ]
                    
        # Create question dictionary
        if question_anno_file is not None:
            question_data = json.load(open(question_anno_file, "r"))
        else:
            question_data = questions

        question_annotations: List[Dict[str, Union[str, int]]] = question_data["questions"]
        for sample in question_annotations:
            id = str(sample["image_id"]) + "<->" + str(sample["question_id"])
            if id not in question_dict:
                question_dict[id] = sample["question"]
                
        return dict(caption_dict), dict(answer_dict), dict(question_dict)


    def add_anno(
        self, 
        add: Optional[Path], 
        traincontext_caption_dict: Dict[int, List[str]], 
        traincontext_answer_dict: Dict[str, List[str]], 
        traincontext_question_dict: Dict[str, str]
    ):
        """Add extra annotations to the annotations dictionaries"""
        if add is not None:
            add_dict = json.load(open(add, "r"))

        caption_add = dict(zip(list(add_dict["image_id"].values()), list(add_dict["caption"].values())))
        combine_ids = [
            str(image_id) + "<->" + str(question_id)
            for image_id, question_id in zip(
                list(add_dict["image_id"].values()), list(add_dict["question_id"].values())
            )
        ]
        answer_add = dict(zip(combine_ids, list(add_dict["answer"].values())))
        question_add = dict(zip(combine_ids, list(add_dict["question"].values())))

        traincontext_caption_dict.update(caption_add)
        traincontext_answer_dict.update(answer_add)
        traincontext_question_dict.update(question_add)

        return traincontext_caption_dict, traincontext_answer_dict, traincontext_question_dict


    def process_answer(self, answer: str) -> str:
        """Processes answer by removing unwanted characters and words"""
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
        openai.api_key = os.getenv("OPENAI_API_KEY")

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
        tag_info: Tuple[int, List[str]] = [img_id, self.tags]

        # Generating image caption
        caption = self.predict_caption(image_pil)
        caption_info: Tuple[int, str] = [img_id, caption]

        # Generating image/question features
        inputs = self.clip_processor(text=[question_str], images=image_pil, return_tensors="np", padding=True)
        # for i in session.get_outputs(): print(i.name)
        outputs = self.clip_session.run(
            output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(inputs)
        )

        # Generating context idxs
        context_idxs: Dict[str, str] = {"0": str(img_id) + "<->" + str(question_id)}

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
