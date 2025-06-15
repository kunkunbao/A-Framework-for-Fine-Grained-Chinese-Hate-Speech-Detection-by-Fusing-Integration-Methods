import json
import torch
import numpy as np
from torch import nn
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    BertForTokenClassification,
    BertModel,
    BertPreTrainedModel,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from torchcrf import CRF
from functools import lru_cache
import os

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签配置
label_config = {
    "Region": {"id": 0, "description": "地域歧视"},
    "Racism": {"id": 1, "description": "种族歧视"}, 
    "Sexism": {"id": 2, "description": "性别歧视"},
    "LGBTQ": {"id": 3, "description": "性取向歧视"},
    "Religion": {"id": 4, "description": "宗教歧视"},
    "Disability": {"id": 5, "description": "残障歧视"},
    "others": {"id": 6, "description": "其他类型仇恨言论"},
    "non-hate": {"id": 7, "description": "非仇恨言论"}
}

# 实体识别标签
ner_labels = ["O", "B-OBJ", "I-OBJ"]
ner_label2id = {label: i for i, label in enumerate(ner_labels)}
ner_id2label = {i: label for i, label in enumerate(ner_labels)}

class HateSpeechDataProcessor:
    def __init__(self, label_config):
        self.label_config = label_config
        self.reverse_label_map = {v["id"]: k for k, v in label_config.items()}
    
    def load_and_preprocess(self, json_path):
        """加载和预处理数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            # 解析文本内容
            text = self._extract_text(item)
            texts.append(text)
            
            # 解析标签
            label = self._extract_label(item)
            labels.append(label)
        
        return texts, labels
    
    def _extract_text(self, item):
        """从数据项中提取文本"""
        if "output" in item:
            parts = item["output"].split(" | ")
            if len(parts) >= 2:
                return parts[0] + " " + parts[1]
        return item["content"]
    
    def _extract_label(self, item):
        """从数据项中提取标签"""
        if "output" in item:
            parts = item["output"].split(" | ")
            if len(parts) >= 3:
                label_str = parts[2].split(", ")[0].strip()
                if label_str in self.label_config:
                    return self.label_config[label_str]["id"]
        
        # 默认返回others类别
        return self.label_config["others"]["id"]
    
    def augment_data(self, texts, labels, augmentation_factor=2):
        """数据增强"""
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # 简单的同义词替换增强
            for _ in range(augmentation_factor):
                augmented_text = self._synonym_replacement(text)
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    def _synonym_replacement(self, text):
        """简单的同义词替换"""
        # 这里可以使用更复杂的同义词库
        synonyms = {
            "黑鬼": ["黑人", "非裔"],
            "女人": ["女性", "女的"],
            "同性恋": ["同志", "LGBT"]
        }
        
        for word, replacements in synonyms.items():
            if word in text:
                text = text.replace(word, np.random.choice(replacements))
        return text

class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, entity_type="target"):
        """
        entity_type: "target"或"argument"，指定要提取的是目标对象还是论点
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.entity_type = entity_type
        self.examples = []
        
        self._preprocess_data()
    
    def _preprocess_data(self):
        for item in tqdm(self.data, desc="Processing NER dataset"):
            text = item["content"]
            
            # 从output字段解析出实体
            if "output" in item:
                parts = item["output"].split(" | ")
                if len(parts) >= 4:
                    target = parts[0].strip()
                    argument = parts[1].strip()
                    
                    # 根据entity_type选择要提取的实体
                    entity = target if self.entity_type == "target" else argument
                    
                    try:
                        # 生成字符级标签
                        char_labels = self._create_char_labels(text, entity)
                        
                        # 分词并对齐标签
                        tokens = []
                        labels = []
                        for char, label in zip(text, char_labels):
                            char_tokens = self.tokenizer.tokenize(char)
                            tokens.extend(char_tokens)
                            
                            if len(char_tokens) > 0:
                                labels.extend([label] + ["O"] * (len(char_tokens)-1))
                        
                        # 转换为ID
                        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                        label_ids = [ner_label2id.get(l, 0) for l in labels]
                        
                        # 截断处理
                        input_ids = input_ids[:self.max_length - 2]
                        label_ids = label_ids[:self.max_length - 2]
                        
                        # 添加特殊token
                        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
                        label_ids = [0] + label_ids + [0]
                        
                        attention_mask = [1] * len(input_ids)
                        
                        # 填充
                        padding_length = self.max_length - len(input_ids)
                        if padding_length > 0:
                            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                            label_ids = label_ids + [0] * padding_length
                            attention_mask = attention_mask + [0] * padding_length
                        
                        self.examples.append({
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": label_ids,
                            "original_text": text,
                            "entity": entity
                        })
                    except ValueError as e:
                        print(f"警告: 项目ID {item['id']} 的实体 '{entity}' 未在文本中找到: {text}")
                        continue
                    except Exception as e:
                        print(f"处理样本时出错 (ID: {item['id']}): {str(e)}")
                        continue
    
    # ... 保持其他方法不变 ...
    
    def _create_char_labels(self, text, entity):
        """生成BIO标注序列"""
        labels = []
        entity_start = text.find(entity)
        
        if entity_start == -1:
            raise ValueError(f"Entity '{entity}' not found in text: {text}")
        
        entity_end = entity_start + len(entity)
        
        for i, char in enumerate(text):
            if i < entity_start or i >= entity_end:
                labels.append("O")
            elif i == entity_start:
                labels.append("B-OBJ")
            else:
                labels.append("I-OBJ")
        
        return labels
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.examples[idx]["input_ids"]),
            "attention_mask": torch.tensor(self.examples[idx]["attention_mask"]),
            "labels": torch.tensor(self.examples[idx]["labels"])
        }

class BertCRFForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.byte())
            return loss
        else:
            return self.crf.decode(logits, mask=attention_mask.byte())

class EnsembleClassifier:
    def __init__(self, model_paths):
        self.models = []
        self.tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
        
        for path in model_paths:
            model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
            model.eval()
            self.models.append(model)
    
    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        all_logits = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(**inputs)
                all_logits.append(outputs.logits)
        
        avg_logits = torch.mean(torch.stack(all_logits), dim=0)
        probabilities = torch.softmax(avg_logits, dim=1)
        predicted_id = torch.argmax(probabilities, dim=1).item()
        
        return predicted_id, probabilities[0].cpu().numpy()

class HateSpeechTrainer:
    def __init__(self, label_config):
        self.label_config = label_config
        self.data_processor = HateSpeechDataProcessor(label_config)
    
    def train_classifier(self, train_json_path, output_dir, model_name="hfl/chinese-roberta-wwm-ext"):
        """训练分类模型"""
        # 加载数据
        texts, labels = self.data_processor.load_and_preprocess(train_json_path)
        
        # 数据增强
        texts, labels = self.data_processor.augment_data(texts, labels)
        
        # 划分训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # 初始化分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_config)
        ).to(device)
        
        # 创建数据集
        train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer)
        val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=True,
            report_to="none"
        )
        
        # 自定义评估指标
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            
            precision = precision_score(labels, preds, average='weighted')
            recall = recall_score(labels, preds, average='weighted')
            f1 = f1_score(labels, preds, average='weighted')
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': (preds == labels).mean()
            }
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # 开始训练
        print("Training classification model...")
        trainer.train()
        
        # 保存模型
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 保存标签配置
        with open(os.path.join(output_dir, "label_config.json"), 'w') as f:
            json.dump(self.label_config, f)
        
        print(f"Model saved to {output_dir}")
        return model, tokenizer
    
    def train_ner_model(self, train_json_path, output_dir, model_name="hfl/chinese-roberta-wwm-ext", model_type="crf", entity_type="target"):
        """训练实体识别模型"""
        # 加载数据
        with open(train_json_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # 初始化分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 初始化模型
        if model_type == "crf":
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = len(ner_labels)
            model = BertCRFForTokenClassification.from_pretrained(
                model_name,
                config=config
            ).to(device)
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(ner_labels),
                id2label=ner_id2label,
                label2id=ner_label2id
            ).to(device)
        
        # 创建数据集 - 这里传入entity_type参数
        dataset = NERDataset(train_data, tokenizer, entity_type=entity_type)
    
    # ... 其余代码保持不变 ...
        
        # 划分训练集和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=15,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            fp16=True,
            report_to="none"
        )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # 开始训练
        print(f"Training NER model ({model_type})...")
        trainer.train()
        
        # 保存模型
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # 保存标签配置
        with open(os.path.join(output_dir, "ner_labels.json"), 'w') as f:
            json.dump({"labels": ner_labels}, f)
        
        print(f"NER model saved to {output_dir}")
        return model, tokenizer

class HateSpeechDetector:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.sensitive_words = self._load_sensitive_words()
        self._load_models()
    
    def _load_sensitive_words(self):
        """加载敏感词库"""
        sensitive_words = set()
        # 这里可以加载外部敏感词库
        sensitive_words.update(["黑鬼", "杂种", "婊子", "变态", "死全家"])
        return sensitive_words
    
    def _load_models(self):
        """加载所有模型"""
        # 分类模型
        self.classifier = EnsembleClassifier([
            self.config["classifier_model_path1"],
            self.config["classifier_model_path2"]
        ])
        
        # 实体识别模型
        self.object_extractor = self._load_ner_model(
            self.config["object_model_path"],
            self.config["object_model_type"]
        )
        self.argument_extractor = self._load_ner_model(
            self.config["argument_model_path"],
            self.config["argument_model_type"]
        )
    
    def _load_ner_model(self, model_path, model_type):
        """加载NER模型"""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if model_type == "crf":
            model = BertCRFForTokenClassification.from_pretrained(model_path).to(self.device)
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        
        model.eval()
        return {"model": model, "tokenizer": tokenizer}
    
    def _preprocess_text(self, text):
        """文本预处理"""
        # 简单的清洗
        text = text.strip()
        text = "".join(c for c in text if c.isprintable())
        return text
    
    def _contains_sensitive_words(self, text):
        """检查是否包含敏感词"""
        return any(word in text for word in self.sensitive_words)
    
    def _extract_entities(self, text, ner_model):
        """使用NER模型提取实体"""
        tokenizer = ner_model["tokenizer"]
        model = ner_model["model"]
        
        # 分词
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:126]  # 保留空间给特殊token
        
        # 转换为ID并添加特殊token
        input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        
        # 填充
        padding_length = 128 - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # 转换为tensor并移动到设备
        input_ids = torch.tensor([input_ids]).to(self.device)
        attention_mask = torch.tensor([attention_mask]).to(self.device)
        
        # 预测
        with torch.no_grad():
            if isinstance(model, BertCRFForTokenClassification):
                predictions = model(input_ids, attention_mask)
                predictions = predictions[0]  # CRF返回的是列表
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=2).squeeze().cpu().numpy()
        
        # 提取实体
        entity_tokens = []
        entities = []
        
        for token_id, pred_id in zip(input_ids[0], predictions):
            if token_id == tokenizer.cls_token_id or token_id == tokenizer.sep_token_id:
                continue
            
            token = tokenizer.convert_ids_to_tokens(token_id.item())
            
            if pred_id == ner_label2id["B-OBJ"]:
                if entity_tokens:
                    entities.append(self._convert_tokens_to_text(tokenizer, entity_tokens))
                    entity_tokens = []
                entity_tokens.append(token)
            elif pred_id == ner_label2id["I-OBJ"] and entity_tokens:
                entity_tokens.append(token)
            elif entity_tokens:
                entities.append(self._convert_tokens_to_text(tokenizer, entity_tokens))
                entity_tokens = []
        
        if entity_tokens:
            entities.append(self._convert_tokens_to_text(tokenizer, entity_tokens))
        
        return entities
    
    def _convert_tokens_to_text(self, tokenizer, tokens):
        """将token转换为文本"""
        text = tokenizer.convert_tokens_to_string(tokens)
        # 清理特殊符号
        text = text.replace(" ##", "").replace("##", "")
        return text.strip()
    
    def _post_process(self, text, category, entities):
        """后处理预测结果"""
        # 特殊规则处理
        if "地域" in text and category != "Region":
            category = "Region"
        
        # 实体修正
        processed_entities = []
        for entity in entities:
            # 过滤掉无意义的实体
            if len(entity) > 1 and not entity.isdigit():
                processed_entities.append(entity)
        
        return category, processed_entities
    
    @lru_cache(maxsize=10000)
    def detect(self, text):
        """检测仇恨言论"""
        # 预处理文本
        cleaned_text = self._preprocess_text(text)
        
        # 快速敏感词检查
        if not self._contains_sensitive_words(cleaned_text):
            return {
                "text": text,
                "category": "non-hate",
                "hateful": False,
                "target": None,
                "argument": None,
                "confidence": 1.0
            }
        
        # 分类
        predicted_id, probabilities = self.classifier.predict(cleaned_text)
        category = self.config["id2label"][predicted_id]
        confidence = probabilities[predicted_id]
        
        # 提取对象和论点
        objects = self._extract_entities(cleaned_text, self.object_extractor)
        arguments = self._extract_entities(cleaned_text, self.argument_extractor)
        
        # 后处理
        category, objects = self._post_process(cleaned_text, category, objects)
        
        return {
            "text": text,
            "category": category,
            "hateful": category != "non-hate",
            "target": objects[0] if objects else None,
            "argument": arguments[0] if arguments else None,
            "confidence": float(confidence)
        }
    
    def batch_detect(self, texts, batch_size=32):
        """批量检测"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            results.extend([self.detect(text) for text in batch])
        return results

def main():
    # 配置
    config = {
        "label_config": label_config,
        "classifier_model_path1": "./models/classifier_v1",
        "classifier_model_path2": "./models/classifier_v2",
        "object_model_path": "./models/object_model_crf",
        "object_model_type": "crf",
        "argument_model_path": "./models/argument_model_crf",
        "argument_model_type": "crf",
        "id2label": {v["id"]: k for k, v in label_config.items()}
    }
    
    # 训练模式
    if True:  # 设为True以训练模型
        trainer = HateSpeechTrainer(label_config)
        
        # 训练分类模型
        trainer.train_classifier(
           "train.json",
            "./models/classifier_v1",
            model_name="hfl/chinese-roberta-wwm-ext"
        )
        
        # 训练第二个分类模型用于集成
        trainer.train_classifier(
            "train.json",
            "./models/classifier_v2",
            model_name="bert-base-chinese"
        )
        
        # 训练目标对象识别模型
        # 训练目标对象识别模型
        trainer.train_ner_model(
            "train.json",
           "./models/object_model_crf",
            model_type="crf",
            entity_type="target"  # 指定提取目标对象
        )
        
        # 训练论点识别模型
        trainer.train_ner_model(
            "train.json",
            "./models/argument_model_crf",
            model_type="crf",
            entity_type="argument"  # 指定提取论点
        )
    
    # 初始化检测器
    detector = HateSpeechDetector(config)
    
    # 测试样例
    test_texts = [
        "没爹的黑孩到处扔",
        "人伦人伦，没听说过狗伦",
        "一群人不等男方回应直接定论",
        "有些人地域黑已经疯了",
        "同性恋不应该受到歧视",
        "这是一个普通的测试句子"
    ]
    
    # 检测
    print("测试结果:")
    for text in test_texts:
        result = detector.detect(text)
        print(f"\n文本: {text}")
        print(f"类别: {result['category']}")
        print(f"仇恨言论: {'是' if result['hateful'] else '否'}")
        print(f"目标对象: {result['target']}")
        print(f"论点: {result['argument']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()