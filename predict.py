# predict.py
import json
from train import HateSpeechDetector, label_config

def load_config():
    """加载模型配置"""
    return {
        "label_config": label_config,
        "classifier_model_path1": "./models/classifier_v1",
        "classifier_model_path2": "./models/classifier_v2",
        "object_model_path": "./models/object_model_crf",
        "object_model_type": "crf",
        "argument_model_path": "./models/argument_model_crf",
        "argument_model_type": "crf",
        "id2label": {v["id"]: k for k, v in label_config.items()}
    }

def load_test_data(json_path):
    """加载测试数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item["content"] for item in data]

def predict_and_save_results(test_json_path, output_txt_path):
    """预测并保存结果"""
    # 加载配置和模型
    config = load_config()
    detector = HateSpeechDetector(config)
    
    # 加载测试数据
    test_texts = load_test_data(test_json_path)
    
    # 批量预测
    results = detector.batch_detect(test_texts)
    
    # 格式化输出
    output_lines = []
    for result in results:
        line = f"{result['target'] if result['target'] else '未识别'} | " \
               f"{result['argument'] if result['argument'] else '未识别'} | " \
               f"{result['category']} | " \
               f"{'hate' if result['hateful'] else 'non-hate'} [END]"
        output_lines.append(line)
    
    # 保存结果
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    print(f"预测结果已保存到 {output_txt_path}")

if __name__ == "__main__":
    # 输入输出路径配置
    test_json_path = "test1.json"  # 测试集路径
    output_txt_path = "predictions.txt"  # 输出文件路径
    
    # 执行预测
    predict_and_save_results(test_json_path, output_txt_path)