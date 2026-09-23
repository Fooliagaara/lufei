import os
import pickle
import torch

# 让 data / saved_dict 的相对路径始终相对本脚本所在目录解析，避免受启动位置影响
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from models.textCNN import Model, Config


dataset = "toutiao"
config = Config(dataset)

# 训练时 n_vocab 是在 build_dataset_CNN 里赋值的，这里同样从词表读取
with open(config.vocab_path, "rb") as f:
    vocab = pickle.load(f)
config.n_vocab = len(vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 构建模型并加载训练好的权重 ----
model = Model(config).to(device)
checkpoint = torch.load(config.save_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

print(f"已加载模型：{config.save_path}")
print(f"词表大小：{config.n_vocab}，类别数：{config.num_classes}")
print(f"类别：{config.class_list}")


def _text_to_ids(text, pad_size=config.pad_size):
    """和训练时一致的预处理：char 级分词 -> 截断/填充 -> 映射为 id"""
    token = list(text)
    if len(token) < pad_size:
        token += ["[PAD]"] * (pad_size - len(token))
    else:
        token = token[:pad_size]
    return [vocab.get(ch, vocab["[UNK]"]) for ch in token]


def text_to_input(text):
    """把单条文本转成模型输入；返回元组 (x, seq_len)，与训练时 DatasetIterater 的格式一致"""
    x = torch.LongTensor(_text_to_ids(text)).unsqueeze(0).to(device)  # [1, pad_size]
    seq_len = torch.LongTensor([min(len(text), config.pad_size)]).to(device)
    return (x, seq_len)


def predict(text):
    """返回 (类别名, 预测索引, 输出logits)"""
    inputs = text_to_input(text)
    with torch.inference_mode():
        outputs = model(inputs)
    pred = torch.argmax(outputs, dim=1).item()
    return config.class_list[pred], pred, outputs


def evaluate_test_set(batch_size=256):
    """在测试集上统计准确率，验证模型加载与预测是否正确"""
    texts, labels = [], []
    with open(config.test_path, "r", encoding="UTF-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            content, label = line.split("\t")
            texts.append(content)
            labels.append(int(label))

    correct, total = 0, len(texts)
    with torch.inference_mode():
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            x = torch.LongTensor([_text_to_ids(t) for t in batch_texts]).to(device)
            seq_len = torch.LongTensor([min(len(t), config.pad_size) for t in batch_texts]).to(device)
            outputs = model((x, seq_len))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            correct += sum(int(p == l) for p, l in zip(preds, batch_labels))
    print(f"测试集准确率：{correct}/{total} = {correct / total:.2%}")


if __name__ == "__main__":
    sample = "2011年高考文科综合试题(重庆卷)"
    label, pred, outputs = predict(sample)
    print(f"\n输入文本：{sample}")
    print(f"模型输出：{outputs}")
    print(f"预测类别：{label}（索引 {pred}）")

    evaluate_test_set()
