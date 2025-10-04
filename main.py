import json
from pathlib import Path
from typing import Dict, Tuple, List

PROMPT_TEMPLATE_PATH = "./PROMPT/prompt_template.txt"

def ground_truth_and_prediction_to_prompt(ground_truth: str, prediction: str) -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("$ground_truth$", ground_truth).replace("$prediction$", prediction)


def load_jsonl(file_path: str) -> Dict[str, dict]:
    """JSONLファイルを読み込み、QA_numberをキーとした辞書を返す"""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                qa_number = item.get('QA_number')
                if qa_number:
                    data[qa_number] = item
    return data


def get_answer_pairs(ground_truth_path: str, prediction_path: str) -> List[Tuple[str, str, str]]:
    """
    同じQA-IDを持つanswer対を取得する
    
    Returns:
        List[Tuple[str, str, str]]: (QA_number, ground_truth_answer, predicted_answer)のリスト
    """
    # 両方のファイルを読み込み
    ground_truth_data = load_jsonl(ground_truth_path)
    prediction_data = load_jsonl(prediction_path)
    
    answer_pairs = []
    
    # 共通のQA_numberを見つけてanswer対を作成
    common_qa_numbers = set(ground_truth_data.keys()) & set(prediction_data.keys())
    
    for qa_number in sorted(common_qa_numbers):
        gt_answer = ground_truth_data[qa_number].get('Answer', '')
        pred_answer = prediction_data[qa_number].get('Answer', '')
        answer_pairs.append((qa_number, gt_answer, pred_answer))
    
    return answer_pairs


def main():
    # ファイルパスを設定
    judge_dir = Path("JUDGE")
    ground_truth_path = judge_dir / "ground_trush.jsonl"
    prediction_path = judge_dir / "predict_dep_qwen_3B.jsonl"
    
    # answer対を取得
    answer_pairs = get_answer_pairs(str(ground_truth_path), str(prediction_path))
    
    print(f"取得したanswer対の数: {len(answer_pairs)}")
    print("-" * 80)
    
    # 最初の5つのペアを表示
    for i, (qa_number, gt_answer, pred_answer) in enumerate(answer_pairs[:5]):
        print(f"QA Number: {qa_number}")
        print(f"Ground Truth: {gt_answer}")
        print(f"Prediction:   {pred_answer}")
        print("-" * 80)
    
    # 全てのペアを表示したい場合は以下のコメントアウトを外してください
    # for qa_number, gt_answer, pred_answer in answer_pairs:
    #     print(f"QA Number: {qa_number}")
    #     print(f"Ground Truth: {gt_answer}")
    #     print(f"Prediction:   {pred_answer}")
    #     print("-" * 80)


if __name__ == "__main__":
    main()
