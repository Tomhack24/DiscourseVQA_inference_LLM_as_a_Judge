import os
import json
from pathlib import Path
from typing import Dict, Tuple, List
from openai import OpenAI
from dotenv import load_dotenv


PROMPT_TEMPLATE_PATH = "./PROMPT/llm-as-a-judge_prompt.txt"
MODEL = "gpt-4o"
load_dotenv()


def ground_truth_and_prediction_to_prompt(question: str, ground_truth: str, prediction: str) -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("$QUESTION$", question).replace("$GROUND_TRUTH$", ground_truth).replace("$PREDICTION$", prediction)


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


def get_answer_pairs(ground_truth_path: str, prediction_path: str) -> List[Tuple[str, str, str, str]]:
    """
    同じQA-IDを持つanswer対を取得する
    
    Returns:
        List[Tuple[str, str, str, str]]: (QA_number, Question, ground_truth_answer, predicted_answer)のリスト
    """
    # 両方のファイルを読み込み
    ground_truth_data = load_jsonl(ground_truth_path)
    prediction_data = load_jsonl(prediction_path)
    
    answer_pairs = []
    
    # 共通のQA_numberを見つけてanswer対を作成
    common_qa_numbers = set(ground_truth_data.keys()) & set(prediction_data.keys())
    
    for qa_number in sorted(common_qa_numbers):
        question = ground_truth_data[qa_number].get('Question', '')
        gt_answer = ground_truth_data[qa_number].get('Answer', '')
        pred_answer = prediction_data[qa_number].get('Answer', '')
        answer_pairs.append((qa_number, question, gt_answer, pred_answer))
    
    return answer_pairs


def gpt4o_call(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=128,
        temperature=0.0,
    )
    return response.choices[0].message.content



def main():
    # ファイルパスを設定    
    judge_dir = Path("JUDGE")
    ground_truth_path = judge_dir / "ground_truth.jsonl"
    prediction_path = judge_dir / "predict_dep_qwen_32B.jsonl"
    
    # answer対を取得
    answer_pairs = get_answer_pairs(str(ground_truth_path), str(prediction_path))
    
    print(f"取得したanswer対の数: {len(answer_pairs)}")
    print("-" * 80)
    
    response = []
    for i, (qa_number, question, gt_answer, pred_answer) in enumerate(answer_pairs[:5]):
        judge = gpt4o_call(ground_truth_and_prediction_to_prompt(question, gt_answer, pred_answer))
        print(f"QA {qa_number}: {judge}")
        response.append(judge)
        print("-" * 80)



if __name__ == "__main__":
    main()
