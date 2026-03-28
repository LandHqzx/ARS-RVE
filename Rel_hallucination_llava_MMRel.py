import sys
from tkinter import FALSE
sys.path.insert(0, "/root/autodl-tmp/rel_hallucination/RVE/transformers/src")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
import json
from tqdm import tqdm
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math
from transformers import set_seed

def create_token_mapping(prompt, tokenizer, model=None):
    """为指定问题创建token映射"""
                             
    full_prompt = prompt
    
                  
    full_token_ids_tensor = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                           
    if full_token_ids_tensor.dim() == 2:
        full_token_ids = full_token_ids_tensor[0].tolist()
    else:
        full_token_ids = full_token_ids_tensor.tolist()
    
                                           
    if IMAGE_TOKEN_INDEX not in full_token_ids:
        return None
    
                        
    img_tok_pos = full_token_ids.index(IMAGE_TOKEN_INDEX)
    
                                
    before_img_tokens = tokenizer.convert_ids_to_tokens(full_token_ids[:img_tok_pos])
    after_img_tokens = tokenizer.convert_ids_to_tokens(full_token_ids[img_tok_pos+1:])
    
                                             
    all_entries = []
    current_pos = 0
    num_img_tokens = 576          
    
                           
    for i in range(img_tok_pos):
        token_text = before_img_tokens[i]
        entry = {
            "token_index": current_pos,
            "seq_position": i,
            "token": token_text,
            "is_image_token": False,
            "patch_index": None
        }
        all_entries.append(entry)
        current_pos += 1
    
                                 
    for j in range(num_img_tokens):
        entry = {
            "token_index": current_pos,
            "seq_position": img_tok_pos,
            "token": f"<image_patch_{j}>",
            "is_image_token": True,
            "patch_index": j
        }
        all_entries.append(entry)
        current_pos += 1
    
                           
    for i in range(len(after_img_tokens)):
        token_text = after_img_tokens[i]
        entry = {
            "token_index": current_pos,
            "seq_position": img_tok_pos + 1 + i,
            "token": token_text,
            "is_image_token": False,
            "patch_index": None
        }
        all_entries.append(entry)
        current_pos += 1
    
    return {
        "all_entries": all_entries,
        "image_token_position": img_tok_pos,
        "num_image_tokens": num_img_tokens,
        "full_token_ids": full_token_ids
    }

def find_word_token_positions(word, tokenizer, token_mapping):
    """找到单词在token序列中的所有位置，只在图像token之后的text token中检测"""
    if not word or not token_mapping:
        return []
    
                 
    word_tokens = tokenizer.tokenize(word)
    if not word_tokens:
        return []
    
                     
    cleaned_word_tokens = []
    for token in word_tokens:
                          
        clean_token = token
        if clean_token.startswith('▁'):
            clean_token = clean_token[1:]
        if clean_token.startswith('##'):
            clean_token = clean_token[2:]
        cleaned_word_tokens.append(clean_token)
    
                    
    img_end_idx = None
    all_entries = token_mapping["all_entries"]
    
                      
    for entry in all_entries:
        if entry["is_image_token"]:
            img_end_idx = entry["token_index"]
    
                      
    positions = []
    
    for i, entry in enumerate(all_entries):
                   
        if entry["is_image_token"]:
            continue
        
                                 
        if img_end_idx is not None and entry["token_index"] <= img_end_idx:
            continue
            
        token_text = entry["token"]
                   
        clean_token_text = token_text
        if clean_token_text.startswith('▁'):
            clean_token_text = clean_token_text[1:]
        if clean_token_text.startswith('##'):
            clean_token_text = clean_token_text[2:]
        
                            
        for word_token in cleaned_word_tokens:
            if clean_token_text.lower() == word_token.lower():
                positions.append({
                    "token_index": entry["token_index"],
                    "seq_position": entry["seq_position"],
                    "token": entry["token"],
                    "word": word,
                    "word_token": word_token
                })
                break
    
    return positions

def parse_attention_heads(heads_str):
    """
    解析注意力头参数,支持多种格式:
    - 单个数字: "5"
    - 多个数字: "0,2,5"
    - 范围区间: "1-13"
    - 混合格式: "0,2,5-8,10"
    """
    if not heads_str:
        return []
    
    heads = []
    parts = heads_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
                             
            try:
                start, end = map(int, part.split('-'))
                if start <= end:
                    heads.extend(range(start, end + 1))
                else:
                    print(f"警告: 无效的范围 {part},起始值应小于等于结束值")
            except ValueError:
                print(f"警告: 无法解析范围 {part}")
        else:
                    
            try:
                heads.append(int(part))
            except ValueError:
                print(f"警告: 无法解析数字 {part}")
    
               
    seen = set()
    unique_heads = []
    for head in heads:
        if head not in seen:
            seen.add(head)
            unique_heads.append(head)
    return unique_heads


def build_random_layer_heads_config(model, k, seed=None):
    """从模型中推断层数与每层头数,随机采样k个(层,头)并按层聚合。"""
    import random as _random
    if seed is not None:
        _random.seed(seed)

                 
    num_layers = getattr(getattr(model, 'config', object()), 'num_hidden_layers', None)
    num_heads = getattr(getattr(model, 'config', object()), 'num_attention_heads', None)

                 
    if num_layers is None:
        try:
            num_layers = len(model.model.layers)
        except Exception:
            raise ValueError('无法自动确定模型层数(num_hidden_layers)。')
    if num_heads is None:
        try:
            probe = getattr(model.model.layers[0].self_attn, 'num_heads', None)
            if probe is None:
                probe = getattr(model.model.layers[0].self_attn, 'num_attention_heads', None)
            num_heads = int(probe)
        except Exception:
            raise ValueError('无法自动确定每层注意力头数(num_attention_heads)。')

    total = num_layers * num_heads
    if k > total:
        raise ValueError(f'随机采样数量k={k} 超过可用总数 {total}')

    all_pairs = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    sampled = _random.sample(all_pairs, k)

    cfg = {}
    for l, h in sampled:
        cfg.setdefault(l, []).append(h)
                       

    meta = {
        'num_layers': num_layers,
        'num_heads_per_layer': num_heads,
        'sampled_count': k,
        'sampled_pairs': sampled,
    }
    return cfg, meta

def parse_layer_heads_config(config_str):
    """
    解析多层注意力头配置,支持格式:
    - 单层: "14:0-1,4-8" (第14层,保留0-1,4-8头)
    - 多层: "14:0-1,4-8;15:2-5,10-12" (第14层保留0-1,4-8头,第15层保留2-5,10-12头)
    - 连续层范围: "1-13:0-1,4-8" (第1到13层,每层都保留0-1,4-8头)
    - 混合格式: "1-13:0-1,4-8;15:2-5,10-12" (第1-13层保留0-1,4-8头,第15层保留2-5,10-12头)
    """
    if not config_str:
        return {}
    
    layer_configs = {}
    layer_parts = config_str.split(';')
    
    for layer_part in layer_parts:
        layer_part = layer_part.strip()
        if ':' in layer_part:
            try:
                layer_idx_str, heads_str = layer_part.split(':', 1)
                heads = parse_attention_heads(heads_str.strip())
                
                                    
                if '-' in layer_idx_str:
                    start_layer, end_layer = map(int, layer_idx_str.split('-'))
                    if start_layer <= end_layer:
                                              
                        for layer_idx in range(start_layer, end_layer + 1):
                            layer_configs[layer_idx] = heads
                        print(f"配置层 {start_layer}-{end_layer}: 每层保留注意力头 {heads}")
                    else:
                        print(f"警告: 无效的层范围 {layer_idx_str},起始层应小于等于结束层")
                else:
                         
                    layer_idx = int(layer_idx_str.strip())
                    layer_configs[layer_idx] = heads
                    print(f"配置层 {layer_idx}: 保留注意力头 {heads}")
                    
            except ValueError as e:
                print(f"警告: 无法解析层配置 {layer_part}: {e}")
        else:
            print(f"警告: 层配置格式错误 {layer_part},应为 '层索引:注意力头' 或 '层范围:注意力头'")
    
    return layer_configs


def eval_model(args):
           
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

                        
    question_file_path = os.path.expanduser(args.question_file)
    try:
                        
        with open(question_file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith('['):
                          
                questions = json.loads(content)
                print(f"成功加载JSON数组格式，共{len(questions)}个问题")
            else:
                                     
                f.seek(0)
                questions = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict):
                            questions.append(parsed)
                        else:
                            print(f"警告: 第{line_num}行不是字典格式，已跳过: {parsed}")
                    except json.JSONDecodeError as e:
                        print(f"警告: 第{line_num}行JSON解析失败，已跳过: {e}")
                print(f"成功加载JSONL格式，共{len(questions)}个问题")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        raise
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
             
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    unknown = 0
    total_questions = len(questions)
    yes_answers = 0
            
    inference_times = []                

                 
    manual_layer_heads_config = parse_layer_heads_config(args.layer_heads_config)
    used_layer_heads_config = manual_layer_heads_config
    random_meta = None
    
    if args.use_random_heads:
        used_layer_heads_config, random_meta = build_random_layer_heads_config(model, args.random_k, args.random_seed)
        print("使用随机采样的注意力头配置:")
        for layer_idx, heads in used_layer_heads_config.items():
            print(f"  第{layer_idx}层: 保留注意力头 {heads}")
        print(f"  元信息: 层数={random_meta['num_layers']}, 每层头数={random_meta['num_heads_per_layer']}, 采样数量={random_meta['sampled_count']}")
    else:
        print(f"解析后的多层注意力头配置(手动):")
        for layer_idx, heads in used_layer_heads_config.items():
            print(f"  第{layer_idx}层: 保留注意力头 {heads}")
    
    layer_heads_config = used_layer_heads_config

    for line_idx, line in enumerate(tqdm(questions)):
                     
        if not isinstance(line, dict):
            raise TypeError(f"Line {line_idx} is not a dictionary. Got type: {type(line)}, value: {line}")
                                       
        if "question_id" in line:
            idx = line["question_id"]
        elif "id" in line:
            idx = line["id"]
        elif "idx" in line:
            idx = line["idx"]
        else:
            available_keys = list(line.keys()) if isinstance(line, dict) else "N/A"
            raise KeyError(f"Neither 'question_id' nor 'id' nor 'idx' field found in the data at line {line_idx}. Available keys: {available_keys}. Line content: {line}")
        
        image_file = line["image"]
                                                       
        qs = line.get("text") or line.get("prompt") or line.get("question") or line.get("query")
        if qs is None:
            raise KeyError("Neither 'text', 'prompt', 'question', nor 'query' field found in the data")
        cur_prompt = qs
        
                      
        true_label = line.get("label", None)
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
                                                
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
                                
        token_mapping = create_token_mapping(prompt, tokenizer, model)
        
                                                                   
        relation_word_positions = {}

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

                                      
        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path):
                                        
            image_filename = os.path.basename(image_file)
            image_path = os.path.join(args.image_folder, image_filename)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {os.path.join(args.image_folder, image_file)} 或 {image_path}")
        
        image = Image.open(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                  
        inference_start_time = time.time()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=2,
                use_cache=True,
                use_rel=args.use_rel,
                layer_heads_config=layer_heads_config,
                tokenizer=tokenizer,
                relation_word_positions=relation_word_positions)

                       
        inference_end_time = time.time()
        inference_latency = inference_end_time - inference_start_time
        inference_times.append(inference_latency)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
                         
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "text": outputs,
                                   "label": true_label,
                                   "inference_latency": inference_latency}) + "\n")
        ans_file.flush()
        
                          
        if true_label is not None:
                   
            gt_answer = true_label.lower().strip()
            gen_answer = outputs.lower().strip()
            
                    
            correct = False
            if gt_answer == 'yes':
                if 'yes' in gen_answer:
                    true_pos += 1
                    yes_answers += 1
                    correct = True
                else:
                    false_neg += 1
            elif gt_answer == 'no':
                if 'no' in gen_answer:
                    true_neg += 1
                    correct = True
                else:
                    false_pos += 1
                    yes_answers += 1
            else:
                print(f'Warning: unknown gt_answer: {gt_answer}')
                unknown += 1
            
                      
            current_total = true_pos + true_neg + false_pos + false_neg
            if current_total > 0:
                current_precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                current_recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                current_f1 = 2 * current_precision * current_recall / (current_precision + current_recall) if (current_precision + current_recall) > 0 else 0
                current_accuracy = (true_pos + true_neg) / current_total
                
                print(f"Sample {idx} - Correct: {correct}, GT: {gt_answer}, Gen: {gen_answer}, Latency: {inference_latency:.4f}s")
                print(f"Running Counts - TP: {true_pos}, TN: {true_neg}, FP: {false_pos}, FN: {false_neg}")
                print(f"Current Metrics - Precision: {current_precision:.4f}, Recall: {current_recall:.4f}, F1: {current_f1:.4f}, Accuracy: {current_accuracy:.4f}")
                
                print("-" * 80)
    
    ans_file.close()
    
                      
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    
            
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_pos + true_neg) / total_questions
    yes_proportion = yes_answers / total_questions
    unknown_prop = unknown / total_questions
    
              
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    min_inference_time = min(inference_times) if inference_times else 0
    max_inference_time = max(inference_times) if inference_times else 0
    total_inference_time = sum(inference_times) if inference_times else 0
    
              
    print(f'Final Precision: {precision:.4f}')
    print(f'Final Recall: {recall:.4f}')
    print(f'Final F1: {f1:.4f}')
    print(f'Final Accuracy: {accuracy:.4f}')
    print(f'Yes Proportion: {yes_proportion:.4f}')
    print(f'Unknown Proportion: {unknown_prop:.4f}')
    print(f'True Positives: {true_pos}')
    print(f'True Negatives: {true_neg}')
    print(f'False Positives: {false_pos}')
    print(f'False Negatives: {false_neg}')
    print(f'Unknown: {unknown}')
    print(f'Total Questions: {total_questions}')
    print(f'\nInference Time Statistics:')
    print(f'  Average Latency per Example: {avg_inference_time:.4f}s')
    print(f'  Min Latency: {min_inference_time:.4f}s')
    print(f'  Max Latency: {max_inference_time:.4f}s')
    print(f'  Total Inference Time: {total_inference_time:.4f}s')
    
               
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "yes_proportion": yes_proportion,
        "unknown_proportion": unknown_prop,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg,
        "unknown": unknown,
        "total_questions": total_questions,
        "inference_time": {
            "avg_latency_per_example": avg_inference_time,
            "min_latency": min_inference_time,
            "max_latency": max_inference_time,
            "total_time": total_inference_time,
            "num_samples": len(inference_times)
        }
    }
    
    metrics_file = os.path.join(os.path.dirname(answers_file), "final_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nFinal metrics saved to: {metrics_file}")
    print("="*80)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/rel_hallucination/Models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/root/autodl-tmp/rel_hallucination/datasets/VG_100K")
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/rel_hallucination/RVE/questions/vg_action_all.jsonl")
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/rel_hallucination/RVE/results/RVE/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
                 
    parser.add_argument("--use_rel", action='store_true', default=True, help="是否使用use_rel方法")
    layer_configs = [
         # "0:22,24,7,23,19,18,30,31,26,8,11,4,12,27,3,0,9,6,29,21,15,1,20,16,13,25,10,14,28,2,17,5;"
    # "1:0,9,10,25,17,18,15,14,24,7,19,27,28,23,22,29,13,6,16,1,26,5,11,20,30,3,4,31,8,12,2,21;"
    # "2:30,6,2,17,18,5,0,24,4,21,13,19,11,23,7,14,9,22,25,12,16,8,26,15,1,10,31,20,27,3,28,29;"
    # "3:27,26,0,18,8,5,11,24,10,9,19,21,20,4,14,29,30,16,17,31,7,13,12,25,2,23,6,28,15,3,22,1;"
    # "4:1,5,17,29,20,0,23,8,9,14,12,13,30,16,21,3,22,11,2,28,25,15,6,31,27,18,24,19,10,4,26,7;"
    # "5:15,30,22,19,20,10,31,14,13,29,23,6,1,12,7,24,2,4,18,9,16,11,5,8,26,0,17,28,21,25,27,3;"
    # "6:31,13,1,15,18,28,22,26,29,4,3,2,10,19,17,20,21,25,14,24,5,6,8,12,7,11,30,0,27,9,23,16;"
    # "7:8,22,11,24,5,6,3,21,14,17,31,16,20,18,0,26,29,13,2,19,12,4,7,1,10,25,15,28,30,27,23,9;"
    # "8:17,24,3,1,25,27,30,5,18,29,6,28,11,13,20,23,15,2,12,9,19,16,22,10,4,7,31,14,8,21,0,26;"
    # "9:2,10,0,18,30,5,7,19,20,6,21,24,12,31,17,9,28,25,8,23,26,3,29,11,1,15,22,14,4,16,27,13;"
    # "10:29,11,4,23,19,17,2,8,27,26,10,1,25,6,9,13,30,21,24,15,31,16,20,3,18,14,28,22,5,0,12,7;"
    # "11:17,30,14,7,8,24,26,22,19,31,11,28,3,6,2,4,12,1,16,27,9,0,5,15,29,25,20,13,18,21,10,23;"
    # "12:1,4,25,21,3,9,14,13,27,11,0,16,5,6,12,18,30,28,29,10,31,19,17,24,8,20,26,7,23,2,22,15;"
    # "13:4,2,30,8,22,31,9,18,16,21,10,12,13,23,1,28,20,17,19,26,5,3,7,0,15,24,25,11,14,27,29,6;"                                                                                                                                                                                   
    "14:9,20,2,3,24,26,29,13,19,27,7,4,12,18,17,11,21,16,5,28,15,0,30,1,6,23,22,25,10,14,31,8;"
    "15:31,14,5,27,19,23,22,10,24,17,4,8,29,30,20,18,9,13,12,0,2,25,15,26,11,6,1,28,21,3,7,16;"
    "16:15,5,13,17,2,3,9,0,27,4,31,24,19,7,20,18,29,30,1,25,16,10,23,14,22,6,26,8,28,12,11,21;"
    "17:18,10,30,13,22,31,9,11,0,27,28,2,8,5,24,16,15,25,1,17,20,29,12,6,19,4,23,26,21,7,3,14;"
    "18:31,10,15,30,23,12,1,18,16,11,26,9,6,27,3,21,4,14,25,13,7,8,29,19,28,2,24,17,20,5,22,0;"
    "19:6,10,7,8,19,14,31,4,15,9,12,17,26,5,21,18,25,27,20,30,13,3,28,24,23,16,22,0,2,29,1,11;"
    "20:0,3,5,12,1,10,29,27,11,18,20,8,30,14,17,28,7,31,15,19,25,6,13,21,9,16,2,22,4,24,26,23;"
    "21:31,9,16,10,26,28,1,30,27,13,4,29,15,5,24,14,0,17,22,2,12,20,23,18,6,3,25,7,8,19,11,21;"
    "22:30,16,19,22,27,23,9,8,17,24,26,21,7,11,31,25,20,5,2,6,4,14,28,18,12,10,15,29,0,3,13,1;"
    "23:20,8,31,16,0,7,17,9,3,12,28,4,24,29,25,1,14,19,18,10,30,22,15,26,5,11,6,13,2,21,27,23;"
    "24:14,3,24,15,1,8,17,16,30,11,4,29,5,23,6,10,19,13,25,0,18,31,20,12,9,2,28,22,27,26,7,21;"
    "25:0,3,21,26,17,28,22,7,11,23,16,9,31,5,24,4,18,19,25,6,10,8,15,27,14,1,30,12,2,20,29,13;"
    "26:14,15,21,24,27,26,25,23,28,30,4,3,9,19,6,2,17,18,16,31,12,0,29,13,7,10,22,5,20,11,1,8;"
    "27:12,7,2,29,16,23,20,15,25,9,24,10,22,1,14,8,17,13,28,0,11,19,26,6,31,4,5,18,30,21,3,27;"
    "28:7,25,14,23,13,18,9,17,28,12,27,21,31,6,22,3,10,20,19,0,26,30,5,2,1,4,16,8,11,15,29,24;"
    "29:10,9,19,5,13,21,26,23,17,8,7,30,2,15,20,12,18,16,28,6,24,22,25,4,3,11,27,31,14,29,0,1;"
    "30:12,14,9,2,7,29,31,15,10,26,21,25,17,5,18,19,8,24,23,6,27,11,4,1,16,3,13,22,30,0,20,28;"
    "31:22,27,25,16,19,4,28,31,26,24,13,30,0,3,17,10,1,9,15,11,6,8,7,18,14,12,20,5,23,21,2,29;"
    ]
    
    layer_heads_config_str = ";".join(layer_configs)
    parser.add_argument("--layer-heads-config", type=str, 
                       default=layer_heads_config_str,
                       help="多层注意力头配置,格式:'层索引:注意力头',如 '0:2,3,6;1:0,1,6,10'")
    parser.add_argument("--seed", type=int, default=42)

               
    parser.add_argument("--use-random-heads", action='store_true', default=False,
                       help="是否启用随机选择注意力头模式")
    parser.add_argument("--random-k", type=int, default=100,
                       help="随机选择的注意力头数量")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="随机选择的种子")
    
    args = parser.parse_args()
    set_seed(args.seed)

                               
    from datetime import datetime
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print("开始评估")
    print(f"代码运行开始时间: {start_time_str}")
    print("=" * 80)
    
    try:
              
        eval_model(args)
    finally:
                
        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time

        print("\n评估完成")
        print(f"代码运行结束时间: {end_time_str}")
        print(f"总运行时长: {duration}")
