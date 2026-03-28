import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
        
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False          
import seaborn
import argparse
from PIL import Image
from transformers import set_seed
import re
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import spacy
import json
from tqdm import tqdm
import shutil
from collections import defaultdict
import random
import seaborn as sns

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

                                        
MODEL_PATH = '/root/autodl-tmp/rel_hallucination/Models/llava-v1.5-7b'
IMAGE_ROOT = '/root/autodl-tmp/rel_hallucination/datasets/VG_100K'
DATA_FILE = '/root/autodl-tmp/rel_hallucination/RVE/For_ARS_metric.jsonl'              
                                         
OUTPUT_BASE_PATH_TEMPLATE = '/root/autodl-tmp/rel_hallucination/RVE'            

NUM_IMG_TOKENS = 576
NUM_PATCHES = 24
NUM_SAMPLES = None                                    

      
ANALYZE_HEAD_ATTENTION_DIFFERENCES = True              
COMPARE_ATTENTION_PATTERNS = True                    
SAVE_PAIR_OUTPUTS = False                                  

                                                                      
SORT_LAST_TOKEN_LAYER_ONE_BASED = False                     
SORT_LAST_TOKEN_HEAD_ONE_BASED = False                      
SORT_LAST_TOKEN_SELECT_MIN = False                            

                      
nlp = spacy.load('en_core_web_sm')


def create_token_mapping(prompt, tokenizer, model=None):                
    if model is not None:
        full_prompt, _ = build_prompt(prompt, model, 'llava_v1')
    else:
                                
        full_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
                  
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
    
                           
    for i, token_text in enumerate(after_img_tokens):
        entry = {
            "token_index": current_pos,
            "seq_position": img_tok_pos + 1 + i,
            "token": token_text,
            "is_image_token": False,
            "patch_index": None
        }
        all_entries.append(entry)
        current_pos += 1
    
                                            
    svo_words = extract_svo(prompt)
    word_entries = {}
    
    for w in svo_words:
        subtoks = tokenizer.tokenize(w)
        entries = []
        for entry in all_entries:
            if entry["token"] in subtoks:
                entries.append((entry["token_index"], entry["token"]))
        if entries:
            word_entries[w] = entries
    
    return {
        "original_token_ids": full_token_ids,
        "all_entries": all_entries,
        "image_token_position": img_tok_pos,
        "num_image_tokens": num_img_tokens,
        "word_entries": word_entries
    }

def calculate_svo_image_attention_difference(correct_head_attn, incorrect_head_attn, 
                                           correct_token_mapping, incorrect_token_mapping,
                                           correct_svo, incorrect_svo, img_pos, num_img_tokens):
    """计算SVO token对图像token的注意力差异"""
    if not correct_svo or not incorrect_svo or len(correct_svo) < 3 or len(incorrect_svo) < 3:
        return None
    
             
    correct_subject, correct_verb, correct_object = correct_svo[0], correct_svo[1], correct_svo[2]
    incorrect_subject, incorrect_verb, incorrect_object = incorrect_svo[0], incorrect_svo[1], incorrect_svo[2]
    
                   
    correct_subject_attn = calculate_word_image_attention(correct_head_attn, correct_token_mapping, correct_subject, img_pos, num_img_tokens)
    correct_verb_attn = calculate_word_image_attention(correct_head_attn, correct_token_mapping, correct_verb, img_pos, num_img_tokens)
    correct_object_attn = calculate_word_image_attention(correct_head_attn, correct_token_mapping, correct_object, img_pos, num_img_tokens)
    
                   
    incorrect_subject_attn = calculate_word_image_attention(incorrect_head_attn, incorrect_token_mapping, incorrect_subject, img_pos, num_img_tokens)
    incorrect_verb_attn = calculate_word_image_attention(incorrect_head_attn, incorrect_token_mapping, incorrect_verb, img_pos, num_img_tokens)
    incorrect_object_attn = calculate_word_image_attention(incorrect_head_attn, incorrect_token_mapping, incorrect_object, img_pos, num_img_tokens)
    
          
    subject_diff = incorrect_subject_attn - correct_subject_attn
    verb_diff = incorrect_verb_attn - correct_verb_attn
    object_diff = incorrect_object_attn - correct_object_attn
    
                                                                                
    eps = 0
             
    if isinstance(incorrect_subject_attn, torch.Tensor):
        subj_inc_tensor = incorrect_subject_attn.to(torch.float32)
    else:
        subj_inc_tensor = torch.tensor(incorrect_subject_attn, dtype=torch.float32)
    if isinstance(correct_subject_attn, torch.Tensor):
        subj_cor_tensor = correct_subject_attn.to(torch.float32)
    else:
        subj_cor_tensor = torch.tensor(correct_subject_attn, dtype=torch.float32)
    if isinstance(subject_diff, torch.Tensor):
        subj_diff_tensor = subject_diff.to(torch.float32)
    else:
        subj_diff_tensor = torch.tensor(subject_diff, dtype=torch.float32)
    subject_inc_norm = torch.norm(subj_inc_tensor, p='fro')
    subject_cor_norm = torch.norm(subj_cor_tensor, p='fro')
    subject_diff_norm = torch.norm(subj_diff_tensor, p='fro')
    subject_denom = (subject_inc_norm + subject_cor_norm) * 0.5 + eps
    subject_frobenius = (subject_diff_norm / subject_denom).item()
          
    if isinstance(incorrect_verb_attn, torch.Tensor):
        verb_inc_tensor = incorrect_verb_attn.to(torch.float32)
    else:
        verb_inc_tensor = torch.tensor(incorrect_verb_attn, dtype=torch.float32)
    if isinstance(correct_verb_attn, torch.Tensor):
        verb_cor_tensor = correct_verb_attn.to(torch.float32)
    else:
        verb_cor_tensor = torch.tensor(correct_verb_attn, dtype=torch.float32)
    if isinstance(verb_diff, torch.Tensor):
        verb_diff_tensor = verb_diff.to(torch.float32)
    else:
        verb_diff_tensor = torch.tensor(verb_diff, dtype=torch.float32)
    verb_inc_norm = torch.norm(verb_inc_tensor, p='fro')
    verb_cor_norm = torch.norm(verb_cor_tensor, p='fro')
    verb_diff_norm = torch.norm(verb_diff_tensor, p='fro')
    verb_denom = (verb_inc_norm + verb_cor_norm) * 0.5 + eps
    verb_frobenius = (verb_diff_norm / verb_denom).item()
            
    if isinstance(incorrect_object_attn, torch.Tensor):
        obj_inc_tensor = incorrect_object_attn.to(torch.float32)
    else:
        obj_inc_tensor = torch.tensor(incorrect_object_attn, dtype=torch.float32)
    if isinstance(correct_object_attn, torch.Tensor):
        obj_cor_tensor = correct_object_attn.to(torch.float32)
    else:
        obj_cor_tensor = torch.tensor(correct_object_attn, dtype=torch.float32)
    if isinstance(object_diff, torch.Tensor):
        obj_diff_tensor = object_diff.to(torch.float32)
    else:
        obj_diff_tensor = torch.tensor(object_diff, dtype=torch.float32)
    object_inc_norm = torch.norm(obj_inc_tensor, p='fro')
    object_cor_norm = torch.norm(obj_cor_tensor, p='fro')
    object_diff_norm = torch.norm(obj_diff_tensor, p='fro')
    object_denom = (object_inc_norm + object_cor_norm) * 0.5 + eps
    object_frobenius = (object_diff_norm / object_denom).item()
    
    return {
        'subject': subject_frobenius,
        'verb': verb_frobenius,
        'object': object_frobenius
    }

def calculate_word_image_attention(head_attn, token_mapping, word, img_pos, num_img_tokens):
    """计算指定单词对图像token的平均注意力，使用预计算的word_entries"""
                  
    clean_word = re.sub(r'[^\w\s]', '', word).strip()
    
                                  
    if clean_word in token_mapping['word_entries']:
        word_entries = token_mapping['word_entries'][clean_word]
        word_token_positions = [entry[0] for entry in word_entries]                 
    else:
                       
        word_token_positions = []
        for entry in token_mapping['all_entries']:
            if not entry['is_image_token'] and clean_word.lower() in entry['token'].lower():
                word_token_positions.append(entry['token_index'])
    
    if not word_token_positions:
        return 0.0
    
                             
    total_attention = 0.0
    count = 0
    
    for word_pos in word_token_positions:
        if word_pos < head_attn.shape[0]:
                                  
                                                
            if word_pos < head_attn.shape[0] and img_pos + num_img_tokens <= head_attn.shape[1]:
                img_attention = head_attn[word_pos, img_pos:img_pos+num_img_tokens]
                total_attention += img_attention
                count += 1
    
    if count > 0:
        avg_attention = total_attention / count
        return avg_attention
    else:
        return 0.0

def calculate_last_token_attention_differences(correct_head_attn, incorrect_head_attn, 
                                            correct_token_mapping, incorrect_token_mapping):
    """计算last token与其他文字token的注意力差值以及与图像token的平均注意力差值"""
    
            
    seq_len = correct_head_attn.shape[0]
    
                                            
    correct_last_token_pos = correct_token_mapping['all_entries'][-1]['token_index']
    incorrect_last_token_pos = incorrect_token_mapping['all_entries'][-1]['token_index']
    
                                                  
    correct_last_to_text_attn = []
    incorrect_last_to_text_attn = []
    
                                       
    for entry in correct_token_mapping['all_entries']:
        if not entry['is_image_token']:
            if entry['token_index'] < seq_len and correct_last_token_pos < seq_len:
                attn_value = correct_head_attn[correct_last_token_pos, entry['token_index']].item()
                correct_last_to_text_attn.append({
                    'token_index': entry['token_index'],
                    'token': entry['token'],
                    'attention': attn_value,
                    'is_self': entry['token_index'] == correct_last_token_pos
                })
    
    for entry in incorrect_token_mapping['all_entries']:
        if not entry['is_image_token']:
            if entry['token_index'] < seq_len and incorrect_last_token_pos < seq_len:
                attn_value = incorrect_head_attn[incorrect_last_token_pos, entry['token_index']].item()
                incorrect_last_to_text_attn.append({
                    'token_index': entry['token_index'],
                    'token': entry['token'],
                    'attention': attn_value,
                    'is_self': entry['token_index'] == incorrect_last_token_pos
                })
    
                                
    correct_last_to_img_attn = 0.0
    incorrect_last_to_img_attn = 0.0
    
    img_pos = correct_token_mapping['image_token_position']
    num_img_tokens = correct_token_mapping['num_image_tokens']
    
    if correct_last_token_pos < seq_len and img_pos + num_img_tokens <= seq_len:
        correct_last_to_img_attn = correct_head_attn[correct_last_token_pos, img_pos:img_pos+num_img_tokens].mean().item()
    
    if incorrect_last_token_pos < seq_len and img_pos + num_img_tokens <= seq_len:
        incorrect_last_to_img_attn = incorrect_head_attn[incorrect_last_token_pos, img_pos:img_pos+num_img_tokens].mean().item()
    
                       
    text_attention_diffs = []
    
                                     
    correct_attn_map = {item['token_index']: item['attention'] for item in correct_last_to_text_attn}
    incorrect_attn_map = {item['token_index']: item['attention'] for item in incorrect_last_to_text_attn}
    
                     
    for item in correct_last_to_text_attn:
        token_index = item['token_index']
        token = item['token']
        is_self = item['is_self']
        
        if token_index in incorrect_attn_map:
                                    
            attn_diff = item['attention'] - incorrect_attn_map[token_index]            
            text_attention_diffs.append({
                'token_index': token_index,
                'token': token,
                'is_self': is_self,
                'correct_attention': item['attention'],
                'incorrect_attention': incorrect_attn_map[token_index],
                'attention_diff': attn_diff
            })
        else:
                               
            text_attention_diffs.append({
                'token_index': token_index,
                'token': token,
                'is_self': is_self,
                'correct_attention': item['attention'],
                'incorrect_attention': 0.0,
                'attention_diff': item['attention']                 
            })
    
                             
    for item in incorrect_last_to_text_attn:
        token_index = item['token_index']
        if token_index not in correct_attn_map:
                               
            text_attention_diffs.append({
                'token_index': token_index,
                'token': item['token'],
                'is_self': item['is_self'],
                'correct_attention': 0.0,
                'incorrect_attention': item['attention'],
                'attention_diff': -item['attention']                  
            })
    
                          
    total_text_diff = sum(item['attention_diff'] for item in text_attention_diffs)
    
                
    last_to_img_diff = correct_last_to_img_attn - incorrect_last_to_img_attn            
    
    return {
        'last_to_text_attention_diff': total_text_diff,              
        'last_to_image_attention_diff': last_to_img_diff,           
        'text_attention_diffs': text_attention_diffs,                  
        'correct_last_to_text': correct_last_to_text_attn,                     
        'incorrect_last_to_text': incorrect_last_to_text_attn,                     
        'correct_last_to_image': correct_last_to_img_attn,
        'incorrect_last_to_image': incorrect_last_to_img_attn,
        'last_token_positions': {
            'correct': correct_last_token_pos,
            'incorrect': incorrect_last_token_pos
        },
        'text_attention_summary': {
            'total_difference': total_text_diff,
            'num_tokens': len(text_attention_diffs),
            'self_attention_diff': next((item['attention_diff'] for item in text_attention_diffs if item['is_self']), 0.0)
        }
    }


                                                             
def calculate_last_token_image_attention(head_attn, token_mapping, img_pos, num_img_tokens):
    """计算last token对图像token的注意力"""
    seq_len = head_attn.shape[0]
    last_token_pos = -1
    if last_token_pos < seq_len and img_pos + num_img_tokens <= seq_len:
        return head_attn[last_token_pos, img_pos:img_pos+num_img_tokens]
    return torch.zeros(num_img_tokens)


def calculate_last_token_image_attention_difference(correct_head_attn, incorrect_head_attn,
                                                    correct_token_mapping, incorrect_token_mapping,
                                                    img_pos, num_img_tokens):
    """计算last token对图像token的注意力差异（Frobenius归一化）"""
    correct_last_token_attn = calculate_last_token_image_attention(correct_head_attn, correct_token_mapping, img_pos, num_img_tokens)
    incorrect_last_token_attn = calculate_last_token_image_attention(incorrect_head_attn, incorrect_token_mapping, img_pos, num_img_tokens)

    last_token_diff = incorrect_last_token_attn - correct_last_token_attn
    eps = 1e-8

    last_inc_tensor = incorrect_last_token_attn.to(torch.float32) if isinstance(incorrect_last_token_attn, torch.Tensor) else torch.tensor(incorrect_last_token_attn, dtype=torch.float32)
    last_cor_tensor = correct_last_token_attn.to(torch.float32) if isinstance(correct_last_token_attn, torch.Tensor) else torch.tensor(correct_last_token_attn, dtype=torch.float32)
    last_diff_tensor = last_token_diff.to(torch.float32) if isinstance(last_token_diff, torch.Tensor) else torch.tensor(last_token_diff, dtype=torch.float32)

    last_inc_norm = torch.norm(last_inc_tensor, p='fro')
    last_cor_norm = torch.norm(last_cor_tensor, p='fro')
    last_diff_norm = torch.norm(last_diff_tensor, p='fro')
    last_denom = (last_inc_norm + last_cor_norm) * 0.5 + eps
    last_token_frobenius = (last_diff_norm / last_denom).item()

    return {'last_token': last_token_frobenius}

def extract_svo(prompt: str):
    """提取SVO（主语-谓语-宾语）"""
    doc = nlp(prompt)
    subj = verb = obj = None
    for token in doc:
        if token.dep_ == 'nsubj' and subj is None:
            subj = token.text
        if token.dep_ == 'ROOT' and verb is None:
            verb = token.text
        if token.dep_ in ('dobj', 'obj') and obj is None:
            obj = token.text
    return [w for w in (subj, verb, obj) if w]

def build_prompt(qs, model, conv_mode='llava_v1'):
    """构建提示词"""
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv

def create_contrastive_question(original_question, answer):
    """通过替换动词创建对比问题（与llava-next脚本保持一致）"""
    clean_answer = answer.strip().lower()
    verb = clean_answer.split()[0] if clean_answer else ""
    if not verb:
        return None
    doc = nlp(original_question)
    new_tokens = []
    verb_replaced = False
    manual_verb_lemmas = {"lie", "crouch", "drink", "dribble"}
    manual_verb_forms = {"lie", "lies", "lying", "lay", "lain", "drink", "drinks", "drinking", "drank", "drunk", "dribble", "dribbles", "dribbling", "dribbled"}
    for token in doc:
        is_manual_verb = token.lemma_.lower() in manual_verb_lemmas or token.text.lower() in manual_verb_forms
        if not verb_replaced and (token.pos_ == 'VERB' or is_manual_verb):
            new_tokens.append(verb)
            verb_replaced = True
        else:
            new_tokens.append(token.text)
    if verb_replaced:
        contrastive_question = ' '.join(new_tokens)
        if not contrastive_question.endswith('?'):
            contrastive_question += '?'
        return contrastive_question
    return None


def load_and_group_data(data_file):
    """从answers.jsonl加载数据，为每个question_id创建对比问题"""
    with open(data_file, 'r') as f:
        lines = f.readlines()

    grouped_data = defaultdict(list)
    for line in lines:
        entry = json.loads(line)
        question_id = entry['question_id']
        grouped_data[question_id].append(entry)

    valid_groups = {}
    for question_id, entries in grouped_data.items():
        if len(entries) >= 1:
            original_entry = entries[0]
            original_question = original_entry.get('question', original_entry.get('text', ''))
            answer = original_entry.get('answer', '')
            if original_question and answer:
                contrastive_question = create_contrastive_question(original_question, answer)
                if contrastive_question and contrastive_question != original_question:
                    valid_groups[question_id] = {
                        'original': {
                            'question': original_question,
                            'label': original_entry.get('label', 'yes'),
                            'image': original_entry.get('image', ''),
                            'question_id': question_id
                        },
                        'contrastive': {
                            'question': contrastive_question,
                            'label': original_entry.get('label', 'yes'),
                            'image': original_entry.get('image', ''),
                            'question_id': question_id
                        },
                        'answer': answer,
                        'image': original_entry.get('image', '')
                    }
                else:
                    print(f"出现无效样本对: question_id={question_id}, original_question={original_question}, answer={answer}, contrastive_question={contrastive_question}")
    print(f"找到 {len(valid_groups)} 个有效对比样本")
    return valid_groups

def process_question_pair(question_id, question_data, tokenizer, model, img_proc, device, save_pair_outputs=SAVE_PAIR_OUTPUTS):
    """处理一个question_id对应的原始与对比问题样例"""
    original_question = question_data['original']
    contrastive_question = question_data['contrastive']
    image_path = question_data['image']

    print(f"处理样本对: {question_id}")

    img_rel = image_path
    img_path = os.path.join(IMAGE_ROOT, img_rel)
    image_id = os.path.splitext(os.path.basename(img_rel))[0]

    output_dir = os.path.join(OUTPUT_BASE_PATH, f'question_{question_id}', image_id)
    if save_pair_outputs:
        ensure_dir(output_dir)
        shutil.copy(img_path, os.path.join(output_dir, 'original.png'))

    question_results = {}
    first_result = process_single_question(original_question, img_path, model, tokenizer, img_proc, device, 1, output_dir, save_outputs=save_pair_outputs)
    second_result = process_single_question(contrastive_question, img_path, model, tokenizer, img_proc, device, 2, output_dir, save_outputs=save_pair_outputs)
    question_results['question_1'] = first_result
    question_results['question_2'] = second_result

                                   
    if first_result['model_answer'].lower() == first_result['expected_answer'].lower():
        correct_result = first_result
        incorrect_result = second_result
    else:
        correct_result = second_result
        incorrect_result = first_result

    results = {
        'correct': {
            'attentions': correct_result['attentions'],
            'prompt': correct_result['prompt']
        },
        'incorrect': {
            'attentions': incorrect_result['attentions'],
            'prompt': incorrect_result['prompt']
        }
    }

    questions_info = {
        'question_id': question_id,
        'correct_question': {
            'prompt': correct_result['prompt'],
            'label': correct_result['expected_answer'],
            'model_answer': correct_result['model_answer'],
            'svo': extract_svo(correct_result['prompt'])
        },
        'incorrect_question': {
            'prompt': incorrect_result['prompt'],
            'label': incorrect_result['expected_answer'],
            'model_answer': incorrect_result['model_answer'],
            'svo': extract_svo(incorrect_result['prompt'])
        }
    }

    if save_pair_outputs:
        with open(os.path.join(output_dir, 'questions_info.json'), 'w') as f:
            json.dump(questions_info, f, indent=2)

    return results


def process_question_pair_and_compute_frobenius(question_id, question_data, tokenizer, model, img_proc, device,
                                                image_root, output_base_path,
                                                frobenius_accumulator,
                                                num_layers, num_heads, conv_mode='llava_v1',
                                                save_pair_outputs=SAVE_PAIR_OUTPUTS):
    """按对处理样本并立即累积Frobenius归一化值（仿llava-next对比逻辑）"""
    first_question = question_data['original']
    second_question = question_data['contrastive']
    image_path = question_data['image']

    print(f"处理样本对: {question_id}")

    img_rel = image_path
    img_path = os.path.join(image_root, img_rel)

    image_id = os.path.splitext(os.path.basename(img_rel))[0]
    output_dir = os.path.join(output_base_path, f'question_{question_id}', image_id)
    if save_pair_outputs:
        ensure_dir(output_dir)
        shutil.copy(img_path, os.path.join(output_dir, 'original.png'))

    question_results = {}
    first_result = process_single_question(first_question, img_path, model, tokenizer, img_proc, device, 1, output_dir, save_outputs=save_pair_outputs)
    second_result = process_single_question(second_question, img_path, model, tokenizer, img_proc, device, 2, output_dir, save_outputs=save_pair_outputs)
    question_results['question_1'] = first_result
    question_results['question_2'] = second_result

                                   
    if first_result['model_answer'].lower() == first_result['expected_answer'].lower():
        correct_result = first_result
        incorrect_result = second_result
    else:
        correct_result = second_result
        incorrect_result = first_result

                       
    correct_token_mapping = create_token_mapping(correct_result['prompt'], tokenizer, model)
    incorrect_token_mapping = create_token_mapping(incorrect_result['prompt'], tokenizer, model)
    if correct_token_mapping is None:
        print(f"[ERROR] 样本 {question_id} 失败: correct_token_mapping 为 None, prompt={correct_result['prompt']}")
        return False
    if incorrect_token_mapping is None:
        print(f"[ERROR] 样本 {question_id} 失败: incorrect_token_mapping 为 None, prompt={incorrect_result['prompt']}")
        return False

                                    
    for layer_idx in range(num_layers):
        correct_layer_attn = correct_result['attentions'][layer_idx]
        incorrect_layer_attn = incorrect_result['attentions'][layer_idx]

        if correct_layer_attn.dim() == 3:
            pass
        elif correct_layer_attn.dim() == 4:
            correct_layer_attn = correct_layer_attn[0]
            incorrect_layer_attn = incorrect_layer_attn[0]
        else:
            print(f'[Warning] 意外的注意力形状: {correct_layer_attn.shape}')
            continue

        for head_idx in range(num_heads):
            correct_head_attn = correct_layer_attn[head_idx]
            incorrect_head_attn = incorrect_layer_attn[head_idx]

            last_token_diff = calculate_last_token_image_attention_difference(
                correct_head_attn, incorrect_head_attn,
                correct_token_mapping, incorrect_token_mapping,
                correct_token_mapping['image_token_position'], correct_token_mapping['num_image_tokens']
            )

            if last_token_diff is not None:
                frobenius_accumulator['last_token'][layer_idx][head_idx].append(last_token_diff['last_token'])

             
    del correct_result['attentions']
    del incorrect_result['attentions']
    if device == 'cuda':
        torch.cuda.empty_cache()

    return True

def calculate_averaged_svo_attentions(all_samples_results, tokenizer, model):
    """计算所有样本的SVO token注意力平均值"""
    
                     
    first_sample = next(iter(all_samples_results.values()))
    num_layers = len(first_sample['correct']['attentions'])
    num_heads = first_sample['correct']['attentions'][0][0].shape[0]        
    
                                
                         
    svo_attn_diff_sum = {
        'subject': [[[] for _ in range(num_heads)] for _ in range(num_layers)],
        'verb': [[[] for _ in range(num_heads)] for _ in range(num_layers)],
        'object': [[[] for _ in range(num_heads)] for _ in range(num_layers)]
    }
    
                           
                                    
    last_token_attn_diff_sum = {
        'text_tokens': [[[] for _ in range(num_heads)] for _ in range(num_layers)],                                
        'image_average': [[[] for _ in range(num_heads)] for _ in range(num_layers)]                               
    }
    
                                  
    svo_attn_sum = {
        'correct': {
            'subject': np.zeros((num_layers, num_heads)),
            'verb': np.zeros((num_layers, num_heads)),
            'object': np.zeros((num_layers, num_heads))
        },
        'incorrect': {
            'subject': np.zeros((num_layers, num_heads)),
            'verb': np.zeros((num_layers, num_heads)),
            'object': np.zeros((num_layers, num_heads))
        }
    }
    
            
    num_samples = len(all_samples_results)
    for sample_id, sample_results in all_samples_results.items():
        correct_prompt = sample_results['correct']['prompt']
        incorrect_prompt = sample_results['incorrect']['prompt']
        
                           
        correct_token_mapping = create_token_mapping(correct_prompt, tokenizer, model)
        incorrect_token_mapping = create_token_mapping(incorrect_prompt, tokenizer, model)
        
        if correct_token_mapping is None or incorrect_token_mapping is None:
            continue
        
               
        correct_svos = extract_svo(correct_prompt)
        incorrect_svos = extract_svo(incorrect_prompt)
        
                             
        for layer_idx in range(num_layers):
            correct_layer_attns = sample_results['correct']['attentions'][layer_idx]                                 
            incorrect_layer_attns = sample_results['incorrect']['attentions'][layer_idx]
            
            for head_idx in range(num_heads):
                correct_head_attn = correct_layer_attns[0][head_idx]                                         
                incorrect_head_attn = incorrect_layer_attns[0][head_idx]                                         
                
                                               
                svo_img_attn_diff = calculate_svo_image_attention_difference(
                    correct_head_attn, incorrect_head_attn,                       
                    correct_token_mapping, incorrect_token_mapping,
                    correct_svos, incorrect_svos,
                    correct_token_mapping['image_token_position'], correct_token_mapping['num_image_tokens']
                )
                
                if svo_img_attn_diff is not None:
                                   
                    svo_attn_diff_sum['subject'][layer_idx][head_idx].append(svo_img_attn_diff['subject'])
                    svo_attn_diff_sum['verb'][layer_idx][head_idx].append(svo_img_attn_diff['verb'])
                    svo_attn_diff_sum['object'][layer_idx][head_idx].append(svo_img_attn_diff['object'])
                
                                   
                last_token_attn_diff = calculate_last_token_attention_differences(
                    correct_head_attn, incorrect_head_attn,
                    correct_token_mapping, incorrect_token_mapping
                )
                
                if last_token_attn_diff is not None:
                                                      
                    for text_diff in last_token_attn_diff['text_attention_diffs']:
                        text_diff_copy = text_diff.copy()            
                        text_diff_copy['layer'] = layer_idx
                        text_diff_copy['head'] = head_idx
                        last_token_attn_diff_sum['text_tokens'][layer_idx][head_idx].append(text_diff_copy)
                    
                                                       
                    image_diff_info = {
                        'layer': layer_idx,
                        'head': head_idx,
                        'image_attention_diff': last_token_attn_diff['last_to_image_attention_diff'],
                        'correct_image_attention': last_token_attn_diff['correct_last_to_image'],
                        'incorrect_image_attention': last_token_attn_diff['incorrect_last_to_image']
                    }
                    last_token_attn_diff_sum['image_average'][layer_idx][head_idx].append(image_diff_info)
                
                                    
                                  
                correct_subject_attn = calculate_word_image_attention(correct_head_attn, correct_token_mapping, correct_svos[0], 
                                                                   correct_token_mapping['image_token_position'], correct_token_mapping['num_image_tokens'])
                correct_verb_attn = calculate_word_image_attention(correct_head_attn, correct_token_mapping, correct_svos[1], 
                                                                 correct_token_mapping['image_token_position'], correct_token_mapping['num_image_tokens'])
                correct_object_attn = calculate_word_image_attention(correct_head_attn, correct_token_mapping, correct_svos[2], 
                                                                   correct_token_mapping['image_token_position'], correct_token_mapping['num_image_tokens'])
                
                                  
                incorrect_subject_attn = calculate_word_image_attention(incorrect_head_attn, incorrect_token_mapping, incorrect_svos[0], 
                                                                     incorrect_token_mapping['image_token_position'], incorrect_token_mapping['num_image_tokens'])
                incorrect_verb_attn = calculate_word_image_attention(incorrect_head_attn, incorrect_token_mapping, incorrect_svos[1], 
                                                                   incorrect_token_mapping['image_token_position'], incorrect_token_mapping['num_image_tokens'])
                incorrect_object_attn = calculate_word_image_attention(incorrect_head_attn, incorrect_token_mapping, incorrect_svos[2], 
                                                                     incorrect_token_mapping['image_token_position'], incorrect_token_mapping['num_image_tokens'])
                
                                                
                if hasattr(correct_subject_attn, 'sum'):
                    svo_attn_sum['correct']['subject'][layer_idx, head_idx] += correct_subject_attn.sum().item()
                    svo_attn_sum['correct']['verb'][layer_idx, head_idx] += correct_verb_attn.sum().item()
                    svo_attn_sum['correct']['object'][layer_idx, head_idx] += correct_object_attn.sum().item()
                else:
                    svo_attn_sum['correct']['subject'][layer_idx, head_idx] += correct_subject_attn
                    svo_attn_sum['correct']['verb'][layer_idx, head_idx] += correct_verb_attn
                    svo_attn_sum['correct']['object'][layer_idx, head_idx] += correct_object_attn
                    
                if hasattr(incorrect_subject_attn, 'sum'):
                    svo_attn_sum['incorrect']['subject'][layer_idx, head_idx] += incorrect_subject_attn.sum().item()
                    svo_attn_sum['incorrect']['verb'][layer_idx, head_idx] += incorrect_verb_attn.sum().item()
                    svo_attn_sum['incorrect']['object'][layer_idx, head_idx] += incorrect_object_attn.sum().item()
                else:
                    svo_attn_sum['incorrect']['subject'][layer_idx, head_idx] += incorrect_subject_attn
                    svo_attn_sum['incorrect']['verb'][layer_idx, head_idx] += incorrect_verb_attn
                    svo_attn_sum['incorrect']['object'][layer_idx, head_idx] += incorrect_object_attn
    
           
    averaged_svo_attn_diff = []
    for layer_idx in range(num_layers):
        layer_diff = []
        for head_idx in range(num_heads):
                             
            avg_diff = {
                'subject': np.mean(svo_attn_diff_sum['subject'][layer_idx][head_idx]),
                'verb': np.mean(svo_attn_diff_sum['verb'][layer_idx][head_idx]),
                'object': np.mean(svo_attn_diff_sum['object'][layer_idx][head_idx])
            }
            layer_diff.append(avg_diff)

        
        averaged_svo_attn_diff.append(layer_diff)
    
                 
    if num_samples > 0:
        for sample_type in ['correct', 'incorrect']:
            for component in ['subject', 'verb', 'object']:
                svo_attn_sum[sample_type][component] /= num_samples
    
                           
    averaged_last_token_text_diffs = []
    averaged_last_token_image_diffs = []
    
    for layer_idx in range(num_layers):
        layer_text_diffs = []
        layer_image_diffs = []
        
        for head_idx in range(num_heads):
                            
            head_text_tokens = last_token_attn_diff_sum['text_tokens'][layer_idx][head_idx]
            if head_text_tokens:
                                      
                token_index_to_diffs = {}
                token_index_to_info = {}                
                for text_diff in head_text_tokens:
                    token_index = text_diff['token_index']
                    if token_index not in token_index_to_diffs:
                        token_index_to_diffs[token_index] = []
                        token_index_to_info[token_index] = {
                            'token': text_diff['token'],
                            'is_self': text_diff['is_self']
                        }
                    token_index_to_diffs[token_index].append(text_diff['attention_diff'])
                
                                
                avg_text_diffs = {}
                for token_index, diffs in token_index_to_diffs.items():
                    token_info = token_index_to_info[token_index]
                    token_name = token_info['token']
                    is_self = token_info['is_self']
                    
                                    
                    if is_self:
                        token_name = f"{token_name}(self)"
                    
                    avg_text_diffs[token_index] = {
                        'token_name': token_name,
                        'attention_diff': np.mean(diffs),
                        'is_self': is_self
                    }
                
                layer_text_diffs.append(avg_text_diffs)
            else:
                layer_text_diffs.append({})
            
                            
            head_image_diffs = last_token_attn_diff_sum['image_average'][layer_idx][head_idx]
            if head_image_diffs:
                avg_image_diff = np.mean([img_diff['image_attention_diff'] for img_diff in head_image_diffs])
                layer_image_diffs.append(avg_image_diff)
            else:
                layer_image_diffs.append(0.0)
        
        averaged_last_token_text_diffs.append(layer_text_diffs)
        averaged_last_token_image_diffs.append(layer_image_diffs)
    
    return {
        'svo_attn_differences': averaged_svo_attn_diff,
        'last_token_text_differences': averaged_last_token_text_diffs,                                           
        'last_token_image_differences': averaged_last_token_image_diffs,                                   
        'svo_attn_sums': svo_attn_sum,
        'num_samples': num_samples,
        'num_layers': num_layers,
        'num_heads': num_heads
    }

def process_single_question(question_data, img_path, model, tokenizer, img_proc, device, question_num, output_dir, save_outputs=SAVE_PAIR_OUTPUTS):
    """处理单个问题，返回结果（对齐13b版：先forward取attention，再生成答案）"""
    prompt_text = question_data.get('prompt') or question_data.get('question')
    expected_answer = question_data['label']

                          
    qs = prompt_text.strip()
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

          
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    image = Image.open(img_path)
    image_tensor = img_proc.preprocess(image, return_tensors='pt')['pixel_values'][0]
    if device == 'cuda':
        images = image_tensor.unsqueeze(0).half().to(device)
    else:
        images = image_tensor.unsqueeze(0).to(device)

                                    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            images=images,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = []
    if outputs.attentions is not None:
        for layer_attn in outputs.attentions:
            layer_attn_cpu = layer_attn.detach().to('cpu')
            if layer_attn_cpu.dim() == 4:
                layer_attn_cpu = layer_attn_cpu[0]                     
            attentions.append(layer_attn_cpu)
    else:
        print(f'[Warning] 未获取到注意力数据，question: {prompt_text}')

          
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=False,
            num_beams=1,
            max_new_tokens=20,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    outputs_text = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs_text = outputs_text.strip()
    if outputs_text.endswith(stop_str):
        outputs_text = outputs_text[:-len(stop_str)]
    model_answer = outputs_text.strip()

    if save_outputs:
        ensure_dir(output_dir)
        with open(os.path.join(output_dir, f'question_{question_num}_answer.txt'), 'w') as f:
            f.write(f"Question: {prompt_text}\n")
            f.write(f"Expected Answer: {expected_answer}\n")
            f.write(f"Model Answer: {model_answer}\n")

             
    del outputs
    del output_ids
    del images
    del input_ids
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'prompt': prompt_text,
        'attentions': attentions,
        'model_answer': model_answer,
        'expected_answer': expected_answer
    }


                                                    
def compute_averaged_frobenius_from_accumulator(frobenius_accumulator, num_layers, num_heads, num_samples):
    """从累积的Frobenius归一化值计算平均值（仅last token）"""
    averaged_last_token = []
    for layer_idx in range(num_layers):
        layer_diff = []
        for head_idx in range(num_heads):
            last_token_values = frobenius_accumulator['last_token'][layer_idx][head_idx]
            layer_diff.append({
                'last_token': np.mean(last_token_values) if last_token_values else 0.0
            })
        averaged_last_token.append(layer_diff)
    return {
        'svo_attn_differences': averaged_last_token,
        'num_samples': num_samples,
        'num_layers': num_layers,
        'num_heads': num_heads
    }


def analyze_averaged_last_token_attention_differences(averaged_attns, summary_output_dir):
    """分析平均后的 last token 注意力差异，仅写入热力图 PNG 与数据 JSON。"""
    svo_attn_diffs = averaged_attns['svo_attn_differences']
    num_layers = averaged_attns['num_layers']
    num_heads = averaged_attns['num_heads']
    num_samples = averaged_attns['num_samples']

    layer_analyses = {}
    for layer_idx in range(num_layers):
        last_token_frobenius_norms = []
        for head_idx in range(num_heads):
            head_svo_diff = svo_attn_diffs[layer_idx][head_idx]
            last_token_frobenius_norms.append(head_svo_diff['last_token'])
        layer_analyses[f'layer_{layer_idx}'] = {
            'layer': layer_idx,
            'num_samples': num_samples,
            'last_token_frobenius_norms': last_token_frobenius_norms,
        }

    create_last_token_heatmap(layer_analyses, summary_output_dir, num_layers, num_heads, num_samples)
    print(
        f"已保存: {os.path.join(summary_output_dir, 'last_token_attention_heatmap.png')} 与 "
        f"{os.path.join(summary_output_dir, 'last_token_heatmap_data.json')} "
        "及 sorted_* 文本（与 select_sort_last_token_heads 一致）。"
    )


def _format_sorted_head_line(layer_idx: int, head_indices, head_one_based: bool) -> str:
    if head_one_based:
        head_indices = [idx + 1 for idx in head_indices]
    heads_str = ",".join(str(x) for x in head_indices)
    return f"\"{layer_idx}:{heads_str};\""


def write_sorted_last_token_heads_files(
    last_token_data,
    output_dir: str,
    layer_one_based: bool = False,
    head_one_based: bool = False,
    select_min: bool = False,
):
    """
    根据 last_token_data（与 last_token_heatmap_data.json 中相同结构的 list[list[float]]）
    写出各层内按敏感度排序的 head 列表，以及按层平均排序的层列表。
    与 Rel_Hallucination/experiments/utils/select_sort_last_token_heads.py 行为一致。
    """
    ensure_dir(output_dir)
    if not isinstance(last_token_data, list) or len(last_token_data) == 0:
        print("警告: last_token_data 为空或无效，跳过 sorted_* 文本输出")
        return

    layer_averages = []
    for layer_idx, layer_vals in enumerate(last_token_data):
        if not isinstance(layer_vals, list) or len(layer_vals) == 0:
            print(f"警告: Layer {layer_idx} 的 head 列表无效，跳过层平均计算")
            continue
        try:
            avg_value = float(np.mean([float(v) for v in layer_vals]))
            layer_averages.append((avg_value, layer_idx))
        except (ValueError, TypeError) as e:
            print(f"警告: Layer {layer_idx} 数值无效，跳过层平均: {e}")
            continue

    if not layer_averages:
        print("警告: 无有效层平均，跳过 sorted_layers 输出")
    else:
        if select_min:
            layer_averages.sort(key=lambda x: x[0])
        else:
            layer_averages.sort(key=lambda x: -x[0])
        sorted_layer_indices = [layer_idx for _, layer_idx in layer_averages]
        formatted_layer_indices = []
        for layer_idx in sorted_layer_indices:
            out_layer_idx = layer_idx + 1 if layer_one_based else layer_idx
            formatted_layer_indices.append(str(out_layer_idx))
        layers_str = ",".join(formatted_layer_indices)
        layer_output_text = f'"{layers_str}"\n'
        layer_results = []
        for rank, (avg_val, layer_idx) in enumerate(layer_averages, 1):
            out_layer_idx = layer_idx + 1 if layer_one_based else layer_idx
            layer_results.append(f"Rank {rank}: Layer {out_layer_idx} (average: {avg_val:.6f})")
            print(f"Rank {rank}: Layer {layer_idx} (average: {avg_val:.6f})")
        layer_detailed_text = "\n".join(layer_results) + "\n"
        if select_min:
            layer_filename = "sorted_min_layers_by_average.txt"
            layer_detailed_filename = "sorted_min_layers_by_average_detailed.txt"
        else:
            layer_filename = "sorted_layers_by_average.txt"
            layer_detailed_filename = "sorted_layers_by_average_detailed.txt"
        layer_out_path = os.path.join(output_dir, layer_filename)
        with open(layer_out_path, "w") as f:
            f.write(layer_output_text)
        layer_detailed_path = os.path.join(output_dir, layer_detailed_filename)
        with open(layer_detailed_path, "w") as f:
            f.write(layer_detailed_text)
        print(f"已保存按层平均排序: {layer_out_path}")
        print(f"已保存按层平均排序(详细): {layer_detailed_path}")

    results = []
    for layer_idx, layer_vals in enumerate(last_token_data):
        if not isinstance(layer_vals, list) or len(layer_vals) == 0:
            print(f"警告: Layer {layer_idx} 的 head 列表无效，跳过该层 head 排序")
            continue
        layer_candidates = []
        for head_idx, v in enumerate(layer_vals):
            try:
                layer_candidates.append((float(v), head_idx))
            except (ValueError, TypeError):
                continue
        if len(layer_candidates) == 0:
            print(f"警告: Layer {layer_idx} 无有效 head，跳过")
            continue
        if select_min:
            layer_candidates.sort(key=lambda x: x[0])
        else:
            layer_candidates.sort(key=lambda x: -x[0])
        sorted_head_indices = [head_idx for _, head_idx in layer_candidates]
        out_layer_idx = layer_idx + 1 if layer_one_based else layer_idx
        line = _format_sorted_head_line(out_layer_idx, sorted_head_indices, head_one_based=head_one_based)
        results.append(line)
        print(f"Layer {layer_idx}: 已排序 {len(sorted_head_indices)} 个 head")

    if select_min:
        heads_filename = "sorted_min_last_token_heads.txt"
    else:
        heads_filename = "sorted_last_token_heads.txt"
    heads_out_path = os.path.join(output_dir, heads_filename)
    with open(heads_out_path, "w") as f:
        f.write("\n".join(results) + "\n")
    print(f"已保存各层 head 排序: {heads_out_path}（共 {len(results)} 层）")


def create_last_token_heatmap(layer_analyses, output_dir, num_layers, num_heads, num_samples):
    """创建跨层跨头的Last Token热力图"""
    last_token_data = np.zeros((num_layers, num_heads))
    for layer_idx in range(num_layers):
        layer_key = f'layer_{layer_idx}'
        if layer_key in layer_analyses:
            layer_analysis = layer_analyses[layer_key]
            for head_idx in range(num_heads):
                if head_idx < len(layer_analysis['last_token_frobenius_norms']):
                    last_token_data[layer_idx, head_idx] = layer_analysis['last_token_frobenius_norms'][head_idx]

    plt.figure(figsize=(20, 16))
    sns.heatmap(
        last_token_data,
        cmap='viridis',
        annot=True,
        fmt='.3f',
        annot_kws={'size': 6},
        cbar_kws={'label': 'Last Token Frobenius Norm'},
        xticklabels=range(num_heads),
        yticklabels=range(num_layers)
    )
    plt.title('Last Token Attention Differences Heatmap (All Layers and Heads)', fontsize=14, pad=20)
    plt.xlabel('Head Index', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'last_token_attention_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    heatmap_data = {
        'last_token_data': last_token_data.tolist(),
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_samples': num_samples
    }
    with open(os.path.join(output_dir, 'last_token_heatmap_data.json'), 'w') as f:
        json.dump(heatmap_data, f, indent=2)

    write_sorted_last_token_heads_files(
        heatmap_data['last_token_data'],
        output_dir,
        layer_one_based=SORT_LAST_TOKEN_LAYER_ONE_BASED,
        head_one_based=SORT_LAST_TOKEN_HEAD_ONE_BASED,
        select_min=SORT_LAST_TOKEN_SELECT_MIN,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_pair_outputs', action='store_true', help='保存每个question_id的成对输出（图像与文本文件）。不加此开关则不保存。')
    parser.add_argument('--num_samples', type=int, default=None, help='从DATA_FILE中随机抽取的样本数量。如果不设置此参数，则处理全部样本。')
    args, unknown = parser.parse_known_args()
    if args.save_pair_outputs:
        SAVE_PAIR_OUTPUTS = True
                       
    torch.manual_seed(55)
    torch.cuda.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    np.random.seed(55)
    random.seed(55)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, img_proc, _ = load_pretrained_model(MODEL_PATH, None, model_name)
    
             
    grouped_data = load_and_group_data(DATA_FILE)
    
                                   
    num_samples_to_use = args.num_samples if args.num_samples is not None else NUM_SAMPLES
    
                                    
    if num_samples_to_use is not None:
        total_available = len(grouped_data)
        if num_samples_to_use > total_available:
            print(f"警告: 请求的样本数量 ({num_samples_to_use}) 大于可用样本数量 ({total_available})，将使用全部 {total_available} 个样本")
            num_samples_to_use = total_available
        
                                 
        sample_keys = list(grouped_data.keys())
        random.shuffle(sample_keys)
        selected_keys = sample_keys[:num_samples_to_use]
        grouped_data = {key: grouped_data[key] for key in selected_keys}
        print(f"从 {total_available} 个样本中随机抽取了 {num_samples_to_use} 个样本进行处理")
    else:
                           
        num_samples_to_use = len(grouped_data)
    
                         
    OUTPUT_BASE_PATH = OUTPUT_BASE_PATH_TEMPLATE.format(num_samples=num_samples_to_use)
    print(f"输出路径: {OUTPUT_BASE_PATH}")

    total_pairs = len(grouped_data)
    valid_pairs = 0
    invalid_pairs = 0

    num_layers = getattr(model.config, 'num_hidden_layers', None)
    num_heads = getattr(model.config, 'num_attention_heads', None)
    if num_layers is None or num_heads is None:
        raise ValueError("无法从模型配置中读取层数或头数信息")
    print(f"模型结构: {num_layers} 层, {num_heads} 头")

    frobenius_accumulator = {
        'last_token': [[[] for _ in range(num_heads)] for _ in range(num_layers)]
    }

    failed_samples = []             
    for question_id, question_data in tqdm(grouped_data.items(), desc='Processing question pairs'):
        try:
            success = process_question_pair_and_compute_frobenius(
                question_id, question_data, tokenizer, model, img_proc, device,
                IMAGE_ROOT, OUTPUT_BASE_PATH,
                frobenius_accumulator,
                num_layers, num_heads, 'llava_v1',
                save_pair_outputs=SAVE_PAIR_OUTPUTS
            )
            if success:
                valid_pairs += 1
            else:
                invalid_pairs += 1
                failed_samples.append(question_id)
                print(f"[FAILED] 样本 {question_id} 处理失败（返回False）")
        except Exception as e:
            print(f"[EXCEPTION] 处理 {question_id} 时出错: {e}")
            invalid_pairs += 1
            failed_samples.append(question_id)
            import traceback
            traceback.print_exc()
            continue

    print(f"样本对处理完成: {valid_pairs}/{total_pairs} 有效")
    if failed_samples:
        print(f"\n失败的样本列表 ({len(failed_samples)} 个):")
        for sample_id in failed_samples:
            print(f"  - {sample_id}")
    else:
        print("\n所有样本处理成功！")

    if ANALYZE_HEAD_ATTENTION_DIFFERENCES and valid_pairs > 0:
        summary_output_dir = os.path.join(OUTPUT_BASE_PATH, 'statistical_summary')
        ensure_dir(summary_output_dir)

        averaged_attentions = compute_averaged_frobenius_from_accumulator(
            frobenius_accumulator,
            num_layers,
            num_heads,
            valid_pairs
        )

        analyze_averaged_last_token_attention_differences(
            averaged_attentions,
            summary_output_dir
        )

    print("分析完成！")
