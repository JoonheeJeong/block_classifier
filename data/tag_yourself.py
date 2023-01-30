from data_split import get_abstract_list
import random

fn = '/workspace/paperassistant/backend/block_classifier/data/total_acl_2021.txt'
#fn_reformed = '/workspace/paperassistant/backend/block_classifier/data/total_reformed_nosep.txt'

def get_next_tag_dict(fn):
    with open(fn, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
    
    next_tag_dict = dict()
    for line in lines:
        sent, tag = line.rstrip().split('\t')
        next_tag_dict[sent] = tag
    
    return next_tag_dict

def end_block_prediction_game(cnts):
    cnt_total_input, cnt_abs_input, cnt_true_curr, cnt_true_next = cnts

    if cnt_total_input == 0:
        print('No input')
        return

    acc_curr = 100 * cnt_true_curr / cnt_total_input
    acc_next = 100 * cnt_true_next / cnt_total_input

    print('\n========== Prediction Score ==========')

    print(f'Current: {cnt_true_curr}/{cnt_total_input} - {acc_curr:.3f}%')
    print(f'Next   : {cnt_true_next}/{cnt_total_input} - {acc_next:.3f}%')

def block_prediction_game(fn):
    with open(fn, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
    abs_list = get_abstract_list(lines)
    random.shuffle(abs_list)

    cnt_total_input = 0
    cnt_abs_input = 0
    cnt_true_curr = 0
    cnt_true_next = 0

    for abs in abs_list:
        ans_curr_tag_list = list()
        ans_next_tag_list = list()
        ref_curr_tag_list = list()
        ref_next_tag_list = list()

        for idx, line in enumerate(abs):
            if idx == 0:
                cnt_abs_input += 1

            if line.startswith('---'):
                continue

            line = line.rstrip()

            print('=======### Guess tags of this sentence and the next sentence. (ex. 2 3) ###=======')
            print(line.split('\t')[0])
            ans = input('Answer? ')
            while len(ans.split()) != 2:
                print('You answered:', ans, 'but correct input format is like 3 4')
                ans = input('Answer? ')
            ans_curr_tag, ans_next_tag = ans.split()

            if ans_next_tag =='q' or ans_curr_tag == 'q':
                cnts = (cnt_total_input, cnt_abs_input, cnt_true_curr, cnt_true_next)
                end_block_prediction_game(cnts)
                ans_curr_tags_str = ' '.join(ans_curr_tag_list)
                ref_curr_tags_str = ' '.join(ref_curr_tag_list)
                ans_next_tags_str = ' '.join(ans_next_tag_list)
                ref_next_tags_str = ' '.join(ref_next_tag_list)
                print(f'current tags your answer: {ans_curr_tags_str}')
                print(f'current tags real label : {ref_curr_tags_str}')
                print(f'next tags your answer: {ans_next_tags_str}')
                print(f'next tags real label : {ref_next_tags_str}')
                return  

            cnt_total_input += 1

            ans_curr_tag_list.append(ans_curr_tag)
            ans_next_tag_list.append(ans_next_tag)
            
            ref_curr_tag = line.split('\t')[1].rstrip()
            ref_curr_tag_list.append(ref_curr_tag)

            next_line = abs[idx+1]
            ref_next_tag = 'e' if next_line.startswith('---') else next_line.split('\t')[1].rstrip()
            ref_next_tag_list.append(ref_next_tag)

            if ans_curr_tag == ref_curr_tag:
                cnt_true_curr += 1
            if ans_next_tag == ref_next_tag:
                cnt_true_next += 1

        ans_curr_tags_str = ' '.join(ans_curr_tag_list)
        ref_curr_tags_str = ' '.join(ref_curr_tag_list)
        ans_next_tags_str = ' '.join(ans_next_tag_list)
        ref_next_tags_str = ' '.join(ref_next_tag_list)
        print(f'current tags your answer: {ans_curr_tags_str}')
        print(f'current tags real label : {ref_curr_tags_str}')
        print(f'next tags your answer: {ans_next_tags_str}')
        print(f'next tags real label : {ref_next_tags_str}')

    cnts = (cnt_total_input, cnt_true_curr, cnt_true_next)
    end_block_prediction_game(cnts)

if __name__ == '__main__':
    # random.seed(42)
    block_prediction_game(fn)