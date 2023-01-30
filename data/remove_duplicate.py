def get_abstracts_without_duplicate(lines, sep):
    # remove duplicate abstracts
    ## 1. Get an abstract
    ## 2. If the first line of the abstract is not in the new list, 
    ##      then put the abstract in the list, else do nothing.
    only_first_lines = set()
    new_abstracts = list()
    end = len(lines)
    start_abs_idx = 0
    end_abs_idx = 0
    cnt_dup = 0
    while end_abs_idx + 1 != end:
        end_abs_idx = lines.index(sep, start_abs_idx)

        # store abstract with a seperator
        abstract = lines[start_abs_idx:end_abs_idx + 1]

        # You have to think a case: same text, different label
        first_line = abstract[0]

        if first_line in only_first_lines:
            print(f"dup #{cnt_dup}: {first_line}")
        else:
            only_first_lines.add(first_line)
            new_abstracts.append(abstract)
        start_abs_idx = end_abs_idx + 1
    
    return new_abstracts, cnt_dup

def abstracts_to_lines(abstracts, with_sep: bool = True):
    lines = list()
    for abst in abstracts:
        if not with_sep:
            abst = abst[:-1]
        lines += abst
    return lines

def remove_duplicate(fn, fn_undup, sep, overwrite: bool = True):
    with open(fn, mode='r', encoding='utf-8') as rf:
        lines = rf.readlines()
        print('len(lines):', len(lines))
	 
    abstracts, cnt_dup = get_abstracts_without_duplicate(lines, sep)

    # shuffle abstracts and unpack them into a new list for each line.
    #random.shuffle(new_abstracts)
    lines_nodup = abstracts_to_lines(abstracts)
    print('len(lines_nodup):', len(lines_nodup))

    with open(fn_undup, mode='w', encoding='utf-8') as wf:
        wf.writelines(lines_nodup)
    
    ret = (lines_nodup, cnt_dup)
    if cnt_dup == 0:
        print("no duplicate")
    
    if not overwrite:
        return ret
    
    fn_dup = ''.join(fn.split('.')[:-1]) + "_dup.txt"
    if cnt_dup != 0:
        rename(fn, fn_dup)
    rename(fn_undup, fn)
    
    import os
    os.remove(fn_undup)
    
    return ret

def rename(fn_origin, fn_new):
    with open(fn_origin, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
    
    with open(fn_new, 'w', encoding="utf-8") as wf:
        wf.writelines(lines)

if __name__ == '__main__':
    #from reform_total import reform_data
    #import random
    #random.seed(42)

    FN        = 'retotal.txt'
    FN_UNDUP  = 'retotal_undup.txt'
    SEP = '----------\n'

    import os
    dirname = os.path.dirname(__file__)
    FN = os.path.join(dirname, FN)
    FN_UNDUP = os.path.join(dirname, FN_UNDUP)

    lines_without_duplicate, cnt_dup = remove_duplicate(FN, FN_UNDUP, SEP, overwrite=True)
    