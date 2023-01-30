'''
	Reduce Tag
	ignore or replace tags

	FN_TOTAL format:
		Abstract
		Seperator
		Abstract
		Seperator
		...
	
	Abstract format:
		Sentence'\t'tag'\n'
		Sentence'\t'tag'\n'
		...

'''

FN_TOTAL = '/workspace/paperassistant/backend/block_classifier/data/total.txt'
FN_TOTAL_TAG_REDUCED_IGNORE = '/workspace/paperassistant/backend/block_classifier/data/total_tag_reduced_ignore.txt'
FN_TOTAL_TAG_REDUCED_REPLACE = '/workspace/paperassistant/backend/block_classifier/data/total_tag_reduced_replace.txt'
FN_TOTAL_TAG_REDUCED_IGNORE56 = '/workspace/paperassistant/backend/block_classifier/data/total_tag_reduced_ignore56.txt'
FN_TOTAL_TAG_REDUCED_REPLACE56 = '/workspace/paperassistant/backend/block_classifier/data/total_tag_reduced_replace56.txt'

modes = [ 'original', 'ignore', 'replace', 'ignore-56', 'replace-56', 'ignore-5' ]

def reduce_tag(tag: str, mode: str = 'ignore'):
	'''
		v1 (mode == 'ignore')
		ignore +tag

		v2 (mode == 'replace')
		change an A+B tag to the B

		v3 (mode == 'ignore-56')
		ignore +tag
		ignore 5, 6

		v4 (mode == 'replace-56')
		change an A+B tag to the B
		ignore 5, 6
	'''
	if mode == 'ignore':
		if len(tag) > 1:
			tag = '+'
	elif mode == 'replace':
		if tag == '1+2':
			tag = '2'
		elif tag == '2+3':
			tag = '3'
		elif tag == '3+4':
			tag = '4'
		if tag == '4+5':
			tag = '5'
	elif mode == 'ignore-56':
		if len(tag) > 1:
			tag = '+'
		elif tag in ('5', '6'):
			tag = '-'
	elif mode == 'replace-56':
		if tag == '1+2':
			tag = '2'
		elif tag == '2+3':
			tag = '3'
		elif tag == '3+4':
			tag = '4'
		elif tag in ('4+5', '5', '6'):
			tag = '-'
	elif mode == 'ignore-5':
		if tag in ('4+5', '5'):
			tag = '4'
		elif len(tag) > 1:
			tag = '+'
	return tag



def count_tag(fn, mode):
	with open(fn, 'r', encoding='utf-8') as rf:
		lines = rf.readlines()
	
	tag_cnt_dict = dict()
	tag_reduced_cnt_dict = dict()
	for idx, line in enumerate(lines):
		if line.startswith('---'):
			continue

		sentence, tag = line.rstrip().split('\t')
		
		if tag_cnt_dict.get(tag) == None:
			tag_cnt_dict[tag] = 0
		tag_cnt_dict[tag] += 1

		tag = reduce_tag(tag, mode)
		if tag_reduced_cnt_dict.get(tag) == None:
			tag_reduced_cnt_dict[tag] = 0
		tag_reduced_cnt_dict[tag] += 1
		
		line = sentence + '\t' + tag + '\n'
		lines[idx] = line


	return (lines, tag_cnt_dict, tag_reduced_cnt_dict)

def remove_useless_tags(fn, remove_unit='line'):
	# after calling 'count_tag'
	with open(fn, 'r', encoding='utf-8') as rf:
		lines = rf.readlines()
	
	lines_removed_useless_tags = list()
	abstract = list()
	next_abstract = False
	for line in lines:
		if line.startswith('---'):
			abstract.append(line)
			# extend abstract
			lines_removed_useless_tags.extend(abstract)
			abstract.clear()
			next_abstract = False
			continue
		
		if next_abstract:
			continue

		sentence, tag = line.rstrip().split('\t')
		if tag in ('+', '-'):
			if remove_unit == 'line':
				continue
			elif remove_unit == 'abstract':
				abstract.clear()
				next_abstract = True
			else:
				raise ValueError
		else:
			abstract.append(line)

	with open(fn, 'w', encoding='utf-8') as wf:
		wf.writelines(lines_removed_useless_tags)

if __name__ == '__main__':
	lines = count_tag(FN_TOTAL, mode='ignore')
	with open(FN_TOTAL_TAG_REDUCED_IGNORE, 'w', encoding='utf-8') as wf:
		wf.writelines(lines)

	lines = count_tag(FN_TOTAL, mode='replace')
	with open(FN_TOTAL_TAG_REDUCED_REPLACE, 'w', encoding='utf-8') as wf:
		wf.writelines(lines)

