def cleaner(string_of_nums):
	cleanered = []
	for i in string_of_nums:
		if i.isdigit() or i =='.':
			cleanered.append(i)
		else: continue
	clean_num = ''.join(cleanered)	
	return clean_num


string_of_nums="$2,097.02"
x=cleaner(string_of_nums)
print(type(x))







