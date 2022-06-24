# 1

def is_two(x):
	"It should accept one input and return True if the passed input is either the number or the string 2, False otherwise."
	if type(x)==int and x == 2:
		return True
	elif type(x)==str and x.lower() == 'two':
		return True
	else:
		return False

# is_two(0)
# is_two(2)
# is_two('zero')
# is_two('Two')
# is_two('two')

# 2

def is_vowel(x):
	"should return True if the passed string is a vowel, False otherwise"
	if type(x)==str and x.lower() in ['a','e','i','o','u']:
		return True
	else:
		return False

# is_vowel('a')
# is_vowel('b')
# is_vowel('D')
# is_vowel('E')

# 3

def is_consonant(x):
	"should return True if the passed string is a consonant, False otherwise"
	if is_vowel(x):
		return False
	else:
		return True

# is_consonant('a')
# is_consonant('b')
# is_consonant('D')
# is_consonant('E')

# 4

def capitalizer(x):
	x=list(x)
	if x[0].islower and is_consonant(x[0]):
		x[0]=x[0].upper()
		x=''.join(x)
		print(x)
	else:
		print('Wrong.')

# capitalizer('cory')
# capitalizer('Cory')
# capitalizer('alfred')
# capitalizer('Alfred')

# 5
def calculate_tip(price,tip):
	if tip<=1 and tip>=0:
		return price*tip
	else:
		print('Wrong.')

# calculate_tip(18.20,.5)
# calculate_tip(1,1)
# calculate_tip(1,0)
# calculate_tip(0,0)
# calculate_tip(0,1)
# calculate_tip(1,2)


# 6
def apply_discount(price,discount):
	if discount<=1 and discount>=0:
		return price-(price*discount)
	else:
		print('Wrong.')	


# apply_discount(18.2,.5)
# apply_discount(1,0)
# apply_discount(0,1)
# apply_discount(1,1)
# apply_discount(0,0)
# apply_discount(1,2)
# 7
def handle_commas(x):
	liss=[]
	for i in x:
		if i == ',':
			continue
		else:
			liss.append(i)

	handled=''.join(liss)
	return handled


# handle_commas('3,4,5')	

# 8
def get_letter_grade(grade):
	if grade in range(60):
		grade = 'F'
	elif grade in range(60,67):
		grade = 'D'
	elif grade in range(67,80):
		grade = 'C'
	elif grade in range(80,88):
		grade = 'B'
	else:
		grade = 'A'
	return grade

# get_letter_grade(0)
# get_letter_grade(50)
# get_letter_grade(60)
# get_letter_grade(70)
# get_letter_grade(80)
# get_letter_grade(90)
# get_letter_grade(100)

# 9
def remove_vowels(x):
	liss=[]
	for i in x:
		if is_vowel(i):
			continue
		else:
			liss.append(i)
	removed=''.join(liss)
	return removed		



# remove_vowels('Frank')

# remove_vowels('Cory')

# remove_vowels('abcdefghijklmnopqrstuvwxyz')

#10
def normalize_name(x):
	import string
	x=x.lower()
	lowercase_alphabets=list(string.ascii_lowercase)
	liss=[]
	for i in x:
		if i=='_':
			liss.append(i)
		if i==' ':
			liss.append('_')
		elif i in lowercase_alphabets:
			liss.append(i)
		else:
			continue
	normalized = ''.join(liss)
	return normalized

#x=x.replace #['!','@','#','$','%','^','&','*','(',')','+','-','=',',' ',]	

# normalize_name('Frank')
# normalize_name('Cory')
# normalize_name('abcde fghijklmn OPQRSTUVW XYZ!@#$%^ &*()_+1234567890-=')

# 11 
def cumulative_sum(x):
	adder_list=list(range(0,len(x)))
	liss=[]
	for i in adder_list:
		if i == 0:
			liss.append(x[i])
		else:
			liss.append(x[i]+liss[i-1])
			
	return(liss)	

# cumulative_sum([1,1,1])
# cumulative_sum([1, 3, 6, 10])
# cumulative_sum([1, 2, 3, 4])


