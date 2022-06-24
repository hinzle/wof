# 1.a
x=input('Enter day of the week: ')
if x in ['Monday', 'monday', 'mon']:
	print(f'Today is {x}.')
else:
	print('Today is not Monday.')

# 1.b
x=input('Enter day of the week: ')
if x in ['Saturday', 'Sunday' ]:
	print(f'Today is a weekend.')
else:
	print('Today is a weekday.')

# 1.b.alt
x=input('Enter day of the week: ')
if day_of_week.lower().startswith('s'):
    print('weekend')
else:
    print('you better work ')

# 1.c
hours_worked=50
hourly_rate=100
if hours_worked>40:
	weeks_pay=((hours_worked-40)*1.5*hourly_rate)+(40*hourly_rate)
else:
	weeks_pay=hours_worked*hourly_rate	
print(f'Weeks pay is ${weeks_pay}.')

# 2.a

# 5 to 15
i=5
while i<=15:
	print(i)
	i+=1

# 0 to 100, by 2's
i=0
while i<=100:
	print(i)
	i+=2
	
# 100 to -10, by 5's
i=100
while i>=-10:
	print(i)
	i-=5

# fibonacci seq but with *
i=2
while i<1_000_000:
	print(i)
	i=i**2

# 100 to 5, by 5's
i=100
while i>=5:
	print(i)
	i-=5

# 2.b.i
x=input('Enter a number:')
x=float(x)
ints=list(range(1,11))
for n in ints:
	ans=x*n
	print(f'{x} x {n} = {ans}')
	
# 2.b.ii
ints=list(range(1,10))
for n in ints:
	ans=n*str(n)
	print(ans)

# 2.c.i

while True:
	x=input('Enter an odd number | 0<n<51: ')
	print (f'Number to skip is: {x}\n')
	if x.isdigit() and int(x)%2==1 and int(x)>0 and int(x)<50:
		for num in range(1, 50, 2):
			if num == int(x):
				print(f'Yikes! skipping number: {num}')	
			else:
				print(f'Here is an odd number: {num}')
		break
					
	else:
		print ('Wrong.')
		continue

# 2.d
n=input('Enter a postive number: ')
n=int(n)
if n>0:
	for i in range(0,(n+1)):
		print(i)

#2.e
n=input('Enter a postive number: ')
n=int(n)
if n>0:
	for i in range(0,(n+1)):
		print(n-i)



# 3
n=100
for i in range(1,(n+1)):
	if i%3==0 and i%5==0:
		print('FizzBuzz')
	elif i%3==0:
		print('Fizz')
	elif i%5==0:
		print('Buzz')
	else:
		print(i)

# 4
# Copied for reference
while True:
    posited_num = input('Please insert a positive integer: ')
    if posited_num.isdigit():
        if int(posited_num) > 0:
            break
proceed = input('Do you want to continue and print a table of powers, y/n? :')
if proceed.lower().startswith('y'):
    posited_num = int(posited_num)
    print()
    print('number | squared | cubed')
    print('------ | ------- | -----')
    for i in range(1, posited_num + 1):
        i_squared = i ** 2
        i_cubed = i ** 3
        print(f'{i: <6} | {i_squared: ^7} | {i_cubed: 5}')

# 5
while True:
    user_number = input("Please enter a numeral between 0 and 100: ")
    if user_number.isdigit():
        user_number = int(user_number)
        if user_number < 0 or user_number > 100:
            print('Wrong.')
            continue
        break
grade = int(user_number)
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
print(grade)

# 6
bookshelf = [
    {'title': 'Annihilation',
    'author': 'Jeff Vandermeer',
    'genre': 'Science Fiction'},
    {'title': 'Octopus Pie',
    'author': 'Maredeth Gran',
    'genre': 'Comic'},
    {'title': 'Cabin At the End of the World',
    'author': 'Paul Tremblay',
    'genre': 'Horror'},
    {'title': 'Severance',
    'author': 'Ling Ma',
    'genre': 'Science Fiction'},
]

for book in bookshelf:
    print('we are living in a single dictionary here')
    [print(key, ': ', book[key]) for key in book]
    print('------')

# 6.a

picked_genre = input('Please pick a genre and I will return the titles of that genre on shelf. \n')

matches = []
for book in bookshelf:
    if book['genre'].lower() == picked_genre.lower():
        matches.append(book['title'])
if matches == []:
    print('no books in that genre available. please check back later')
else:
    print(f'I have the following titles in the genre {picked_genre}')
    [print(match) for match in matches]
