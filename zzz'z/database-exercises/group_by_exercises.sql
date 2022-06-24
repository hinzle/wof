-- 1
use	employees;
select * from titles;

-- 2
-- 7
select distinct title from titles;

-- 3
SELECT last_name
FROM employees 
WHERE last_name LIKE 'E%E'
GROUP BY last_name;

-- 4
SELECT first_name, last_name
FROM employees 
WHERE last_name LIKE 'E%E'
GROUP BY first_name, last_name;

-- 5
select last_name
from employees
where last_name LIKE '%q%'
and last_name NOT LIKE '%qu%'
group by last_name;

-- 6
select last_name, count(*)
from employees
where last_name LIKE '%q%'
and last_name NOT LIKE '%qu%'
group by last_name;

-- 7
SELECT first_name, gender, count(*)
FROM employees
WHERE first_name IN ('Irena', 'Vidya', 'Maya')
group by first_name, gender;

-- 8
select 
lower(
	concat(
		substr(first_name,1,1), 
		substr(last_name,1,4),
		'_',
		substr(birth_date,6,2),
		substr(birth_date,3,2)
	)
)
AS username, count(*)
FROM employees
group by username
order by count(*) desc;

-- 9.a
select emp_no,
avg(salary)
from salaries
group by emp_no;

-- 9.b
select * from dept_emp;
	
select dept_no, count(*)
from dept_emp
where to_date > NOW()
group by dept_no;

-- 9.c
select * from salaries;

select emp_no, count(*)
from salaries
group by emp_no;

-- 9.d
select emp_no, MAX(salary)
from salaries
group by emp_no;

-- 9.e
select emp_no, MIN(salary)
from salaries
group by emp_no;

-- 9.f
select emp_no, STDDEV(salary)
from salaries
group by emp_no;

-- 9.g
select emp_no, max(salary) as mmm
from salaries
group by emp_no
having mmm>150000;

-- 9.h
select emp_no, max(salary) as mmm
from salaries
group by emp_no
having mmm between 80000 and 90000;
