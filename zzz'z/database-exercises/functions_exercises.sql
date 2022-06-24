-- 1
USE employees;
select * from employees;

-- 2
SELECT 
	CONCAT(first_name,' ', last_name) AS full_name
FROM employees
WHERE last_name LIKE 'E%E';

-- 3
SELECT 
UPPER(
	CONCAT(first_name,' ', last_name) 
	)
	AS full_name
FROM employees
WHERE last_name LIKE 'E%E';

-- 4
SELECT *,
DATEDIFF(
	CURDATE(), hire_date
	)
	AS emp_length
FROM employees
WHERE birth_date LIKE '%%%%-12-25'
AND hire_date LIKE '199%'
ORDER BY hire_date; 

-- 5
SELECT MIN(salary), MAX(salary)
FROM salaries;

-- 6 SUBSTR(string, start_index, length)
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
AS username, first_name, last_name
FROM employees;