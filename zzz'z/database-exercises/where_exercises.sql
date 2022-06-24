-- 1 
use employees;
select * from employees;

-- 2 709 rows
SELECT *
FROM employees
WHERE first_name IN ('Irena', 'Vidya', 'Maya')
ORDER BY first_name;

-- 3 709 rows
SELECT *
FROM employees
WHERE first_name='Irena' OR first_name='Vidya' OR first_name='Maya';

-- 4 
-- 441 records returned
SELECT *
FROM employees
WHERE (first_name='Irena' OR first_name='Vidya' OR first_name='Maya') AND gender='M';

-- 5
-- 7330
SELECT *
FROM employees
WHERE last_name LIKE 'E%'
ORDER BY last_name;

-- 6
-- 30723
SELECT *
FROM employees
WHERE last_name LIKE 'E%' OR last_name LIKE '%e'
ORDER BY last_name;

-- 23393
SELECT *
FROM employees
WHERE last_name LIKE '%e' and last_name NOT LIKE 'E%'
ORDER BY last_name;

-- 7
-- 899
SELECT *
FROM employees
WHERE last_name LIKE 'E%' AND last_name LIKE '%e'
ORDER BY last_name;

-- 24292
SELECT *
FROM employees
WHERE last_name LIKE '%e'
ORDER BY last_name;

-- 8
-- 135214
SELECT *
FROM employees
WHERE hire_date LIKE '199%'
ORDER BY hire_date;

-- 9 
-- 346
SELECT *
FROM employees
WHERE birth_date LIKE '%%%%-12-25'
ORDER BY birth_date;

-- 10
-- 362
SELECT *
FROM employees
WHERE birth_date LIKE '%%%%-12-25'
AND hire_date LIKE '199%'
ORDER BY hire_date; 


-- 11
-- 1873
SELECT *
FROM employees
WHERE last_name LIKE '%q%'
ORDER BY last_name;

-- 12
-- 547
SELECT *
FROM employees
WHERE last_name LIKE '%q%'
AND last_name NOT LIKE '%qu%'
ORDER BY last_name;