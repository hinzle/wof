-- 1
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

-- 2
-- Irena Reutenauer
SELECT *
FROM employees
WHERE first_name IN ('Irena', 'Vidya', 'Maya')
ORDER BY first_name;

-- Vidya Awdeh
 SELECT *
FROM employees
WHERE first_name IN ('Irena', 'Vidya', 'Maya')
ORDER BY first_name DESC;

-- 3
-- Irena Acton
SELECT *
FROM employees
WHERE first_name IN ('Irena', 'Vidya', 'Maya')
ORDER BY first_name, last_name;

-- Vidya Zweizig
SELECT *
FROM employees
WHERE first_name IN ('Irena', 'Vidya', 'Maya')
ORDER BY first_name DESC, last_name DESC;

-- 4 
-- Irena Acton
SELECT *
FROM employees
WHERE first_name IN ('Irena', 'Vidya', 'Maya')
ORDER BY last_name, first_name;

-- Maya Zyda
SELECT *
FROM employees
WHERE first_name IN ('Irena', 'Vidya', 'Maya')
ORDER BY last_name DESC, first_name DESC;

-- 5
-- 899, 10021, Ramzi Erde
SELECT *
FROM employees
WHERE last_name LIKE 'E%'
AND last_name LIKE '%e'
ORDER BY emp_no;

-- 499648, Tadahiro Erde
SELECT *
FROM employees
WHERE last_name LIKE 'E%'
AND last_name LIKE '%e'
ORDER BY emp_no DESC;

-- 6
-- 899, Teiji Eldridge, Sergi Erde
SELECT *
FROM employees
WHERE last_name LIKE 'E%'
AND last_name LIKE '%e'
ORDER BY hire_date DESC;

-- Sergi Erde
SELECT *
FROM employees
WHERE last_name LIKE 'E%'
AND last_name LIKE '%e'
ORDER BY hire_date;

-- 7
-- 362, Tremaine Eugenio, Douadi Pettis
SELECT *
FROM employees
WHERE birth_date LIKE '%%%%-12-25'
AND hire_date LIKE '199%'
ORDER BY birth_date, hire_date; 

-- Douadi Pettis
SELECT *
FROM employees
WHERE birth_date LIKE '%%%%-12-25'
AND hire_date LIKE '199%'
ORDER BY birth_date DESC, hire_date; 





