-- 1
USE employees;
select * from employees;

-- 2
SELECT DISTINCT last_name FROM employees
LIMIT 10;

-- 3 
/*
Alselm	Cappello
Utz	Mandell
Bouchung	Schreiter
Baocai	Kushner
Petter	Stroustrup
*/

SELECT *
FROM employees
WHERE birth_date LIKE '%%%%-12-25'
AND hire_date LIKE '199%'
ORDER BY hire_date
LIMIT 5;

SELECT first_name, last_name
FROM employees
WHERE birth_date LIKE '%%%%-12-25'
AND hire_date LIKE '199%'
ORDER BY hire_date
LIMIT 5;

-- 4
-- OFFSET is the page number you want to view. Limit is the number of results you want to view per page.
SELECT *
FROM employees
WHERE birth_date LIKE '%%%%-12-25'
AND hire_date LIKE '199%'
ORDER BY hire_date
LIMIT 5 OFFSET 10;