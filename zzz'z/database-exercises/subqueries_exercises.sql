-- 1
SELECT concat(e.first_name,' ',e.last_name) namessa
	FROM employees e	
	WHERE e.hire_date LIKE 
 		(
 		SELECT hire_date
 		FROM employees
		WHERE emp_no = 101010
		)
 	AND e.emp_no IN
 		(
 		SELECT emp_no
 		FROM dept_emp
 		WHERE to_date>NOW()
 		)
;

-- 2
SELECT *
	FROM titles t	
	WHERE emp_no IN
		(
		SELECT emp_no
		FROM employees
		WHERE first_name='Aamod'
		)
		AND	 emp_no IN
		(
		SELECT emp_no
		FROM dept_emp
		WHERE to_date>NOW()
		)
;

-- 3
-- 85108
SELECT *
FROM employees 
WHERE emp_no IN
	(
	SELECT emp_no
	FROM dept_emp
	WHERE to_date <NOW()
	)	
;

-- 4
/* Isamu Legleitner
Karsten Sigstam
Leon DasSarma
Hilary Kambil */

SELECT concat(e.first_name,' ',e.last_name) namessa
FROM employees e
WHERE emp_no IN
	(
	SELECT emp_no
	FROM dept_manager
	WHERE to_date >NOW()
	)
	AND gender LIKE 'f'
	
;

-- 5
-- 154543
SELECT *
FROM salaries
WHERE salary>
	(
	SELECT AVG(salary)
	FROM salaries
	)
AND 
to_date >NOW()
;

6
-- The following code is copied from an accomplice.
-- Ryan Orsinger and I sat after class and hashed this problem out with our own code but due to a series of unfortunate sqlace saves, none of that code is any longer in existance to the great disdain of us all :(

SELECT COUNT(salary) 'Within a Standard Deviation of Highest Salary'
  FROM salaries 
 WHERE to_date LIKE '9%'
   AND salary >
		   (SELECT MAX(salary) FROM salaries WHERE to_date LIKE '9%')
		   - (SELECT STDDEV(salary) FROM salaries WHERE to_date LIKE '9%'); # Salaries within 1 standard deviation: 83
		   
SELECT(
	   (SELECT COUNT(*)
		FROM salaries
		WHERE to_date LIKE '9%'
     AND salary > (
	   (SELECT MAX(salary) FROM salaries WHERE to_date LIKE '9%')
		   - (SELECT STDDEV(salary) FROM salaries WHERE to_date LIKE '9%')
             )
             )
           /(SELECT COUNT(*)
           FROM salaries
           WHERE to_date > now())) * 100 AS 'percentage within a standard deviation of highest salary'
;