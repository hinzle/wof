-- 1

SELECT

CONCAT(e.first_name,' ', e.last_name) 'Employee',
de.dept_no 'Department',
de.from_date 'Start Date',
de.to_date 'End Date',
de.to_date>NOW() AS is_current_employee

FROM employees e
JOIN dept_emp de
USING (emp_no)
;

-- Zachs more complicated version
-- SELECT
--     e.emp_no,
--     de.dept_no,
--     e.start_date,
--     IF(e.end_date > NOW(), NULL, e.end_date),
--     e.is_current_employee
-- FROM dept_emp de
-- JOIN (
--     SELECT
--         e.emp_no AS emp_no,
--         e.hire_date AS start_date,
--         MAX(de.to_date) AS end_date,
--         MAX(de.to_date) > NOW() AS is_current_employee
--     FROM employees e
--     JOIN dept_emp de USING (emp_no)
--     GROUP BY e.emp_no
-- ) e ON de.emp_no = e.emp_no AND de.to_date = e.end_date;
-- 2
SELECT
CONCAT(e.first_name,' ', e.last_name) 'Employee',
	CASE 
		WHEN e.last_name BETWEEN 'A%' AND 'I%' THEN 'A-H'
		WHEN LEFT (e.last_name, 1) <= 'I' THEN 'I-Q'
		WHEN SUBSTR(e.last_name,1,1) <= 'Z' THEN 'R-Z'
	-- this could be broken if null somewhere ELSE 'R-Z'
	END 'alpha_group'
FROM employees e
;

-- 3
SELECT
	COUNT(CASE WHEN e.birth_date LIKE '194%' THEN '40s' ELSE NULL END) AS '40s',
	COUNT(CASE WHEN e.birth_date like '195%' THEN '50s' ELSE NULL END) AS '50s',
	COUNT(CASE WHEN e.birth_date like '196%' THEN '60s' ELSE NULL END) AS '60s',
	COUNT(CASE WHEN e.birth_date like '197%' THEN '70s' ELSE NULL END) AS '70s',
	COUNT(CASE WHEN e.birth_date like '198%' THEN '80s' ELSE NULL END) AS '80s',
	COUNT(CASE WHEN e.birth_date like '199%' THEN '90s' ELSE NULL END) AS '90s',
	COUNT(CASE WHEN e.birth_date like '200%' THEN '00s' ELSE NULL END) AS '00s',
	COUNT(CASE WHEN e.birth_date like '201%' THEN '10s' ELSE NULL END) AS '10s'	
FROM employees e
;
SELECT
	concat(substr(birth_date, 1,3), '0') AS decade,
	count(*)
FROM employees
GROUP BY decade;

-- 4


-- garbage
/* 
SELECT *
FROM departments d
	JOIN dept_emp de
	USING (dept_no)
/* 	JOIN salaries s
	USING (emp_no) *\/
WHERE d.dept_name IN ('R&D', 'Sales & Marketing', 'Prod & QM', 'Finance & HR', 'Customer Service') 
	AND s.to_date >now() 
	AND de.to_date >now()

LIMIT 10
;

;
SELECT AVG (salary )
FROM salaries s
WHERE s.to_date >now() 
;

SELECT *
	CASE WHEN dept_name IN ('R&D', 'Sales & Marketing', 'Prod & QM', 'Finance & HR', 'Customer Service' THEN )
FROM departments d
	JOIN dept_emp de
	USING (dept_no)
 	JOIN salaries s
	USING (emp_no) *\/
WHERE s.to_date >now() 
	AND de.to_date >now()

LIMIT 10
;



SELECT
	dept_name,
	CASE WHEN dept_no IN ('R&D') THEN 
FROM departments

LIMIT 10
;

*/

SELECT 
   CASE
       WHEN dept_name IN ('research', 'development') THEN 'R&D'
       WHEN dept_name IN ('sales', 'marketing') THEN 'Sales & Marketing'
       WHEN dept_name IN ('Production', 'Quality Management') THEN 'Prod & QM'
       ELSE dept_name
   END AS dept_group,
	format(AVG(s.salary), 3) AS avg_salary
FROM departments d
	JOIN dept_emp de USING (dept_no)
 	JOIN salaries s USING (emp_no) 
WHERE s.to_date >now() AND de.to_date >now()
GROUP BY dept_group
;





SELECT Count(1, 0)
;




