-- SELECT 
-- 	emp_no,
-- 	sum(salary),
-- 	sum(DATEDIFF(SELECT MAX(hire_date) FROM employees), e.hire_date),
-- FROM employees e
-- LEFT JOIN salaries s
-- GROUP BY emp_no

	



SELECT *
FROM dept_emp de
	LEFT JOIN employees e USING (emp_no)
	LEFT JOIN departments d using (dept_no)
WHERE de.to_date > NOW()
	AND (d.dept_name = 'Sales'
		OR d.dept_name = 'Marketing')

;


-- SELECT emp_no,
-- 	SUM(salary),
-- 	DATEDIFF()
-- FROM salaries s
-- GROUP BY emp_no
-- ;

-- SELECT DATEDIFF(s.to_date,s.from_date)
-- from salaries s;


SELECT emp_no,
	SUM(salary) AS salary,
	DATEDIFF(CURDATE(),MIN(to_date)) AS tenure
FROM salaries
GROUP BY emp_no
;