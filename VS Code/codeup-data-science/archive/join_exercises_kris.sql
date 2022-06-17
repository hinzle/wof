-- Join Example Database

-- Question 1:

USE join_example_db;
SELECT * FROM users;
SELECT * FROM roles;

-- Question 2: 

SELECT * FROM users
JOIN roles 
ON users.role_id = roles.id;

SELECT * FROM users
LEFT JOIN roles
ON users.role_id = roles.id;

SELECT * FROM users
RIGHT JOIN roles
ON users.role_id = roles.id;

SELECT * FROM roles
RIGHT JOIN users
ON roles.id = users.role_id;

-- Question 3:

SELECT roles.name, COUNT(users.role_id)
FROM roles
LEFT JOIN users ON users.role_id = roles.id
GROUP BY roles.name;

-- Employees Database

-- Question 1: 

USE employees;

-- Question 2: 

SELECT d.dept_name as 'Department Name', CONCAT(e.first_name, ' ', e.last_name) AS 'Department Manager' 
FROM employees as e 
JOIN dept_manager as dm
ON e.emp_no = dm.emp_no
JOIN departments as d
ON d.dept_no = dm.dept_no
WHERE dm.to_date = '9999-01-01';

-- Question 3: 

SELECT d.dept_name as 'Department Name', CONCAT(e.first_name, ' ', e.last_name) AS 'Department Manager' 
FROM employees as e 
JOIN dept_manager as dm
ON e.emp_no = dm.emp_no
JOIN departments as d
ON d.dept_no = dm.dept_no
WHERE dm.to_date = '9999-01-01' AND e.gender = 'f';

-- Question 4: 

SELECT t.title as 'Title', COUNT(t.emp_no) as 'Count'
FROM titles as t
JOIN dept_emp as de
USING (emp_no)
JOIN departments as d
USING (dept_no)
WHERE t.to_date = '9999-01-01' AND d.dept_name = 'Customer Service' AND de.to_date = '9999-01-01'
GROUP BY t.title
ORDER BY t.title;

-- Question 5: 

SELECT d.dept_name as 'Department Name', CONCAT(e.first_name, ' ', e.last_name) AS 'Name', s.salary as 'Salary' 
FROM employees as e 
JOIN dept_manager as dm
ON e.emp_no = dm.emp_no
JOIN departments as d
ON d.dept_no = dm.dept_no
JOIN salaries as s
ON s.emp_no = e.emp_no
WHERE dm.to_date = '9999-01-01' AND s.to_date = '9999-01-01'
ORDER BY d.dept_name;

-- Question 6: 

SELECT de.dept_no, d.dept_name, COUNT(de.emp_no)
FROM dept_emp as de
JOIN departments as d 
USING (dept_no)
WHERE de.to_date = '9999-01-01'
GROUP BY de.dept_no
ORDER BY de.dept_no;

-- Question 7: 

SELECT d.dept_name, AVG(s.salary) as average_salary 
FROM departments as d
JOIN dept_emp as de
USING (dept_no)
JOIN salaries as s
USING (emp_no)
WHERE de.to_date = '9999-01-01' AND s.to_date = '9999-01-01'
GROUP BY d.dept_name
ORDER BY average_salary DESC LIMIT 1;

-- Question 8: 

SELECT e.first_name, e.last_name
FROM employees as e
JOIN salaries as s
USING (emp_no)
JOIN dept_emp as de
USING (emp_no)
JOIN departments as d
USING (dept_no)
WHERE de.dept_no = 'd001' AND s.to_date = '9999-01-01' AND de.to_date = '9999-01-01'
GROUP BY s.salary, e.first_name, e.last_name
ORDER BY s.salary DESC LIMIT 1;

-- Question 9: 

SELECT e.first_name, e.last_name, s.salary, d.dept_name
FROM employees as e
JOIN salaries as s
USING (emp_no)
JOIN dept_manager as dm
USING (emp_no)
JOIN departments as d 
USING (dept_no)
WHERE s.to_date = '9999-01-01' AND dm.to_date = '9999-01-01'
GROUP BY e.first_name, e.last_name, s.salary, d.dept_name
ORDER BY s.salary DESC LIMIT 1;

-- Question 10: 

SELECT d.dept_name, ROUND(AVG(s.salary)) AS average_salary
FROM dept_emp as de
JOIN salaries as s
USING (emp_no)
JOIN departments as d 
USING (dept_no)
GROUP BY d.dept_name
ORDER BY average_salary DESC;

-- Bonus 
-- Question 11: 

SELECT CONCAT(e.first_name, ' ', e.last_name) as 'Employee Name', d.dept_name as 'Department Name', CONCAT(e.first_name, ' ', e.last_name) as 'Manager Name'
FROM departments as d
JOIN dept_emp as de 
USING (dept_no)
JOIN employees as e
USING (emp_no)
JOIN dept_manager as dm
USING (dept_no)
WHERE dm.to_date = '9999-01-01' AND de.to_date = '9999-01-01'
GROUP BY e.first_name, e.last_name, d.dept_name, dm.emp_no;




























