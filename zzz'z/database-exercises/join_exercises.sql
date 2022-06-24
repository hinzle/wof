-- Join Example Database
-- 1
use join_example_db;

select * from users;
select * from roles;
-- select * from users join roles on 1=1;

-- using(countrycode)

-- 2
-- guess 4
select * 
from users 
join roles;

-- guess 6
select * 
from users 
left join roles
on roles.id=users.role_id;

-- guess 4
select * 
from users 
right join roles
on roles.id=users.role_id;

-- 3
select roles.name, count(users.role_id)
from users 
right join roles
on roles.id=users.role_id
group by roles.id;

-- Employees Database
-- 1
use employees;

-- 2 Using the example in the Associative Table Joins section as a guide, write a query that shows each department along with the name of the current manager for that department.
/*
select *
from dept_manager
where to_date > NOW();
-- group by dept_no;
*/

select departments.dept_name as 'Department Name',
	concat(employees.first_name,' ',employees.last_name) as 'Department Manager'
from dept_manager
join employees
using(emp_no)
-- on dept_manager.emp_no=employees.emp_no
join departments
using(dept_no)
where to_date>NOW();


-- where to_date>NOW()
-- group by dept_no
-- join departments
-- using(dept_no);

-- 3
select departments.dept_name as 'Department Name',
	concat(employees.first_name,' ',employees.last_name) as 'Department Manager'
from dept_manager
join employees
using(emp_no)
-- on dept_manager.emp_no=employees.emp_no
join departments
using(dept_no)
where to_date>NOW()
	and employees.gender like 'f';

-- 4
select titles.title as Title,
	count(title) as Count
from departments
join dept_emp
using(dept_no)
join titles
using(emp_no)
where dept_emp.to_date>NOW()
	and titles.to_date>NOW()
	and dept_name like 'customer service'
group by titles.title
order by titles.title;

-- 5
select -- *

-- /*
departments.dept_name as 'Department Name',
	concat(employees.first_name,' ',employees.last_name) as 'Name', 
	salaries.salary as Salary


-- */	
	-- titles.title as Title,
	-- count(title) as Count
from departments
join dept_manager
using(dept_no)
join salaries
using(emp_no)
join employees
using(emp_no)
where salaries.to_date>NOW()
	and dept_manager.to_date>NOW()
	-- and dept_name like 'customer service'
-- group by titles.title
order by departments.dept_name
-- limit 10
;

-- the following work is copied from a deskmate. i made a save mistake. we did all work together on much of the code and i did solve all the questions myself before the data loss.


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

SELECT 
CONCAT(e.first_name, ' ', e.last_name) as 'Employee Name', 
d.dept_name as 'Department Name', 
CONCAT(e.first_name, ' ', e.last_name) as 'Manager Name'
FROM departments as d
JOIN dept_emp as de 
USING (dept_no)
JOIN employees as e
USING (emp_no)
JOIN dept_manager as dm
USING (dept_no)
WHERE dm.to_date = '9999-01-01' AND de.to_date = '9999-01-01'
GROUP BY e.first_name, e.last_name, d.dept_name, dm.emp_no;







