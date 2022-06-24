-- 1.0
use innis_1652;

CREATE TEMPORARY TABLE employees_with_departments
	AS 
		SELECT -- de.emp_no, 
		first_name, 
		last_name,
		-- dept_no, 
		dept_name
		FROM employees.employees
		JOIN employees.dept_emp de USING(emp_no)
		JOIN employees.departments d USING(dept_no)

;

-- 1.a
ALTER TABLE employees_with_departments ADD full_name VARCHAR(40);
	-- (
	-- select max(length(concat(first_name,' ', last_name)))
	-- from employees_with_departments 
	-- )

-- 1.b
UPDATE employees_with_departments
SET full_name = CONCAT(first_name, ' ', last_name);

-- 1.c
ALTER TABLE employees_with_departments DROP COLUMN first_name, last_name ;

-- 1.d
-- create temp table with a subquery

-- select max(length(concat(first_name,' ', last_name)))
-- from employees_with_departments
-- ;

-- DESCRIBE employees_with_departments;
-- DROP table employees_with_departments;

-- 2
CREATE TEMPORARY TABLE temp_payments
	AS
		SELECT *
		FROM sakila.payment;
ALTER TABLE temp_payments ADD temp_amt dec(20,2);
UPDATE temp_payments SET temp_amt = 100*amount;
ALTER TABLE temp_payments DROP COLUMN amount;
ALTER TABLE temp_payments ADD amount int;
UPDATE temp_payments SET amount = temp_amt;
ALTER TABLE temp_payments DROP COLUMN temp_amt;

SELECT *
FROM temp_payments
limit 5;

-- 3
-- copy paste from instructor demo

create temporary table historic_aggregates as (
    select avg(salary) as avg_salary, std(salary) as std_salary
    from employees.salaries 
);

create temporary table current_info as (
    select dept_name, avg(salary) as department_current_average
    from employees.salaries
    join employees.dept_emp using(emp_no)
    join employees.departments using(dept_no)
    where employees.dept_emp.to_date > curdate()
    and employees.salaries.to_date > curdate()
    group by dept_name
);

select * from current_info;

alter table current_info add historic_avg float(10,2);
alter table current_info add historic_std float(10,2);
alter table current_info add zscore float(10,2);

update current_info set historic_avg = (select avg_salary from historic_aggregates);
update current_info set historic_std = (select std_salary from historic_aggregates);

select * from current_info;

update current_info 
set zscore = (department_current_average - historic_avg) / historic_std;

select * from current_info
order by zscore desc;

-- SELECT Database();

-- SHOW TABLES;
-- DESCRIBE temp_payments;
-- DROP TABLE temp_payments;


