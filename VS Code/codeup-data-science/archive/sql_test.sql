select firstname, lastname, cp.prod_desc
from customers c
left join current_product cp
using (prod_id)
where prod_id not like 'null'
;

-- 17
select avg(n), min(n), max(n), sum(n), stddev(n)
from numbers
;

-- 21
select distinct make from cars;

-- 22
select count(distinct make) from cars;

--24
select 
	distinct(make) 'make', 
	count(distinct make) 'count'
from cars
group by 'make';