
-- 2
use albums_db;

-- 3.a) 31
select * from albums;

-- 3.b) 31
select distinct name from albums;

-- 3.c) id
describe albums;

-- 3.d) 1967, 2011
SELECT * FROM albums ORDER BY release_date;
SELECT * FROM albums ORDER BY release_date DESC;

-- 4.a) The Dark Side of the Moon, The Wall
SELECT * FROM albums WHERE artist='Pink Floyd';

-- 4.b) 1967
SELECT release_date FROM albums WHERE name= 'Sgt. Pepper\'s Lonely Hearts Club Band';

-- 4.c) Grunge, Alternative rock
SELECT genre FROM albums WHERE name='Nevermind';

-- 4.d) 
/* 
Jagged Little Pill
Come On Over
Falling into You
Let's Talk About Love
Dangerous
The Immaculate Collection
Titanic: Music from the Motion Picture
Metallica
Nevermind
Supernatural
*/
SELECT name FROM albums WHERE release_date LIKE '199%';

-- 4.e) 
/*
Grease: The Original Soundtrack from the Motion Picture
Bad
Sgt. Pepper's Lonely Hearts Club Band
Dirty Dancing
Let's Talk About Love
Dangerous
The Immaculate Collection
Abbey Road
Born in the U.S.A.
Brothers in Arms
Titanic: Music from the Motion Picture
Nevermind
The Wall
*/
SELECT name FROM albums WHERE sales < 20;

-- 4.f) the first is an exact match. the second line fixes the second problem
/*
Sgt. Pepper's Lonely Hearts Club Band
1
Abbey Road
Born in the U.S.A.
Supernatural
*/

SELECT name FROM albums WHERE genre ='rock';

SELECT name FROM albums WHERE genre LIKE '%rock%';




