-- 기존 영어 진료과 코드를 한글로 업데이트
-- auth_user 테이블의 department 컬럼 업데이트

-- 1. auth_user 테이블의 department 업데이트
UPDATE auth_user 
SET department = '원무과'
WHERE department = 'admin';

UPDATE auth_user 
SET department = '호흡기내과'
WHERE department = 'respiratory';

UPDATE auth_user 
SET department = '외과'
WHERE department = 'surgery';

-- 2. patients_appointment 테이블의 doctor_department 업데이트
UPDATE patients_appointment 
SET doctor_department = '원무과'
WHERE doctor_department = 'admin';

UPDATE patients_appointment 
SET doctor_department = '호흡기내과'
WHERE doctor_department = 'respiratory';

UPDATE patients_appointment 
SET doctor_department = '외과'
WHERE doctor_department = 'surgery';

-- 3. (선택사항) 예약 제목에서 영어 진료과를 한글로 변경
UPDATE patients_appointment 
SET title = REPLACE(title, 'respiratory', '호흡기내과')
WHERE title LIKE '%respiratory%';

UPDATE patients_appointment 
SET title = REPLACE(title, 'surgery', '외과')
WHERE title LIKE '%surgery%';

-- 결과 확인
SELECT id, username, department, doctor_id 
FROM auth_user 
WHERE department IN ('원무과', '호흡기내과', '외과')
ORDER BY department, id;

SELECT id, title, doctor_name, doctor_department 
FROM patients_appointment 
WHERE doctor_department IN ('원무과', '호흡기내과', '외과')
ORDER BY doctor_department, start_time DESC
LIMIT 10;

