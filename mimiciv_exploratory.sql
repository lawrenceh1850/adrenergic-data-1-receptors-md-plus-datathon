-- patients

select count(p.subject_id) from mimiciv_hosp.patients p ;

select pts.subject_id , pts.gender , pts.anchor_age  as age into pts_final from mimiciv_hosp.patients pts;

select count(subject_id) from pts_final ;

-- omr

select
	count(distinct o.subject_id)
from
	mimiciv_hosp.omr o;

drop table OMR_results;

select
	o.subject_id ,
	o.result_name ,
	o.result_value
into
	OMR_results
from
	mimiciv_hosp.omr o
where
	o.result_name in ('BMI (kg/m2)', 'BMI', 'Blood Pressure');

select
	b.subject_id ,
	b.result_name ,
	max(b.result_value) as max_value
into
	omr_bmi_bp
from
	omr_results as b
group by
	b.subject_id ,
	b.result_name;

select
	*
from
	omr_bmi_bp;

drop table omr_bmi_bp_final;

select
	omr_bmi_bp.subject_id,
	MAX(case when omr_bmi_bp.result_name in ('BMI', 'BMI (kg/m2)') then omr_bmi_bp.max_value else null end) as max_bmi,
	MAX(case when omr_bmi_bp.result_name = 'Blood Pressure' then omr_bmi_bp.max_value else null end) as max_blood_pressure
into
	omr_bmi_bp_final
from
	omr_bmi_bp
group by
	omr_bmi_bp.subject_id;

select
	*
from
	omr_bmi_bp_final;
;

select * from omr_bmi_bp_final obbf where obbf.subject_id =10246901;

-- d_icd_diagnoses
drop table diag_final ;
select diag.subject_id , diag.hadm_id , diag.icd_code , diag.icd_version into diag_final from mimiciv_hosp.diagnoses_icd diag;

-- admissions
drop table adm_final;
select distinct adm.subject_id , adm.hadm_id, string_agg(adm.race, '|') as race , string_agg(adm.insurance , '|') as insurance , string_agg(adm."language", '|') as language into adm_final from mimiciv_hosp.admissions adm group by adm.subject_id, adm.hadm_id  ;

select * from diag_final where subject_id = 10132888 ;

-- join

drop table initial_features ;

select
	p.subject_id,
	p.age,
	p.gender,
	o.bmi,
	o.max_blood_pressure,
	d.icd_code,
	d.icd_version,
	a.race,
	a.insurance,
	a.language
into
	initial_features
from
	pts_final p
full join omr_bmi_bp_final o on
	p.subject_id = o.subject_id
full join diag_final d on
	p.subject_id = d.subject_id
full join adm_final a on
	a.subject_id = p.subject_id;

-- remove patients without icd codes
drop table initial_features_icd_notnull;

select * into initial_features_icd_notnull from initial_features i where icd_code notnull;

select count(distinct inn.subject_id) from initial_features_icd_notnull inn;
select count(*) from initial_features_icd_notnull inn;

-- export to csv
copy initial_features_icd_notnull to '/Users/lhuang21/Documents/Programming/MD+_Datathon_2023/initial_features_raw.csv' delimiter ',' csv header;


-- add hadm_id to data

select
	p.subject_id,
	d.hadm_id,
	p.age,
	p.gender,
	o.max_bmi,
	o.max_blood_pressure,
	d.icd_code,
	d.icd_version,
	a.race,
	a.insurance,
	a.language
into
	initial_features_w_hadm_id
from
	pts_final p
full join omr_bmi_bp_final o on
	p.subject_id = o.subject_id
full join diag_final d on
	p.subject_id = d.subject_id
full join adm_final a on
	a.subject_id = p.subject_id;

-- remove patients without icd codes
drop table initial_features_w_hadm_id_icd_notnull;

select * into initial_features_w_hadm_id_icd_notnull from initial_features_w_hadm_id i where icd_code notnull;

select count(distinct inn.subject_id) from initial_features_w_hadm_id_icd_notnull inn;
select count(*) from initial_features_w_hadm_id_icd_notnull inn;

-- export to csv
copy initial_features_w_hadm_id_icd_notnull to '/Users/lhuang21/Documents/Programming/MD+_Datathon_2023/initial_features_w_hadm_id_raw.csv' delimiter ',' csv header;
10246901

-- exploratory queries

select count(distinct i.subject_id) from initial_features i;

select * from initial_features_icd_notnull i ;

select count(distinct i.icd_code) from initial_features i;

select count(distinct d_diag.icd_code) from mimiciv_hosp.d_icd_diagnoses d_diag;

select count(distinct i.subject_id) from initial_features i join mimiciv_hosp.d_icd_diagnoses did on i.icd_code =did.icd_code ;

-- diagnoses

select * from mimiciv_hosp.diagnoses_icd di where di.subject_id = 10246901 AND di.icd_code like 'N186%';
select * from mimiciv_hosp.diagnoses_icd di where di.subject_id = 10246901 AND di.icd_code like 'I120%';

select pts.subject_id , pts.gender , pts.anchor_age  as age from mimiciv_hosp.patients pts where pts.subject_id = 10246901;
