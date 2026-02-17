-- PS3_Zhou.sql
-- Econ 5253, Spring 2026

-- (a) Read in the Florida insurance CSV file
.mode csv
.import FL_insurance_sample.csv florida_insurance

-- (b) Print the first 10 rows
SELECT * FROM florida_insurance LIMIT 10;

-- (c) List unique counties in the sample
SELECT DISTINCT county
FROM florida_insurance
ORDER BY county;

-- (d) Compute average property appreciation from 2011 to 2012
SELECT AVG(CAST(tiv_2012 AS REAL) - CAST(tiv_2011 AS REAL)) AS avg_appreciation
FROM florida_insurance;

-- (e) Frequency table of the construction variable
SELECT
    construction,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 1.0 / (SELECT COUNT(*) FROM florida_insurance), 4) AS fraction
FROM florida_insurance
GROUP BY construction
ORDER BY count DESC;
