-- business_queries.sql
-- A set of business-question queries an airline ops analyst would actually ask.
-- Run these with: sqlite3 data/flights.db < sql/business_queries.sql
-- (or paste into the sql/run_queries.py runner for formatted output)

-- ============================================================
-- Q1: Which airlines have the worst on-time performance?
-- ============================================================
SELECT
    carrier_code,
    COUNT(*)                                   AS total_flights,
    ROUND(AVG(is_delayed_15) * 100, 1)          AS pct_delayed_15min,
    ROUND(AVG(arr_delay_minutes), 1)            AS avg_arr_delay_min,
    ROUND(AVG(cancelled) * 100, 2)              AS pct_cancelled
FROM flights
GROUP BY carrier_code
ORDER BY pct_delayed_15min DESC;


-- ============================================================
-- Q2: Which origin airports create the most delay risk?
-- ============================================================
SELECT
    origin,
    COUNT(*)                                  AS departures,
    ROUND(AVG(is_delayed_15) * 100, 1)         AS pct_delayed_15min,
    ROUND(AVG(dep_delay_minutes), 1)           AS avg_dep_delay_min
FROM flights
GROUP BY origin
ORDER BY pct_delayed_15min DESC
LIMIT 10;


-- ============================================================
-- Q3: Delay rate by hour of day AND season (route x time heatmap source)
-- ============================================================
SELECT
    season,
    crs_dep_hour,
    COUNT(*)                            AS flights,
    ROUND(AVG(is_delayed_15) * 100, 1)  AS pct_delayed_15min
FROM flights
GROUP BY season, crs_dep_hour
ORDER BY season, crs_dep_hour;


-- ============================================================
-- Q4: Rank each carrier's WORST 3 routes (origin->dest) by delay rate
--     -> demonstrates window functions (RANK / PARTITION BY)
-- ============================================================
WITH route_stats AS (
    SELECT
        carrier_code,
        origin,
        dest,
        COUNT(*) AS n_flights,
        AVG(is_delayed_15) AS delay_rate
    FROM flights
    GROUP BY carrier_code, origin, dest
    HAVING COUNT(*) >= 30        -- ignore tiny/noisy route samples
),
ranked AS (
    SELECT *,
           RANK() OVER (PARTITION BY carrier_code ORDER BY delay_rate DESC) AS rnk
    FROM route_stats
)
SELECT carrier_code, origin, dest, n_flights, ROUND(delay_rate*100,1) AS pct_delayed
FROM ranked
WHERE rnk <= 3
ORDER BY carrier_code, rnk;


-- ============================================================
-- Q5: Month-over-month delay trend with a 3-month moving average
--     -> demonstrates window function (moving average)
-- ============================================================
WITH monthly AS (
    SELECT
        month,
        ROUND(AVG(is_delayed_15) * 100, 2) AS pct_delayed
    FROM flights
    GROUP BY month
)
SELECT
    month,
    pct_delayed,
    ROUND(AVG(pct_delayed) OVER (
        ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS moving_avg_3mo
FROM monthly
ORDER BY month;


-- ============================================================
-- Q6: Root-cause breakdown - what actually causes delays, by season?
-- ============================================================
SELECT
    season,
    delay_cause,
    COUNT(*) AS n_delays,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY season), 1) AS pct_of_season_delays
FROM flights
WHERE is_delayed_15 = 1 AND delay_cause != ''
GROUP BY season, delay_cause
ORDER BY season, n_delays DESC;


-- ============================================================
-- Q7: Cascading delay proxy - does departure delay predict arrival delay
--     more strongly on short-haul vs long-haul flights?
-- ============================================================
SELECT
    CASE WHEN distance < 800 THEN 'Short-haul (<800mi)'
         WHEN distance < 1800 THEN 'Medium-haul (800-1800mi)'
         ELSE 'Long-haul (1800mi+)' END AS haul_type,
    COUNT(*) AS flights,
    ROUND(AVG(arr_delay_minutes - dep_delay_minutes), 2) AS avg_recovery_minutes
FROM flights
WHERE cancelled = 0
GROUP BY haul_type;


-- ============================================================
-- Q8: Weekend vs weekday delay comparison (feeds the t-test in Python)
-- ============================================================
SELECT
    CASE WHEN day_of_week IN (4, 5, 6) THEN 'Weekend/Friday' ELSE 'Weekday' END AS day_type,
    COUNT(*) AS flights,
    ROUND(AVG(is_delayed_15) * 100, 2) AS pct_delayed
FROM flights
GROUP BY day_type;
