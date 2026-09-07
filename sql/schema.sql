-- schema.sql
-- Normalized star-schema style design: a fact table (flights) plus small
-- dimension tables (airlines, airports). This is what interviewers expect
-- to see instead of one flat denormalized table.

DROP TABLE IF EXISTS flights;
DROP TABLE IF EXISTS airlines;
DROP TABLE IF EXISTS airports;

CREATE TABLE airlines (
    carrier_code TEXT PRIMARY KEY,
    carrier_name TEXT
);

CREATE TABLE airports (
    airport_code TEXT PRIMARY KEY,
    congestion_factor REAL  -- synthetic-data-only field; drop for real BTS data
);

CREATE TABLE flights (
    flight_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_date       TEXT,
    carrier_code      TEXT REFERENCES airlines(carrier_code),
    flight_num        INTEGER,
    tail_num          TEXT,
    origin             TEXT REFERENCES airports(airport_code),
    dest               TEXT REFERENCES airports(airport_code),
    crs_dep_hour      INTEGER,
    crs_dep_minute    INTEGER,
    crs_elapsed_time  INTEGER,
    distance          INTEGER,
    day_of_week       INTEGER,   -- 0=Mon .. 6=Sun
    month             INTEGER,
    season            TEXT,
    dep_delay_minutes REAL,
    arr_delay_minutes REAL,
    is_delayed_15     INTEGER,   -- 1 if arrival delay >= 15 min
    cancelled         INTEGER,
    delay_cause       TEXT
);

CREATE INDEX idx_flights_carrier ON flights(carrier_code);
CREATE INDEX idx_flights_origin  ON flights(origin);
CREATE INDEX idx_flights_date    ON flights(flight_date);
