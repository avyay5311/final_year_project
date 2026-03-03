-- schema.sql
-- Database schema for Exam Proctoring System
-- Run this file in MySQL to create the required tables

-- NOTE: sign_up table already exists - DO NOT RUN this part
-- CREATE TABLE quizo.sign_up (
--     email VARCHAR(255) PRIMARY KEY,
--     username VARCHAR(255) NOT NULL,
--     password VARCHAR(255) NOT NULL
-- );

-- -------------------------------------------------
-- New Tables for Proctoring System
-- -------------------------------------------------

-- Candidates table: Links user accounts to proctoring data
CREATE TABLE IF NOT EXISTS quizo.candidates (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(255) NOT NULL,
    face_encoding_path VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (email) REFERENCES sign_up(email) ON DELETE CASCADE
);

-- Exam sessions table: Tracks each exam attempt
CREATE TABLE IF NOT EXISTS quizo.exam_sessions (
    session_id INT PRIMARY KEY AUTO_INCREMENT,
    candidate_id INT NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    status ENUM('in_progress', 'completed', 'terminated') DEFAULT 'in_progress',
    FOREIGN KEY (candidate_id) REFERENCES candidates(id) ON DELETE CASCADE
);

-- Integrity reports table: Stores final proctoring reports
CREATE TABLE IF NOT EXISTS quizo.integrity_reports (
    report_id INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    integrity_score INT NOT NULL,
    risk_level ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
    report_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES exam_sessions(session_id) ON DELETE CASCADE
);

-- -------------------------------------------------
-- Indexes for better query performance
-- -------------------------------------------------

CREATE INDEX idx_candidates_email ON quizo.candidates(email);
CREATE INDEX idx_exam_sessions_candidate ON quizo.exam_sessions(candidate_id);
CREATE INDEX idx_integrity_reports_session ON quizo.integrity_reports(session_id);
