-- Add transcript and transcriptTimestamps columns to Videos table
-- Run this SQL script in your MySQL database

USE pbl6;

-- Check if columns exist before adding
ALTER TABLE Videos 
ADD COLUMN IF NOT EXISTS transcript TEXT NULL COMMENT 'Transcribed text from video',
ADD COLUMN IF NOT EXISTS transcriptTimestamps TEXT NULL COMMENT 'JSON array of timestamps for each sentence';

-- Verify the changes
DESCRIBE Videos;
