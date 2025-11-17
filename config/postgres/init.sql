-- EFMM-Toll Database Initialization
CREATE DATABASE efmm_toll;
CREATE USER efmm_user WITH PASSWORD 'efmm_password';
GRANT ALL PRIVILEGES ON DATABASE efmm_toll TO efmm_user;
