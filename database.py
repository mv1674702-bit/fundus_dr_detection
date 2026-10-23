import sqlite3


DATABASE = "patients.db"



# ============================================
# DATABASE CONNECTION
# ============================================

def get_connection():

    conn = sqlite3.connect(DATABASE)

    return conn



# ============================================
# CREATE PATIENT TABLE
# ============================================

def create_database():

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

    CREATE TABLE IF NOT EXISTS patients(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        patient_name TEXT NOT NULL,

        age INTEGER,

        prediction TEXT,

        confidence REAL,

        image TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )

    """)


    conn.commit()

    conn.close()



# ============================================
# CREATE USER TABLE
# ============================================

def create_users_table():

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

    CREATE TABLE IF NOT EXISTS users(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        username TEXT UNIQUE,

        email TEXT UNIQUE,

        password TEXT

    )

    """)


    conn.commit()

    conn.close()



# ============================================
# REGISTER USER
# ============================================

def register_user(
        username,
        email,
        password
):

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

    INSERT INTO users
    (
        username,
        email,
        password
    )

    VALUES(?,?,?)

    """,
    (
        username,
        email,
        password
    ))


    conn.commit()

    conn.close()



# ============================================
# LOGIN USER
# ============================================

def login_user(
        email,
        password
):

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

    SELECT * FROM users

    WHERE email=?

    AND password=?

    """,
    (
        email,
        password
    ))


    user = cursor.fetchone()


    conn.close()


    return user



# ============================================
# INSERT PATIENT PREDICTION
# ============================================

def insert_patient(
        name,
        age,
        prediction,
        confidence,
        image
):

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

    INSERT INTO patients

    (
        patient_name,
        age,
        prediction,
        confidence,
        image
    )

    VALUES(?,?,?,?,?)

    """,
    (
        name,
        age,
        prediction,
        confidence,
        image
    ))


    conn.commit()

    conn.close()



# ============================================
# GET ALL PATIENT RECORDS
# ============================================

def get_all_patients():

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

    SELECT * FROM patients

    ORDER BY id DESC

    """)


    data = cursor.fetchall()


    conn.close()


    return data



# ============================================
# SEARCH PATIENT
# ============================================

def search_patient(name):

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

    SELECT * FROM patients

    WHERE patient_name LIKE ?

    ORDER BY id DESC

    """,
    (
        "%" + name + "%",
    ))


    data = cursor.fetchall()


    conn.close()


    return data



# ============================================
# TOTAL PATIENT COUNT
# ============================================

def get_total_patients():

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute("""

    SELECT COUNT(*) FROM patients

    """)


    count = cursor.fetchone()[0]


    conn.close()


    return count