import mysql.connector
from config import Config

global cnx
global isInserted

#Create a connection to the database
cnx = mysql.connector.connect(
    host=Config.DB_HOST,
    user=Config.DB_USER,
    password=Config.DB_PASSWORD,
    database=Config.DB_NAME
)

def get_all_details():
    cursor = cnx.cursor(buffered=True)

    query = ("SELECT * FROM quizo.sign_up")
    cursor.execute(query)

    rows = cursor.fetchall()

    for row in rows:
        print(row)
    cursor.close()
    return

def insert_signup(email, username, password):
    try:
        #Create a cursor object
        cursor = cnx.cursor()

        query = "INSERT INTO quizo.sign_up (email, username, password) VALUES (%s, %s, %s)"
        # query2 = "INSERT INTO quizo.login_credentials () VALUES ()"
        
        cursor.execute(query, (email, username, password))
        cnx.commit()
        cursor.close()
        print("Sign-Up data credentials inserted successfully!")
        return 1

    except mysql.connector.Error as err:
        print("Error inserting the order item:", err)
        #Rollback changes if necessary
        cnx.rollback()
        return -1
    
    except Exception as e:
        print(f"An error occurred: {e}")
        #Rollback changes if necessary
        cnx.rollback()
        return -1

def search_login_credentials(email, password):
    #Create a cursor object
    cursor = cnx.cursor(buffered=True)

    query = ("SELECT email,password FROM quizo.sign_up where email=%s and password=%s")
    cursor.execute(query, (email, password))
    rows = cursor.fetchall()
    cursor.close()
    if rows:
        print("Data found")
        return True
    else:
        print("No data found.")
    return False


# -------------------------------------------------
# NEW FUNCTIONS FOR PROCTORING SYSTEM
# Added for Flask web application integration
# -------------------------------------------------

import json


def get_connection():
    """Create and return a new database connection."""
    return mysql.connector.connect(
        host=Config.DB_HOST,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        database=Config.DB_NAME
    )


def get_username_by_email(email):
    """Get username from email."""
    cursor = cnx.cursor(buffered=True)
    query = "SELECT username FROM quizo.sign_up WHERE email=%s"
    cursor.execute(query, (email,))
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else None


# -------------------------------------------------
# Candidate Functions
# -------------------------------------------------

def insert_candidate(email, username, face_encoding_path=None):
    """Insert new candidate record."""
    try:
        cursor = cnx.cursor()
        query = """
            INSERT INTO quizo.candidates (email, username, face_encoding_path)
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (email, username, face_encoding_path))
        cnx.commit()
        candidate_id = cursor.lastrowid
        cursor.close()
        print(f"Candidate inserted with ID: {candidate_id}")
        return candidate_id
    except mysql.connector.Error as err:
        print("Error inserting candidate:", err)
        cnx.rollback()
        return None


def get_candidate_by_email(email):
    """Get candidate record by email."""
    cursor = cnx.cursor(dictionary=True, buffered=True)
    query = "SELECT * FROM quizo.candidates WHERE email=%s"
    cursor.execute(query, (email,))
    result = cursor.fetchone()
    cursor.close()
    return result


def get_candidate_by_id(candidate_id):
    """Get candidate record by ID."""
    cursor = cnx.cursor(dictionary=True, buffered=True)
    query = "SELECT * FROM quizo.candidates WHERE id=%s"
    cursor.execute(query, (candidate_id,))
    result = cursor.fetchone()
    cursor.close()
    return result


def update_candidate_face_encoding(candidate_id, face_encoding_path):
    """Update candidate's face encoding path."""
    try:
        cursor = cnx.cursor()
        query = "UPDATE quizo.candidates SET face_encoding_path=%s WHERE id=%s"
        cursor.execute(query, (face_encoding_path, candidate_id))
        cnx.commit()
        cursor.close()
        return True
    except mysql.connector.Error as err:
        print("Error updating face encoding:", err)
        cnx.rollback()
        return False


# -------------------------------------------------
# Exam Session Functions
# -------------------------------------------------

def create_exam_session(candidate_id):
    """Create new exam session."""
    try:
        cursor = cnx.cursor()
        query = """
            INSERT INTO quizo.exam_sessions (candidate_id, status)
            VALUES (%s, 'in_progress')
        """
        cursor.execute(query, (candidate_id,))
        cnx.commit()
        session_id = cursor.lastrowid
        cursor.close()
        print(f"Exam session created with ID: {session_id}")
        return session_id
    except mysql.connector.Error as err:
        print("Error creating exam session:", err)
        cnx.rollback()
        return None


def end_exam_session(session_id, status='completed'):
    """End exam session and update status."""
    try:
        cursor = cnx.cursor()
        query = """
            UPDATE quizo.exam_sessions 
            SET end_time=CURRENT_TIMESTAMP, status=%s 
            WHERE session_id=%s
        """
        cursor.execute(query, (status, session_id))
        cnx.commit()
        cursor.close()
        return True
    except mysql.connector.Error as err:
        print("Error ending exam session:", err)
        cnx.rollback()
        return False


def get_exam_session(session_id):
    """Get exam session by ID."""
    cursor = cnx.cursor(dictionary=True, buffered=True)
    query = "SELECT * FROM quizo.exam_sessions WHERE session_id=%s"
    cursor.execute(query, (session_id,))
    result = cursor.fetchone()
    cursor.close()
    return result


# -------------------------------------------------
# Integrity Report Functions
# -------------------------------------------------

def save_integrity_report(session_id, integrity_score, risk_level, report_json):
    """Save integrity report to database."""
    try:
        cursor = cnx.cursor()
        query = """
            INSERT INTO quizo.integrity_reports 
            (session_id, integrity_score, risk_level, report_json)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (
            session_id,
            integrity_score,
            risk_level,
            json.dumps(report_json)
        ))
        cnx.commit()
        report_id = cursor.lastrowid
        cursor.close()
        print(f"Integrity report saved with ID: {report_id}")
        return report_id
    except mysql.connector.Error as err:
        print("Error saving integrity report:", err)
        cnx.rollback()
        return None


def get_integrity_report(session_id):
    """Get integrity report by session ID."""
    cursor = cnx.cursor(dictionary=True, buffered=True)
    query = "SELECT * FROM quizo.integrity_reports WHERE session_id=%s"
    cursor.execute(query, (session_id,))
    result = cursor.fetchone()
    cursor.close()
    if result and result.get('report_json'):
        result['report_json'] = json.loads(result['report_json'])
    return result


if __name__ == "__main__":
    print(get_all_details())
    # print(search_login_credentials('kumar1166@gmail.com', 'Kris@2223'))
    # insert_signup('kumar1166@gmail.com', 'kris6', 'Kris@2223')
    # print(get_all_details())