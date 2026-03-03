# database/__init__.py
# Database package

from database.db_helper import (
    get_all_details,
    insert_signup,
    search_login_credentials,
    get_username_by_email,
    insert_candidate,
    get_candidate_by_email,
    get_candidate_by_id,
    update_candidate_face_encoding,
    create_exam_session,
    end_exam_session,
    get_exam_session,
    save_integrity_report,
    get_integrity_report
)

__all__ = [
    'get_all_details',
    'insert_signup',
    'search_login_credentials',
    'get_username_by_email',
    'insert_candidate',
    'get_candidate_by_email',
    'get_candidate_by_id',
    'update_candidate_face_encoding',
    'create_exam_session',
    'end_exam_session',
    'get_exam_session',
    'save_integrity_report',
    'get_integrity_report'
]
