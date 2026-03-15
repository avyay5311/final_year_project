# app.py
# Main Flask application for Exam Proctoring System

import cv2
import sys
import os
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
from flask_socketio import SocketIO, emit

from config import Config

# Import database helper
from database import db_helper

# Import services
from services import (
    capture_face_encoding,
    verify_face_exists,
    get_encoding_path,
    verify_identity,
    start_exam_proctoring,
    stop_exam_proctoring,
    get_current_warnings,
    get_video_frame,
    get_identity_status,
    run_integrity_engine,

    calibrate_gaze,
    calibration_exists
)
from services.camera_service import camera

# -------------------------------------------------
# Flask App Initialization
# -------------------------------------------------

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# -------------------------------------------------
# Warning Broadcast Thread
# -------------------------------------------------

warning_thread = None
warning_thread_running = False


def broadcast_warnings():
    """Background thread to broadcast warnings via WebSocket."""
    global warning_thread_running
    
    while warning_thread_running:
        warnings = get_current_warnings()
        identity_valid = get_identity_status()
        socketio.emit('proctoring_status', {
            'warnings': warnings,
            'identity_valid': identity_valid
        })
        time.sleep(0.5)  # Check every 500ms


# -------------------------------------------------
# Routes: Authentication
# -------------------------------------------------

@app.route('/')
def index():
    """Home page - redirect to login."""
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    User registration page.
    - GET: Render registration form
    - POST: Create account and redirect to face capture
    """
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Validate input
        if not all([email, username, password]):
            return render_template('register.html', error="All fields are required")
        
        # Insert into sign_up table
        result = db_helper.insert_signup(email, username, password)
        if result != 1:
            return render_template('register.html', error="Email already registered")
        
        # Create candidate record
        candidate_id = db_helper.insert_candidate(email, username)
        if candidate_id is None:
            return render_template('register.html', error="Failed to create candidate")
        
        # Store candidate_id in session for face capture
        session['pending_registration'] = {
            'candidate_id': candidate_id,
            'email': email
        }
        
        return redirect(url_for('capture_face'))
    
    return render_template('register.html')


@app.route('/capture_face', methods=['GET', 'POST'])
def capture_face():
    """
    Face capture page during registration.
    - GET: Render face capture instructions
    - POST: Trigger face capture
    """
    if 'pending_registration' not in session:
        return redirect(url_for('register'))
    
    if request.method == 'POST':
        candidate_id = session['pending_registration']['candidate_id']
        
        # Capture face encoding using registration service
        success, encoding_path, message = capture_face_encoding(candidate_id)
        
        if success:
            # Update candidate with encoding path
            db_helper.update_candidate_face_encoding(candidate_id, encoding_path)
            
            # Clear pending registration
            del session['pending_registration']
            
            return redirect(url_for('login', message="Registration successful! Please login."))
        else:
            return render_template('capture_face.html', error=message)
    
    return render_template('capture_face.html')


@app.route('/registration_video_feed')
def registration_video_feed():
    """
    Video feed endpoint for face registration.
    Simple camera feed without AI processing.
    """
    def generate_frames():
        camera.start()
        while True:
            frame = camera.read_frame()
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    User login page.
    - GET: Render login form
    - POST: Validate credentials and create session
    """
    message = request.args.get('message')
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Validate credentials
        if db_helper.search_login_credentials(email, password):
            # Get candidate info
            candidate = db_helper.get_candidate_by_email(email)
            
            if candidate is None:
                return render_template('login.html', error="Candidate profile not found")
            
            if not verify_face_exists(candidate['id']):
                return render_template('login.html', error="Face encoding not found. Please re-register.")
            
            # Create session
            session['logged_in'] = True
            session['email'] = email
            session['candidate_id'] = candidate['id']
            session['username'] = candidate['username']
            
            return redirect(url_for('calibration'))
        else:
            return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html', message=message)


@app.route('/logout')
def logout():
    """Clear session and redirect to login."""
    session.clear()
    return redirect(url_for('login'))


# -------------------------------------------------
# Routes: Exam Flow
# -------------------------------------------------

@app.route('/calibration', methods=['GET', 'POST'])
def calibration():
    """
    Calibration page before exam.
    - Confirms primary identity
    - Prepares for exam
    """
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Proceed to exam
        return redirect(url_for('exam'))
    
    return render_template('calibration.html', username=session.get('username'))


@app.route('/calibration_video_feed')
def calibration_video_feed():
    """
    Video feed endpoint for calibration page.
    Simple camera feed without AI processing.
    """
    def generate_frames():
        camera.start()
        while True:
            frame = camera.read_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/verify_identity_check')
def verify_identity_check():
    """
    Verify candidate identity via AJAX.
    Returns JSON with verification result.
    """
    if 'logged_in' not in session:
        return jsonify({'verified': False, 'message': 'Not logged in'})
    
    candidate_id = session.get('candidate_id')
    if not candidate_id:
        return jsonify({'verified': False, 'message': 'No candidate ID found'})
    
    try:
        print(f"Verifying identity for candidate: {candidate_id}")
        verified, distance, message = verify_identity(candidate_id)
        print(f"Verification result: verified={verified}, distance={distance}, message={message}")
        
        return jsonify({
            'verified': verified,
            'distance': distance,
            'message': message
        })
    except Exception as e:
        print(f"Error during identity verification: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'verified': False,
            'distance': None,
            'message': f'Verification error: {str(e)}'
        })


@app.route('/calibrate_gaze_check')
def calibrate_gaze_check():
    """
    Perform gaze calibration via AJAX.
    User should be looking at the center of the screen.
    Returns JSON with calibration result.
    """
    if 'logged_in' not in session:
        return jsonify({'calibrated': False, 'message': 'Not logged in'})
    
    candidate_id = session.get('candidate_id')
    if not candidate_id:
        return jsonify({'calibrated': False, 'message': 'No candidate ID found'})
    
    try:
        print(f"Starting gaze calibration for candidate: {candidate_id}")
        success, message, _ = calibrate_gaze(candidate_id, duration_sec=2.0)
        print(f"Calibration result: success={success}, message={message}")
        
        return jsonify({
            'calibrated': success,
            'message': message
        })
    except Exception as e:
        print(f"Error during gaze calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'calibrated': False,
            'message': f'Calibration error: {str(e)}'
        })


@app.route('/exam')
def exam():
    """
    Main exam page.
    - Displays exam content
    - Shows camera feed widget
    - Receives real-time warnings via WebSocket
    """
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Create exam session in database
    candidate_id = session.get('candidate_id')
    exam_session_id = db_helper.create_exam_session(candidate_id)
    
    if exam_session_id is None:
        return render_template('error.html', error="Failed to create exam session. Please try again.")
    
    session['exam_session_id'] = exam_session_id
    print(f"Exam session created: {exam_session_id} for candidate: {candidate_id}")
    
    # Start proctoring using exam service
    success, message = start_exam_proctoring(candidate_id)
    if not success:
        return render_template('error.html', error=message)
    
    # Start warning broadcast thread
    global warning_thread, warning_thread_running
    warning_thread_running = True
    warning_thread = threading.Thread(target=broadcast_warnings, daemon=True)
    warning_thread.start()
    
    return render_template('exam.html', username=session.get('username'))


@app.route('/video_feed')
def video_feed():
    """
    Video feed endpoint.
    Streams MJPEG frames to browser.
    """
    def generate_frames():
        while True:
            frame = get_video_frame()
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    """
    Submit exam and generate integrity report.
    - Stops proctoring
    - Finalizes channels
    - Runs integrity engine
    - Saves report to database
    """
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Stop warning broadcast
    global warning_thread_running
    warning_thread_running = False
    
    # Get session_id early and validate
    session_id = session.get('exam_session_id')
    if session_id is None:
        print("ERROR: exam_session_id is None in submit_exam")
        return render_template('error.html', error="No active exam session found. Please start an exam first.")
    
    print(f"Submitting exam for session: {session_id}")
    
    # Stop proctoring using exam service
    stopped, stop_msg = stop_exam_proctoring()
    print(f"Stop proctoring: {stopped} - {stop_msg}")
    
    # Run integrity engine
    report = run_integrity_engine()
    
    # Save to database
    db_helper.end_exam_session(session_id, 'completed')
    report_id = db_helper.save_integrity_report(
        session_id=session_id,
        integrity_score=report['integrity_score'],
        risk_level=report['risk_level'],
        report_json=report
    )
    
    if report_id is None:
        print(f"ERROR: Failed to save integrity report for session {session_id}")
        return render_template('error.html', error="Failed to save integrity report.")
    
    print(f"Integrity report saved with ID: {report_id}")
    
    return redirect(url_for('result'))


@app.route('/result')
def result():
    """
    Display exam result with integrity report.
    """
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    session_id = session.get('exam_session_id')
    report = db_helper.get_integrity_report(session_id)
    
    if report is None:
        return render_template('error.html', error="Report not found")
    
    # Extract human-readable reasons from flags
    report_data = report['report_json']
    flags = report_data.get('flags', [])
    
    # Convert flags to readable reasons (show top 10)
    reasons = []
    for flag in flags[:10]:
        flag_type = flag.get('type', 'Unknown')
        severity = flag.get('severity', '')
        duration = flag.get('duration')
        
        # Format the reason string
        reason = f"[{severity}] {flag_type.replace('_', ' ').title()}"
        if duration:
            reason += f" ({duration:.1f}s)"
        reasons.append(reason)
    
    return render_template('result.html',
                         username=session.get('username'),
                         integrity_score=report['integrity_score'],
                         risk_level=report['risk_level'],
                         reasons=reasons,
                         channel_penalties=report_data.get('channel_penalties', {}),
                         flag_summary=report_data.get('flag_summary', {}))


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------

@app.route('/api/warnings')
def api_warnings():
    """API endpoint to get current warnings."""
    warnings = get_current_warnings()
    return jsonify({'warnings': warnings})


# -------------------------------------------------
# WebSocket Events
# -------------------------------------------------

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    print("Client connected")
    emit('connected', {'status': 'ok'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print("Client disconnected")


# -------------------------------------------------
# Main Entry Point
# -------------------------------------------------

if __name__ == '__main__':
    Config.init_directories()
    print("Starting Exam Proctoring System...")
    print("Access at: http://localhost:5000")
    socketio.run(app, debug=False, host='127.0.0.1', port=5000, use_reloader=False, allow_unsafe_werkzeug=True)
