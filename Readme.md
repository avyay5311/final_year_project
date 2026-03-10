# AI-Powered Exam Proctoring System

A real-time, AI-driven online exam proctoring web application built with **Python, Flask, OpenCV, and Socket.IO**. The system monitors candidates during online examinations through multiple behavioral analysis channels—gaze tracking, blink detection, head-pose estimation, mouth-movement analysis, and face-identity verification—and produces a composite integrity score with risk classification.

---

## Features

- **User Registration & Face Enrollment** – Candidates register and capture a face encoding via webcam for identity verification.
- **Pre-Exam Calibration** – Two-step calibration (identity verification + gaze baseline) before the exam begins.
- **Real-Time Proctoring** – Live camera feed processed through a multi-channel AI pipeline during the exam.
- **WebSocket Warnings** – Flask-SocketIO broadcasts behavioral warnings (gaze deviation, face not detected, etc.) to the exam UI in real time.
- **Integrity Engine** – Aggregates per-channel summaries (gaze, blink, head-pose, mouth, face identity) into a single integrity score and risk level (LOW / MEDIUM / HIGH / CRITICAL).
- **Result Dashboard** – Post-exam report showing the integrity score, risk level, and detailed analysis flags.

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python, Flask, Flask-SocketIO |
| AI / CV | OpenCV (DNN face detection, landmark detection), Gaze tracking, Blink detection, Head-pose estimation, Mouth-movement analysis |
| Real-Time | WebSocket (Socket.IO) for live proctoring status |
| Database | MySQL |
| Frontend | HTML, CSS, JavaScript, Jinja2 Templates |

## Application Flow

1. **Register** → **Capture Face** → **Login**
2. **Calibration** (Verify Identity + Calibrate Gaze) → **Exam**
3. **Submit Exam** → **Integrity Report** → **Logout**

## Templates Overview

| Template | Purpose |
|----------|---------|
| `base.html` | Master layout inherited by all pages |
| `register.html` | New user registration form |
| `capture_face.html` | Webcam-based face encoding capture |
| `login.html` | User authentication |
| `calibration.html` | Pre-exam identity verification & gaze calibration (AJAX) |
| `exam.html` | Main exam page with live camera widget & Socket.IO warnings |
| `result.html` | Post-exam integrity score and flagged-behavior details |
| `error.html` | Generic error display |

---

## Resume – LaTeX Project Entry

Copy the block below into the **Projects** section of your LaTeX resume:

```latex
{\textbf{AI-Powered Exam Proctoring System} $|$ \emph{Python, Flask, OpenCV, Socket.IO, MySQL}}{\href{https://github.com/avyay5311/final_year_project}{GitHub}}
          \resumeItemListStart
              \resumeItem{Built a real-time exam proctoring web app that uses OpenCV DNN-based face detection, gaze tracking, blink detection, and head-pose estimation to monitor candidates during online examinations.}
              \resumeItem{Engineered a multi-channel integrity engine that aggregates five behavioral signals—gaze deviation, blink patterns, mouth movements, head pose, and face identity—to compute a composite integrity score with risk classification.}
              \resumeItem{Implemented WebSocket-driven live warnings using Flask-SocketIO, along with a two-step pre-exam calibration flow (identity verification and gaze baseline) to ensure reliable proctoring throughout the session.}
          \resumeItemListEnd
```
