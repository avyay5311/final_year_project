// static/js/exam.js
// WebSocket client for real-time warnings during exam

document.addEventListener('DOMContentLoaded', function() {
    // -------------------------------------------------
    // Timer Constants and State
    // -------------------------------------------------

    const EXAM_DURATION_SECONDS = 30  // 90 minutes
    const WARNING_THRESHOLD     = 20;   // 10 minutes remaining → chime + orange
    const CRITICAL_THRESHOLD    = 5;    // 1 minute remaining  → red pulse

    let timeRemaining  = EXAM_DURATION_SECONDS;
    let timerInterval  = null;
    let chimeTriggered = false;
    let autoSubmitting = false;

    // -------------------------------------------------
    // Chime Alert (Web Audio API)
    // -------------------------------------------------

    function playChime() {
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            const ctx = new AudioCtx();

            const beeps = [
                { freq: 880,  start: 0.0, end: 0.3 },
                { freq: 880,  start: 0.5, end: 0.8 },
                { freq: 1100, start: 1.0, end: 1.5 }
            ];

            beeps.forEach(function(beep) {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.value = beep.freq;
                gain.gain.setValueAtTime(0.3, ctx.currentTime + beep.start);
                gain.gain.setValueAtTime(0, ctx.currentTime + beep.end);
                osc.start(ctx.currentTime + beep.start);
                osc.stop(ctx.currentTime + beep.end + 0.1);
            });
        } catch(e) {
            // Audio not supported — fail silently
        }
    }

    // -------------------------------------------------
    // Timer Display
    // -------------------------------------------------

    function updateTimerDisplay(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs    = seconds % 60;
        const display = String(minutes).padStart(2, '0') + ':' + String(secs).padStart(2, '0');

        const timerDisplay = document.getElementById('timer-display');
        const timerEl      = document.getElementById('exam-timer');

        if (timerDisplay) timerDisplay.textContent = display;

        if (timerEl) {
            // Reset classes first
            timerEl.classList.remove('timer-warning', 'timer-critical');

            if (seconds <= CRITICAL_THRESHOLD) {
                timerEl.classList.add('timer-critical');
            } else if (seconds <= WARNING_THRESHOLD) {
                timerEl.classList.add('timer-warning');
            }
        }
    }

    // -------------------------------------------------
    // Auto Submit
    // -------------------------------------------------

    function autoSubmitExam() {
        if (autoSubmitting) return;
        autoSubmitting = true;

        // Remove beforeunload guard so form can submit
        window.removeEventListener('beforeunload', preventLeave);

        // Show time up overlay
        const overlay = document.getElementById('time-up-overlay');
        if (overlay) overlay.classList.remove('hidden');

        // Submit after 3 seconds so candidate can read the message
        setTimeout(function() {
            const form = document.getElementById('exam-form');
            if (form) form.submit();
        }, 3000);
    }

    // -------------------------------------------------
    // Countdown Timer
    // -------------------------------------------------

    function startTimer() {
        updateTimerDisplay(timeRemaining);

        timerInterval = setInterval(function() {
            timeRemaining -= 1;
            updateTimerDisplay(timeRemaining);

            // Chime at exactly 10 minutes remaining
            if (timeRemaining === WARNING_THRESHOLD && !chimeTriggered) {
                playChime();
                chimeTriggered = true;
            }

            // Auto-submit when timer hits zero
            if (timeRemaining <= 0) {
                clearInterval(timerInterval);
                autoSubmitExam();
            }
        }, 1000);
    }

    // -------------------------------------------------
    // WebSocket Connection
    // -------------------------------------------------
    
    const socket = io();
    
    const warningBar = document.getElementById('warning-bar');
    const warningMessage = document.getElementById('warning-message');
    
    let warningTimeout = null;
    
    // Handle connection
    socket.on('connect', function() {
        console.log('Connected to proctoring server');
    });
    
    // Handle disconnection
    socket.on('disconnect', function() {
        console.log('Disconnected from proctoring server');
    });
    
    // Handle warnings from server
    socket.on('warnings', function(data) {
        if (data.messages && data.messages.length > 0) {
            // Show warning bar
            warningBar.classList.remove('hidden');
            warningMessage.textContent = data.messages.join(' | ');
            
            // Clear previous timeout
            if (warningTimeout) {
                clearTimeout(warningTimeout);
            }
            
            // Auto-hide after 3 seconds if no new warnings
            warningTimeout = setTimeout(function() {
                warningBar.classList.add('hidden');
            }, 3000);
        }
    });
    
    // Handle proctoring status (warnings + identity)
    socket.on('proctoring_status', function(data) {
        // Handle warnings
        if (data.warnings && data.warnings.length > 0) {
            warningBar.classList.remove('hidden');
            warningMessage.textContent = data.warnings.join(' | ');
            
            if (warningTimeout) {
                clearTimeout(warningTimeout);
            }
            
            warningTimeout = setTimeout(function() {
                warningBar.classList.add('hidden');
            }, 3000);
        }
        
        // Handle identity status
        const identityIndicator = document.getElementById('identity-indicator');
        const identityText = document.getElementById('identity-text');
        
        if (identityIndicator && identityText) {
            if (data.identity_valid) {
                identityIndicator.style.color = '#27ae60';
                identityText.textContent = 'Identity Verified';
            } else {
                identityIndicator.style.color = '#e74c3c';
                identityText.textContent = 'Identity Not Detected';
            }
        }
    });
    
    // Handle connection status
    socket.on('connected', function(data) {
        console.log('Proctoring connection established:', data);
    });
    
    // -------------------------------------------------
    // Page Leave Prevention
    // -------------------------------------------------
    
    function preventLeave(e) {
        e.preventDefault();
        e.returnValue = 'Are you sure you want to leave? Your exam progress may be lost.';
        return e.returnValue;
    }
    
    window.addEventListener('beforeunload', preventLeave);
    
    // -------------------------------------------------
    // Exam Submission
    // -------------------------------------------------
    
    const examForm = document.getElementById('exam-form');
    if (examForm) {
        examForm.addEventListener('submit', function(e) {
            const confirmed = confirm(
                'Are you sure you want to submit the exam?\n\n' +
                'This action cannot be undone.'
            );
            
            if (!confirmed) {
                e.preventDefault();
            } else {
                // Remove beforeunload listener to allow submission
                window.removeEventListener('beforeunload', preventLeave);
                if (timerInterval) clearInterval(timerInterval);
            }
        });
    }
    
    // -------------------------------------------------
    // Video Feed Error Handling
    // -------------------------------------------------
    
    const videoFeed = document.getElementById('video-feed');
    const cameraStatusText = document.getElementById('camera-status-text');
    
    if (videoFeed) {
        videoFeed.onerror = function() {
            console.error('Video feed error');
            if (cameraStatusText) {
                cameraStatusText.textContent = 'Camera Error';
                cameraStatusText.style.color = '#e74c3c';
            }
        };
        
        videoFeed.onload = function() {
            if (cameraStatusText) {
                cameraStatusText.textContent = 'Camera Active';
                cameraStatusText.style.color = '#27ae60';
            }
        };
    }
    
    // -------------------------------------------------
    // Keyboard Shortcuts Prevention (Optional)
    // -------------------------------------------------
    
    // Uncomment to prevent certain keyboard shortcuts during exam
    /*
    document.addEventListener('keydown', function(e) {
        // Prevent Ctrl+C, Ctrl+V, Ctrl+P, etc.
        if (e.ctrlKey && ['c', 'v', 'p', 'u'].includes(e.key.toLowerCase())) {
            e.preventDefault();
            console.log('Keyboard shortcut blocked');
        }
        
        // Prevent F12 (Developer Tools)
        if (e.key === 'F12') {
            e.preventDefault();
        }
    });
    */
    
    // -------------------------------------------------
    // Right-click Prevention (Optional)
    // -------------------------------------------------
    
    // Uncomment to prevent right-click during exam
    /*
    document.addEventListener('contextmenu', function(e) {
        e.preventDefault();
    });
    */
    
    console.log('Exam monitoring initialized');

    // Start the exam countdown timer
    startTimer();
});
