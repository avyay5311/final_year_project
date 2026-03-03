// static/js/exam.js
// WebSocket client for real-time warnings during exam

document.addEventListener('DOMContentLoaded', function() {
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
});
