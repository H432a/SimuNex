// Dark Mode Toggle
const themeToggle = document.getElementById('themeToggle');
const moonIcon = themeToggle.querySelector('i');

// Function to toggle dark mode
function toggleDarkMode(isDarkMode) {
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        moonIcon.classList.remove('fa-moon');
        moonIcon.classList.add('fa-sun');
    } else {
        document.body.classList.remove('dark-mode');
        moonIcon.classList.remove('fa-sun');
        moonIcon.classList.add('fa-moon');
    }
}

// Check for saved theme preference on page load
window.addEventListener('load', () => {
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    toggleDarkMode(savedDarkMode); // Apply the saved theme
});

// Toggle dark mode on button click
themeToggle.addEventListener('click', () => {
    const isDarkMode = !document.body.classList.contains('dark-mode');
    toggleDarkMode(isDarkMode);
    localStorage.setItem('darkMode', isDarkMode); // Save the preference
});

// Timer Functionality
let timerInterval;
let timeLimit;
let isPaused = false;
let remainingTime = 0;

function startTimer() {
    const timeInput = document.getElementById('timeInput');
    if (!timeInput || timeInput.value < 1) {
        alert('Please enter a valid time (minimum 1 minute).');
        return;
    }

    if (isPaused) {
        timeLimit = remainingTime; // Use remaining time if paused
    } else {
        timeLimit = timeInput.value * 60; // Convert minutes to seconds
    }

    clearInterval(timerInterval); // Clear any existing timer

    timerInterval = setInterval(() => {
        if (!isPaused) {
            timeLimit--;
            const minutes = Math.floor(timeLimit / 60);
            const seconds = timeLimit % 60;
            document.getElementById('time').innerText = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
            document.getElementById('clock').innerText = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

            // Auto-submit when time runs out
            if (timeLimit <= 0) {
                clearInterval(timerInterval);
                validateCode(); // Auto-submit on time limit
            }
        }
    }, 1000);
}

function pauseTimer() {
    isPaused = !isPaused;
    if (isPaused) {
        remainingTime = timeLimit; // Save remaining time
    }
}

function stopTimer() {
    clearInterval(timerInterval);
    document.getElementById('time').innerText = '00:00';
    document.getElementById('clock').innerText = '00:00';
    document.getElementById('timeInput').value = '';
    isPaused = false;
    remainingTime = 0;
}

// Function to trigger confetti effect
function triggerConfetti() {
    confetti({
        particleCount: 300, // Number of confetti particles
        spread: 300, // Spread of the confetti
        origin: { y: 0.6 }, // Origin of the confetti (bottom of the screen)
    });
}

// Code Validation Function
function validateCode() {
    const code = document.getElementById('codeInput').value;
    if (!code) {
        alert('Please write some code before submitting.');
        return;
    }

    // Get the current problem number from the URL
    const problemNumber = window.location.href.match(/problem(\d+)/)[1];

    // Validate code based on the problem number
    switch (problemNumber) {
        case '1': // Problem 1: Blink an LED
            if (code.includes('pinMode') && code.includes('digitalWrite') && code.includes('delay(1000)')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use pinMode, digitalWrite, and delay(1000).');
            }
            break;

        case '2': // Problem 2: Fade an LED
            if (code.includes('analogWrite') && code.includes('delay(30)')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use analogWrite and delay(30).');
            }
            break;

        case '3': // Problem 3: Button-Controlled LED
            if (code.includes('digitalRead') && code.includes('digitalWrite') && code.includes('if (digitalRead')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use digitalRead, digitalWrite, and an if condition.');
            }
            break;

        case '4': // Problem 4: Traffic Light System
            if (code.includes('digitalWrite') && code.includes('delay(1000)') && code.includes('HIGH') && code.includes('LOW')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use digitalWrite, delay, HIGH, and LOW.');
            }
            break;

        case '5': // Problem 5: Temperature Sensor
            if (code.includes('analogRead') && code.includes('Serial.begin') && code.includes('Serial.print')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use analogRead, Serial.begin, and Serial.print.');
            }
            break;

        case '6': // Problem 6: Servo Motor Control
            if (code.includes('Servo') && code.includes('attach') && code.includes('write')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use Servo, attach, and write.');
            }
            break;

        case '7': // Problem 7: Ultrasonic Distance Sensor
            if (code.includes('pulseIn') && code.includes('digitalWrite') && code.includes('digitalRead')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use pulseIn, digitalWrite, and digitalRead.');
            }
            break;

        case '8': // Problem 8: Light Sensor (LDR)
            if (code.includes('analogRead') && code.includes('Serial.begin') && code.includes('Serial.print')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use analogRead, Serial.begin, and Serial.print.');
            }
            break;

        case '9': // Problem 9: Potentiometer-Controlled LED Brightness
            if (code.includes('analogRead') && code.includes('analogWrite') && code.includes('map')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use analogRead, analogWrite, and map.');
            }
            break;

        case '10': // Problem 10: 7-Segment Display
            if (code.includes('digitalWrite') && code.includes('for') && code.includes('HIGH') && code.includes('LOW')) {
                alert('Code is valid!');
                triggerConfetti(); // Trigger confetti
            } else {
                alert('Code is invalid. Ensure you use digitalWrite, for loop, HIGH, and LOW.');
            }
            break;

        default:
            alert('Invalid problem number.');
            break;
    }
}