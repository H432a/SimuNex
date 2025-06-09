document.addEventListener("DOMContentLoaded", () => {
    // Get project ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    const projectId = urlParams.get("id");

    // Map project IDs to their respective Wokwi URLs
    const wokwiProjects = {
        "1": "https://wokwi.com/projects/426050699253933057", // LED Blinking
        "2": "https://wokwi.com/projects/426105067857656833"  // Servo Motor Testing (Replace with actual URL)
    };

    // Select iframe and button
    const iframe = document.getElementById("wokwi-frame");
    const loadProjectBtn = document.getElementById("loadProject");

    if (wokwiProjects[projectId]) {
        // Set initial Wokwi project URL
        iframe.src = wokwiProjects[projectId];

        // Load project when button is clicked
        loadProjectBtn.addEventListener("click", () => {
            iframe.src = wokwiProjects[projectId];
        });
    } else {
        // If project ID is invalid, show an error
        document.querySelector(".container").innerHTML = "<h1>Project Not Found</h1>";
    }
});
