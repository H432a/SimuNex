document.addEventListener("DOMContentLoaded", () => {
    const projectListContainer = document.getElementById("project-list");

    // List of projects
    const projects = [
        { id: 1, title: "Arduino LED Blinking" },
        { id: 2, title: "Servo Motor Testing" },
       
    ];

    // Dynamically create buttons for each project
    projects.forEach(project => {
        const projectDiv = document.createElement("div");
        projectDiv.classList.add("project-item");

        const title = document.createElement("h3");
        title.innerText = project.title;
        projectDiv.appendChild(title);

        const button = document.createElement("button");
        button.innerText = "View Experiment";
        button.addEventListener("click", () => {
            window.location.href = `labpro?id=${project.id}`;
        });

        projectDiv.appendChild(button);
        projectListContainer.appendChild(projectDiv);
    });
});
