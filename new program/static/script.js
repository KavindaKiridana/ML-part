document.addEventListener('DOMContentLoaded', function () {
    // Check if we have a visualization
    checkVisualization();

    // Train form submission
    document.getElementById('trainForm').addEventListener('submit', function (e) {
        e.preventDefault();

        const fileInput = document.getElementById('dataFile');
        const file = fileInput.files[0];

        if (!file) {
            showResult('trainResult', false, 'Please select a data file.');
            return;
        }

        // Disable button and show loading state
        const trainButton = document.getElementById('trainButton');
        trainButton.disabled = true;
        trainButton.textContent = 'Training...';

        const formData = new FormData();
        formData.append('file', file);

        fetch('/train', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showResult('trainResult', true, 'Model trained successfully!');
                    checkVisualization();
                } else {
                    showResult('trainResult', false, `Error: ${data.error}`);
                }
            })
            .catch(error => {
                showResult('trainResult', false, `Error: ${error.message}`);
            })
            .finally(() => {
                // Re-enable button
                trainButton.disabled = false;
                trainButton.textContent = 'Train Model';
            });
    });

    // Predict form submission
    document.getElementById('predictForm').addEventListener('submit', function (e) {
        e.preventDefault();

        const targetWeek = document.getElementById('targetWeek').value;
        const temperature = document.getElementById('temperature').value || '24.5';
        const humidity = document.getElementById('humidity').value || '75.0';
        const lightLevel = document.getElementById('lightLevel').value || '30000';

        // Disable button and show loading state
        const predictButton = document.getElementById('predictButton');
        predictButton.disabled = true;
        predictButton.textContent = 'Predicting...';

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                targetWeek: targetWeek,
                temperature: temperature,
                humidity: humidity,
                lightLevel: lightLevel
            })
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const resultHTML = `
                    <h3>Prediction Results for Week ${targetWeek}:</h3>
                    <p><strong>Predicted Vine Length:</strong> ${data.predictions.length} meters</p>
                    <p><strong>Predicted Vine Leaves:</strong> ${data.predictions.leaves}</p>
                `;
                    document.getElementById('predictResult').innerHTML = resultHTML;
                    document.getElementById('predictResult').className = 'result-box success';
                    document.getElementById('predictResult').style.display = 'block';
                } else {
                    showResult('predictResult', false, `Error: ${data.error}`);
                }
            })
            .catch(error => {
                showResult('predictResult', false, `Error: ${error.message}`);
            })
            .finally(() => {
                // Re-enable button
                predictButton.disabled = false;
                predictButton.textContent = 'Predict Growth';
            });
    });

    // Function to show result message
    function showResult(elementId, isSuccess, message) {
        const element = document.getElementById(elementId);
        element.textContent = message;
        element.className = isSuccess ? 'result-box success' : 'result-box error';
        element.style.display = 'block';
    }

    // Function to check if visualization exists
    function checkVisualization() {
        const img = document.getElementById('growthChart');
        const noViz = document.getElementById('noVisualization');

        // Add timestamp to prevent caching
        const timestamp = new Date().getTime();
        fetch(`static/vanilla_growth_trends.png?t=${timestamp}`, { method: 'HEAD' })
            .then(response => {
                if (response.ok) {
                    img.src = `static/vanilla_growth_trends.png?t=${timestamp}`;
                    img.style.display = 'block';
                    noViz.style.display = 'none';
                } else {
                    img.style.display = 'none';
                    noViz.style.display = 'block';
                }
            })
            .catch(() => {
                img.style.display = 'none';
                noViz.style.display = 'block';
            });
    }
});