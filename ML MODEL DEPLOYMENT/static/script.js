// script.js
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const resultText = document.getElementById('result-text');
    const loader = document.getElementById('loader');
    const submitBtn = document.getElementById('submit-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent the default form submission (page reload)

        // Show loader, hide previous result, and disable the button
        loader.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        submitBtn.disabled = true;

        // 1. Gather data from the form
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            // Convert numeric fields to the correct type
            const numericFields = ['no of adults', 'no of children', 'lead time'];
            if (numericFields.includes(key)) {
                data[key] = parseInt(value, 10);
            } else {
                data[key] = value;
            }
        });

        // Extract the algorithm choice and remove it from the input data
        const algorithm = data.algorithm;
        delete data.algorithm;

        // 2. Format the data into the structure the Flask API expects
        const apiPayload = {
            algorithm: algorithm,
            data: [data]
        };

        try {
            // 3. Send the data to the server
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(apiPayload),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            // 4. Display the result
            displayResult(result.prediction[0]);

        } catch (error) {
            console.error('Error:', error);
            resultText.textContent = 'An error occurred while contacting the server. Please try again.';
            resultContainer.className = 'danger'; // Set class for error styling
            resultContainer.classList.remove('hidden');
        } finally {
            // Hide the loader and re-enable the button
            loader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    function displayResult(prediction) {
        // Based on the LabelEncoder, 1 is usually the positive class
        if (prediction === 1) {
            resultText.textContent = 'Booking Confirmed (Not Canceled)';
            resultContainer.className = 'success';
        } else {
            resultText.textContent = 'Booking Canceled';
            resultContainer.className = 'danger';
        }
        resultContainer.classList.remove('hidden');
    }
});