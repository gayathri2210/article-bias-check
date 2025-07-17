document.addEventListener('DOMContentLoaded', () => {
    const analyzeButton = document.getElementById('analyzeButton');
    const articleText = document.getElementById('articleText');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const predictionResult = document.getElementById('predictionResult');
    
    // Left article elements
    const leftOutlet = document.getElementById('leftOutlet');
    const leftExcerpt = document.getElementById('leftExcerpt');
    const leftUrl = document.getElementById('leftUrl');

    // Right article elements
    const rightOutlet = document.getElementById('rightOutlet');
    const rightExcerpt = document.getElementById('rightExcerpt');
    const rightUrl = document.getElementById('rightUrl');

    analyzeButton.addEventListener('click', async () => {
        const text = articleText.value.trim();
        if (!text) {
            alert('Please paste some text to analyze.');
            return;
        }

        // Show loading spinner and hide previous results
        loadingDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');

        try {
            // The fetch API communicates with our Python backend
            const response = await fetch('http://127.0.0.1:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Hide loading and display results
            loadingDiv.classList.add('hidden');
            displayResults(data);
            resultsDiv.classList.remove('hidden');

        } catch (error) {
            loadingDiv.classList.add('hidden');
            alert(`An error occurred: ${error.message}`);
            console.error('Fetch error:', error);
        }
    });

    function displayResults(data) {
        // Display prediction
        const label = data.prediction.label.toUpperCase();
        const confidence = (data.prediction.confidence * 100).toFixed(1);
        predictionResult.textContent = `Predicted Bias: ${label} (Confidence: ${confidence}%)`;

        // Display left-leaning article
        const left = data.articles.left;
        if (left) {
            leftOutlet.textContent = `Outlet: ${left.outlet}`;
            leftExcerpt.textContent = `"${left.meaningful_phrase}"`;
            leftUrl.href = left.url;
        }

        // Display right-leaning article
        const right = data.articles.right;
        if (right) {
            rightOutlet.textContent = `Outlet: ${right.outlet}`;
            rightExcerpt.textContent = `"${right.meaningful_phrase}"`;
            rightUrl.href = right.url;
        }
    }
});
