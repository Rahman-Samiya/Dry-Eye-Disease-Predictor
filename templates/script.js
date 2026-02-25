document.getElementById('eyeForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);

  // Convert form fields to a plain object matching the column names
  const payload = {};
  for (let [key, value] of formData.entries()) {
    payload[key] = value;
  }

  // show loading state
  const resultBox = document.getElementById('resultBox');
  const title = document.getElementById('resultTitle');
  const text = document.getElementById('resultText');
  title.textContent = 'Checking...';
  text.textContent = 'Please wait.';
  resultBox.hidden = false;

  try {
    const resp = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!resp.ok) throw new Error('Server error: ' + resp.statusText);
    const data = await resp.json();

    if (data.success) {
      title.textContent = data.prediction === 1 ? 'Likely Dry Eye Disease' : 'Unlikely Dry Eye Disease';
      text.innerHTML = `<strong>Probability (approx.):</strong> ${Math.round((data.probability || 0)*100)}% <br>
                        <strong>Model note:</strong> ${data.message || 'Prediction saved to DB.'}`;
    } else {
      title.textContent = 'Error';
      text.textContent = data.message || 'Prediction failed';
    }
  } catch (err) {
    title.textContent = 'Error';
    text.textContent = err.message;
  }
});
