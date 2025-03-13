document.getElementById("predictionForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const inputValues = [
        formData.get("Pregnancies"),
        formData.get("Glucose"),
        formData.get("BloodPressure"),
        formData.get("SkinThickness"),
        formData.get("Insulin"),
        formData.get("BMI"),
        formData.get("DiabetesPedigreeFunction"),
        formData.get("Age")
    ].map(Number);
    const modelChoice = formData.get("model");

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: inputValues, model: modelChoice })
        });
        const result = await response.json();
        document.getElementById("result").innerText = `Prediction: ${result.prediction} (Probability: ${result.probability.toFixed(2)}) using ${result.model}`;
    } catch (error) {
        document.getElementById("result").innerText = `Error: ${error.message}`;
    }
});