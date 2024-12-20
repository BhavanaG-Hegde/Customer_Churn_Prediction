document.getElementById("churnForm").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent default form submission behavior

    // Extract form data
    const age = document.getElementById("age").value;
    const balance = document.getElementById("balance").value;
    const gender = document.getElementById("gender").value;
    const active = document.getElementById("active").value;
    const numProducts = document.getElementById("num-products").value;

    try {
        // Send data to the backend
        const response = await fetch("/churn", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",  // Ensure JSON is sent
            },
            body: JSON.stringify({
                age: age,
                balance: balance,
                gender: gender,
                active: active,
                numProducts: numProducts,
            }),
        });

        if (!response.ok) {
            throw new Error("Failed to fetch churn prediction.");
        }

        const data = await response.json();

        // Display the churn result
        if (data.churn) {
            document.getElementById("churn").textContent = data.churn.toFixed(2);
            document.querySelector(".result-section").classList.remove("hidden");
        } else {
            alert("Error: " + data.error);
        }
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while fetching churn prediction.");
    }
});
