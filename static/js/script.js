// Main website JavaScript

document.addEventListener("DOMContentLoaded", function () {

    const predictionForm = document.getElementById("predictionForm");

    if (predictionForm) {

        predictionForm.addEventListener("submit", async function (event) {

            event.preventDefault();

            const formData = new FormData(predictionForm);

            const data = {};

            formData.forEach((value, key) => {

                if (
                    key === "Inventory Level" ||
                    key === "Units Ordered" ||
                    key === "Price" ||
                    key === "Discount" ||
                    key === "Holiday/Promotion" ||
                    key === "Competitor Pricing"
                ) {
                    data[key] = Number(value);
                } else {
                    data[key] = value;
                }

            });

            try {

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                document.getElementById("predictionValue").textContent =
                    result.predicted_demand;

                document
                    .getElementById("predictionResult")
                    .classList.remove("hidden");

            } catch (error) {

                alert("Prediction failed. Please try again.");

            }

        });

    }

});


function calculateReorder() {

    const currentStock =
        Number(document.getElementById("currentStock").value);

    const predictedDemand =
        Number(document.getElementById("predictedDemand").value);

    const leadTime =
        Number(document.getElementById("leadTime").value);

    const safetyStock =
        Number(document.getElementById("safetyStock").value);

    const dailyDemand = predictedDemand / 30;

    const reorderPoint =
        (dailyDemand * leadTime) + safetyStock;

    const orderQuantity =
        Math.max(0, Math.ceil(reorderPoint - currentStock));

    const status =
        document.getElementById("reorderStatus");

    const quantity =
        document.getElementById("orderQuantity");

    if (orderQuantity > 0) {

        status.textContent =
            "Reorder is recommended for this product.";

        quantity.textContent =
            "Recommended Order Quantity: " + orderQuantity;

    } else {

        status.textContent =
            "Current inventory is sufficient. No reorder required.";

        quantity.textContent =
            "Recommended Order Quantity: 0";

    }

}


function searchInventory() {

    const input =
        document.getElementById("inventorySearch");

    const filter =
        input.value.toLowerCase();

    const table =
        document.getElementById("inventoryTable");

    const rows =
        table.getElementsByTagName("tr");

    for (let i = 1; i < rows.length; i++) {

        const text =
            rows[i].textContent.toLowerCase();

        if (text.includes(filter)) {
            rows[i].style.display = "";
        } else {
            rows[i].style.display = "none";
        }

    }

}