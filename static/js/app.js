// This file contains JavaScript code for handling client-side interactions, such as form submissions and displaying loading spinners.

document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("loan-form");
    const loader = document.getElementById("page-loader");

    if (form) {
        form.addEventListener("submit", function() {
            if (loader) {
                loader.classList.remove("d-none");
            }
        });
    }
});

// Save this file as /LoanApprovalPrediction/LoanApprovalPrediction/static/js/app.js. 
// Next, please let me know what file you would like to generate next.