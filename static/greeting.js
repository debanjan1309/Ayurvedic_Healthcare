$(document).ready(function () {
    // Get the stored username from session storage
    var username = sessionStorage.getItem('username');

    // Get the greeting element by its ID
    var greetingElement = $("#greeting");

    // Update the greeting message if the username is available
    if (username) {
        greetingElement.text('Hello, ' + username);
    }
});
