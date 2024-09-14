$(document).ready(function () {
    $("#register_button a").click(function (e) {
      e.preventDefault();
  
      var username = $(".input_text[type='text']").val();
      var password = $(".input_text[type='password']").val();
  
      if (username && password) {
        $.post("/register", { username: username, password: password }, function (data) {
          if (data === 'Registration successful. You can now log in.') {
            alert('Registration successful!');
            window.location.href = '/'; // Redirect to login page
          }
           else {
            alert('Registration failed. Please try again.');
           }
        });
      } else {
        alert('Please enter both username and password.');
      }
    });
  });