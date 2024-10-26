document.addEventListener("DOMContentLoaded", function () {
    const themeSwitch = document.getElementById("themeSwitch");
    const body = document.body;

    if (localStorage.getItem("theme") === "dark") {
        body.classList.add("dark-mode");
        themeSwitch.checked = true;
    }

    themeSwitch.addEventListener("change", function () {
        if (themeSwitch.checked) {
            body.classList.add("dark-mode");
            localStorage.setItem("theme", "dark");
        } else {
            body.classList.remove("dark-mode");
            localStorage.setItem("theme", "light");
        }
    });
});

document.getElementById("iconUpload").addEventListener("click", function () {
    document.getElementById("imageUpload").click();
});

document.getElementById("imageForm").addEventListener("submit", function (event) {
    event.preventDefault();

    const fileInput = document.getElementById("imageUpload");
    const textInput = document.getElementById("textInput");
    const file = fileInput.files[0];
    const text = textInput.value;
    const csrfmiddlewaretoken = document.getElementsByName("csrfmiddlewaretoken")[0].value;

    const formData = new FormData();
    formData.append('photo', file);
    formData.append('query_text', text);

    axios.post(
        '/classify-image/',
        formData,
        { headers: { "X-CSRFToken": csrfmiddlewaretoken } }
    )
        .then(response => {
            const data = response.data;
            if (data.success) {
                if (data.uploaded_file_url && data.prediction) {
                    addImageToChat('User', data.uploaded_file_url);
                    addMessageToChat('Bot', `Predicted Result: <strong>${data.prediction}</strong>`);
                } else if (data.query_text && data.response_text) {
                    addMessageToChat('User', data.query_text);
                    addMessageToChat('Bot', `<strong>${data.response_text}</strong>`);
                }
            } else {
                addMessageToChat('Bot', data.message || "An error occurred.");
            }
        })
        .catch(error => {
            console.error('Error with Axios operation:', error);
            addMessageToChat('Bot', "Error occurred during classification.");
        })
});

function addImageToChat(sender, imageUrl) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('image-msg');
    messageDiv.innerHTML = `<img src="${imageUrl}" alt="uploaded image" style="max-width: 300px;">`;
    document.getElementById('chatWindow').appendChild(messageDiv);
    document.getElementById('chatWindow').scrollTop = document.getElementById('chatWindow').scrollHeight;
}

function addMessageToChat(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add(sender === 'User' ? 'user-msg' : 'bot-msg');
    messageDiv.innerHTML = `${message}`;
    document.getElementById('chatWindow').appendChild(messageDiv);
    document.getElementById('chatWindow').scrollTop = document.getElementById('chatWindow').scrollHeight;
}