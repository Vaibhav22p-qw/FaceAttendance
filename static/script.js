document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('videoElement');

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                videoElement.srcObject = stream;
            })
            .catch((err) => {
                console.error('Error accessing the webcam: ', err);
            });
    } else {
        console.error('getUserMedia is not supported in this browser.');
    }
});
