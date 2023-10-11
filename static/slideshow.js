// -------------------------------------AYURVEDA------------------------------------
const images = ["static/image/D2.jpg", "static/image/D3.jpg", "static/image/D4.jpg", "static/image/D5.jpg", "static/image/D6.jpg"];
const slideshowElement = document.getElementById("slideshow");
let currentIndex = 0;
function updateSlideshow() {
    slideshowElement.src = images[currentIndex];
    currentIndex = (currentIndex + 1) % images.length;
}
updateSlideshow();
setInterval(updateSlideshow, 3000);

// -------------------------------------PLANT IDENTIFICATION------------------------------------
const images1 = ["static/image/E1.jpg", "static/image/E2.jpg", "static/image/E3.jpg", "static/image/E4.jpg", "static/image/E5.jpg"];
const slideshowElement1 = document.getElementById("slideshow1");
let currentIndex1 = 0;
function updateSlideshow1() {
    slideshowElement1.src = images1[currentIndex1];
    currentIndex1 = (currentIndex1 + 1) % images1.length;
}
updateSlideshow1();
setInterval(updateSlideshow1, 3000);

// -------------------------------------Check Symptoms---------------------------
const images2 = ["static/image/H1.jpg", "static/image/H2.jpg", "static/image/H3.jpg", "static/image/H4.jpg", "static/image/H5.jpg"];
const slideshowElement2 = document.getElementById("slideshow2");
let currentIndex2 = 0;
function updateSlideshow2() {
    slideshowElement2.src = images2[currentIndex2];
    currentIndex2 = (currentIndex2 + 1) % images2.length;
}
updateSlideshow2();
setInterval(updateSlideshow2, 3000);

// -------------------------------------Check Disease---------------------------
const images3 = ["static/image/G1.jpg", "static/image/G2.jpg", "static/image/G3.jpg", "static/image/G4.jpg", "static/image/G5.jpg"];
const slideshowElement3 = document.getElementById("slideshow3");
let currentIndex3 = 0;
function updateSlideshow3() {
    slideshowElement3.src = images3[currentIndex3];
    currentIndex3 = (currentIndex3 + 1) % images3.length;
}
updateSlideshow3();
setInterval(updateSlideshow3, 3000);

// -------------------------------------Check Bloodbank---------------------------
const images4 = ["static/image/F1.jpg", "static/image/F2.jpg", "static/image/F3.jpg", "static/image/F4.jpg", "static/image/F5.jpg", "static/image/F6.jpg"];
const slideshowElement4 = document.getElementById("slideshow4");
let currentIndex4 = 0;
function updateSlideshow4() {
    slideshowElement4.src = images4[currentIndex4];
    currentIndex4 = (currentIndex4 + 1) % images4.length;
}
updateSlideshow4();
setInterval(updateSlideshow4, 3000);