    
function simulateOverlay() {
    const output = document.getElementById('output');
    output.style.display = 'block';
}
const images = [
    "images-image-1",
    "images-image-2",
    "images-image-3",
];

let currentIndex = 0;

function changeImage() {
    const currentImage = document.getElementById(images[currentIndex]);
    currentImage.classList.remove('active');
    currentIndex = (currentIndex + 1) % images.length;
    const nextImage = document.getElementById(images[currentIndex]);
    nextImage.classList.add('active');
}

setInterval(changeImage, 5000);

//game JS
var rows = 3;
var columns = 3;
var currTile;
var otherTile;
var imgOrder = ["7", "4", "1", "9", "2", "6", "5", "3", "8"];
let turns=0;
window.onload = function() {
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            let tile = document.createElement("img");
            tile.id = r.toString() + "-" + c.toString();
            tile.src = imgOrder.shift() + ".jpg";
            tile.addEventListener("dragstart", dragStart);
            tile.addEventListener("dragover", dragOver);
            tile.addEventListener("dragenter", dragEnter);
            tile.addEventListener("dragleave", dragLeave);
            tile.addEventListener("drop", dragDrop);
            tile.addEventListener("dragend", dragEnd);
            document.getElementById("game").append(tile);
        }
    }
};

function dragStart(e) {
    currTile = this;
}

function dragOver(e) {
    e.preventDefault();
}

function dragEnter(e) {
    e.preventDefault();
}

function dragLeave(e) {}

function dragDrop(e) {
    otherTile = this;
    let currImg = currTile.src;
    let otherImg = otherTile.src;
    currTile.src = otherImg;
    otherTile.src = currImg;
    turns+=1;
    document.getElementById("turns").innerText=turns;
}

function dragEnd(e) {
    currTile = null;
    otherTile = null;
}
