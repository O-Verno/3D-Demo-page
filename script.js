// ================================
// Image Overlay Simulation
// ================================

// When called, displays the element with ID 'output'
function simulateOverlay() {
    const output = document.getElementById('output');
    output.style.display = 'block';
}

// ================================
// Rotating Image Slideshow
// ================================

// Array of image element IDs
const images = [
    "images-image-1",
    "images-image-2",
    "images-image-3",
];

let currentIndex = 0;

// Function to cycle to the next image every 5 seconds
function changeImage() {
    // Remove 'active' class from current image
    const currentImage = document.getElementById(images[currentIndex]);
    currentImage.classList.remove('active');

    // Move to next image index (loop back to 0 at end)
    currentIndex = (currentIndex + 1) % images.length;

    // Add 'active' class to next image
    const nextImage = document.getElementById(images[currentIndex]);
    nextImage.classList.add('active');
}

// Automatically change image every 5 seconds
setInterval(changeImage, 5000);

// ================================
// Puzzle Game Logic
// ================================

var rows = 3;
var columns = 3;
var currTile;   // Currently dragged tile
var otherTile;  // Target tile where drop happens

// Initial scrambled image order
var imgOrder = ["7", "4", "1", "9", "2", "6", "5", "3", "8"];

let turns = 0; // Count the number of moves

// Create game board on page load
window.onload = function () {
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            let tile = document.createElement("img");

            // Assign ID to tile based on grid position (e.g., "0-1")
            tile.id = r.toString() + "-" + c.toString();

            // Assign image source based on shuffled array
            tile.src = imgOrder.shift() + ".jpg";

            // Add drag-and-drop event listeners
            tile.addEventListener("dragstart", dragStart);   // When drag starts
            tile.addEventListener("dragover", dragOver);     // While hovering
            tile.addEventListener("dragenter", dragEnter);   // When entering another tile
            tile.addEventListener("dragleave", dragLeave);   // (Optional)
            tile.addEventListener("drop", dragDrop);         // When drop happens
            tile.addEventListener("dragend", dragEnd);       // After drag ends

            // Add tile to the game board
            document.getElementById("game").append(tile);
        }
    }
};

// Called when dragging starts
function dragStart(e) {
    currTile = this;
}

// Allow dropping
function dragOver(e) {
    e.preventDefault();
}

// Allow entering another tile
function dragEnter(e) {
    e.preventDefault();
}

// Not used, but could be used to style tiles on leave
function dragLeave(e) {}

// Swap the two tiles (images) when drop occurs
function dragDrop(e) {
    otherTile = this;

    // Swap image sources between current and target tiles
    let currImg = currTile.src;
    let otherImg = otherTile.src;

    currTile.src = otherImg;
    otherTile.src = currImg;

    // Increment move count
    turns += 1;
    document.getElementById("turns").innerText = turns;
}

// Reset current tiles after drag ends
function dragEnd(e) {
    currTile = null;
    otherTile = null;
}
// ================================
// Navigation Indicator Logic
// ================================
document.addEventListener("DOMContentLoaded", function () {
    const listItems = document.querySelectorAll('.list');
    const indicator = document.querySelector('.indicater');

    listItems.forEach((item, index) => {
        const link = item.querySelector('a');
        const href = link.getAttribute('href');

        // Gør dette element aktivt hvis URL matcher
        if (window.location.href.includes(href)) {
            item.classList.add('active');

            // Flyt indikator med samme afstand som i CSS (85px)
            if (indicator) {
                indicator.style.transform = `translateX(${85 * index}px)`;
            }
        } else {
            item.classList.remove('active');
        }
    });
});

