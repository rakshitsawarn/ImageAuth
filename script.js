// Global variables
let selectedFile = null;
let analysisStartTime = null;

// DOM Elements
const uploadContainer = document.getElementById('uploadContainer');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const imageInfo = document.getElementById('imageInfo');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');

// Initialize event listeners
function initializeEventListeners() {
    // Browse button click
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Upload container click
    uploadContainer.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop events
    uploadContainer.addEventListener('dragover', handleDragOver);
    uploadContainer.addEventListener('dragleave', handleDragLeave);
    uploadContainer.addEventListener('drop', handleDrop);

    // Remove button
    removeBtn.addEventListener('click', resetUpload);

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);

    // New analysis button
    newAnalysisBtn.addEventListener('click', resetAll);

    // Prevent default drag behaviors on document
    ['dragover', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadContainer.classList.add('drag-over');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadContainer.classList.remove('drag-over');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadContainer.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file select
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Validate and handle file
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, PNG, or WEBP)');
        return;
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
        alert('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    displayPreview(file);
}

// Display image preview
function displayPreview(file) {
    const reader = new FileReader();

    reader.onload = function (e) {
        previewImage.src = e.target.result;

        // Get image dimensions
        const img = new Image();
        img.onload = function () {
            displayImageInfo(file, img.width, img.height);
        };
        img.src = e.target.result;
    };

    reader.readAsDataURL(file);

    // Hide upload section and show preview
    uploadContainer.style.display = 'none';
    previewSection.style.display = 'block';
    resultsSection.style.display = 'none';
}

// Display image information
function displayImageInfo(file, width, height) {
    const fileSize = formatFileSize(file.size);
    const fileType = file.type.split('/')[1].toUpperCase();

    imageInfo.innerHTML = `
        <div>
            <span>File Name</span>
            <span>${file.name}</span>
        </div>
        <div>
            <span>File Type</span>
            <span>${fileType}</span>
        </div>
        <div>
            <span>File Size</span>
            <span>${fileSize}</span>
        </div>
        <div>
            <span>Dimensions</span>
            <span>${width} × ${height} px</span>
        </div>
    `;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// Reset upload
function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    imageInfo.innerHTML = '';
    uploadContainer.style.display = 'block';
    previewSection.style.display = 'none';
}

// Reset all
function resetAll() {
    resetUpload();
    resultsSection.style.display = 'none';
}

// Analyze image
async function analyzeImage() {
    if (!selectedFile) return;

    analysisStartTime = Date.now();
    showLoading();

    try {
        const result = await performMLAnalysis(selectedFile);

        hideLoading();
        displayResults(result);
    } catch (error) {
        hideLoading();
        alert('An error occurred during analysis. See console for details.');
        console.error('Analysis error:', error);
    }
}

// Perform ML analysis (calls backend)
async function performMLAnalysis(file) {
    const formData = new FormData();
    formData.append("file", file);

    // Adjust URL if your backend is hosted elsewhere
    const endpoint = "http://localhost:8000/predict";

    const res = await fetch(endpoint, {
        method: "POST",
        body: formData
    });

    if (!res.ok) {
        // attempt to read text body for better error message
        let text = await res.text();
        throw new Error('Server responded with ' + res.status + ': ' + text);
    }

    const data = await res.json();

    // handle server-side error wrapper
    if (data.error) throw new Error(data.error + (data.details ? (': ' + data.details) : ''));

    // prefer explicit probabilities if provided by server
    return {
        isReal: data.isReal,
        confidence: data.confidence ?? (data.prob_real ?? 0),
        prob_real: data.prob_real ?? (1 - (data.prob_fake ?? 0)),
        prob_fake: data.prob_fake ?? (1 - (data.prob_real ?? 0)),
        imageWidth: data.imageWidth ?? 0,
        imageHeight: data.imageHeight ?? 0,
        fileSize: file.size
    };
}

// Display results
function displayResults(result) {
    const analysisTime = ((Date.now() - analysisStartTime) / 1000).toFixed(2);
    const confidencePercent = (result.confidence * 100).toFixed(1);

    // Determine confidence level (UI only)
    let confidenceLevel = 'high';
    if (result.confidence < 0.7) confidenceLevel = 'low';
    else if (result.confidence < 0.85) confidenceLevel = 'medium';

    // Update confidence badge (match CSS class 'confidence-indicator')
    const confidenceBadge = document.getElementById('confidenceBadge');
    confidenceBadge.className = `confidence-indicator ${confidenceLevel}`;
    confidenceBadge.textContent = `${confidencePercent}% Confidence`;

    // Update result card using explicit probabilities to avoid label ambiguity
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultDescription = document.getElementById('resultDescription');

    // Decide final verdict using prob_real vs prob_fake
    if (result.prob_real >= result.prob_fake) {
        resultIcon.className = 'result-icon real';
        resultTitle.className = 'result-title real';
        resultTitle.textContent = 'Authentic Image';
        resultDescription.textContent = `Model: ${ (result.prob_real*100).toFixed(1) }% real, ${ (result.prob_fake*100).toFixed(1) }% fake.`;
    } else {
        resultIcon.className = 'result-icon fake';
        resultTitle.className = 'result-title fake';
        resultTitle.textContent = 'AI-Generated Image';
        resultDescription.textContent = `Model: ${ (result.prob_fake*100).toFixed(1) }% fake, ${ (result.prob_real*100).toFixed(1) }% real.`;
    }

    // Update progress bar (match CSS .confidence-fill)
    const progressFill = document.getElementById('progressFill');
    progressFill.className = `confidence-fill ${confidenceLevel}`;
    progressFill.style.width = `${confidencePercent}%`;

    // Update detail values
    document.getElementById('confidenceValue').textContent = `${confidencePercent}%`;
    document.getElementById('analysisTime').textContent = `${analysisTime}s`;
    document.getElementById('imageDimensions').textContent = `${result.imageWidth} × ${result.imageHeight} px`;
    document.getElementById('fileSize').textContent = formatFileSize(result.fileSize);

    // Show results section
    previewSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Show loading overlay
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

// Hide loading overlay
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initializeEventListeners);
