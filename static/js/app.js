// Avocado Price Prediction - Enhanced UI with Anime.js
document.addEventListener('DOMContentLoaded', function() {
    console.log('🥑 Avocado Price Prediction App Loaded');
    
    // Initialize all animations and interactions
    initializeAnimations();
    initializeFormInteractions();
    initializeFormValidation();
    
    // Show initial load animations
    showPageLoadAnimations();
});

// Initialize entrance animations
function initializeAnimations() {
    // Hide elements initially for animation
    const elementsToAnimate = document.querySelectorAll('.fade-in, .slide-in-left, .slide-in-right, .scale-in');
    elementsToAnimate.forEach(el => {
        el.style.opacity = '0';
    });
}

// Page load animations sequence
function showPageLoadAnimations() {
    // Header animation
    anime({
        targets: '.header',
        opacity: [0, 1],
        translateY: [-50, 0],
        duration: 800,
        easing: 'easeOutCubic'
    });

    // Header text animations
    anime({
        targets: '.header h1',
        opacity: [0, 1],
        translateY: [-30, 0],
        duration: 1000,
        delay: 300,
        easing: 'easeOutCubic'
    });

    anime({
        targets: '.header p',
        opacity: [0, 1],
        translateY: [20, 0],
        duration: 800,
        delay: 500,
        easing: 'easeOutCubic'
    });

    // Form container animation
    anime({
        targets: '.prediction-form',
        opacity: [0, 1],
        scale: [0.9, 1],
        duration: 1000,
        delay: 600,
        easing: 'easeOutBack'
    });

    // Form elements stagger animation
    anime({
        targets: '.form-group',
        opacity: [0, 1],
        translateY: [30, 0],
        duration: 600,
        delay: anime.stagger(100, {start: 800}),
        easing: 'easeOutCubic'
    });

    // Button animation
    anime({
        targets: '.predict-btn',
        opacity: [0, 1],
        scale: [0.8, 1],
        duration: 800,
        delay: 1200,
        easing: 'easeOutElastic(1, .8)'
    });

    // Info cards animation
    if (document.querySelector('.info-cards')) {
        anime({
            targets: '.info-card',
            opacity: [0, 1],
            translateY: [50, 0],
            duration: 800,
            delay: anime.stagger(150, {start: 1000}),
            easing: 'easeOutCubic'
        });
    }
}

// Form interactions and micro-animations
function initializeFormInteractions() {
    const formInputs = document.querySelectorAll('.form-control, .form-select');
    const predictBtn = document.querySelector('.predict-btn');

    // Input focus animations
    formInputs.forEach(input => {
        input.addEventListener('focus', function() {
            anime({
                targets: this.parentElement.querySelector('.form-label'),
                color: '#4CAF50',
                scale: [1, 1.05],
                duration: 300,
                easing: 'easeOutCubic'
            });

            // Add floating effect
            anime({
                targets: this,
                boxShadow: ['0 2px 10px rgba(76, 175, 80, 0.1)', '0 8px 25px rgba(76, 175, 80, 0.25)'],
                duration: 300,
                easing: 'easeOutCubic'
            });
        });

        input.addEventListener('blur', function() {
            anime({
                targets: this.parentElement.querySelector('.form-label'),
                color: '#2E7D32',
                scale: [1.05, 1],
                duration: 300,
                easing: 'easeOutCubic'
            });

            anime({
                targets: this,
                boxShadow: ['0 8px 25px rgba(76, 175, 80, 0.25)', '0 2px 10px rgba(76, 175, 80, 0.1)'],
                duration: 300,
                easing: 'easeOutCubic'
            });
        });

        // Input change animation
        input.addEventListener('input', function() {
            if (this.value) {
                anime({
                    targets: this,
                    borderColor: ['#8BC34A', '#4CAF50', '#8BC34A'],
                    duration: 400,
                    easing: 'easeInOutQuad'
                });
            }
        });
    });

    // Button hover animations
    if (predictBtn) {
        predictBtn.addEventListener('mouseenter', function() {
            anime({
                targets: this,
                scale: 1.02,
                duration: 300,
                easing: 'easeOutCubic'
            });
        });

        predictBtn.addEventListener('mouseleave', function() {
            if (!this.classList.contains('loading')) {
                anime({
                    targets: this,
                    scale: 1,
                    duration: 300,
                    easing: 'easeOutCubic'
                });
            }
        });
    }
}

// Form validation with animations
function initializeFormValidation() {
    const form = document.querySelector('form');
    const predictBtn = document.querySelector('.predict-btn');

    if (!form || !predictBtn) return;

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate form
        const isValid = validateForm();
        
        if (isValid) {
            showLoadingState();
            submitForm();
        } else {
            showValidationError();
        }
    });
}

function validateForm() {
    const requiredFields = document.querySelectorAll('[required]');
    let isValid = true;
    const invalidFields = [];

    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            isValid = false;
            invalidFields.push(field);
            
            // Add error styling
            field.style.borderColor = '#f44336';
            field.style.boxShadow = '0 0 0 3px rgba(244, 67, 54, 0.2)';
            
            // Shake animation
            anime({
                targets: field,
                translateX: [0, -10, 10, -10, 10, 0],
                duration: 600,
                easing: 'easeInOutCubic'
            });

            // Remove error styling after animation
            setTimeout(() => {
                field.style.borderColor = '';
                field.style.boxShadow = '';
            }, 3000);
        }
    });

    return isValid;
}

function showValidationError() {
    // Create or show validation message
    let errorMsg = document.querySelector('.validation-error');
    if (!errorMsg) {
        errorMsg = document.createElement('div');
        errorMsg.className = 'validation-error';
        errorMsg.style.cssText = `
            background: linear-gradient(135deg, #FFCDD2 0%, #FFEBEE 100%);
            color: #C62828;
            padding: 1rem;
            border-radius: 12px;
            margin-top: 1rem;
            text-align: center;
            border: 2px solid #F44336;
            font-weight: 600;
        `;
        errorMsg.textContent = 'Please fill in all required fields';
        document.querySelector('.prediction-form').appendChild(errorMsg);
    }

    // Animate error message
    anime({
        targets: errorMsg,
        opacity: [0, 1],
        translateY: [-20, 0],
        scale: [0.9, 1],
        duration: 500,
        easing: 'easeOutBack'
    });

    // Auto-hide after 4 seconds
    setTimeout(() => {
        anime({
            targets: errorMsg,
            opacity: 0,
            translateY: -20,
            duration: 300,
            complete: () => errorMsg.remove()
        });
    }, 4000);
}

function showLoadingState() {
    const predictBtn = document.querySelector('.predict-btn');
    const loadingOverlay = createLoadingOverlay();
    
    // Button loading state
    predictBtn.classList.add('loading');
    predictBtn.disabled = true;
    
    const originalText = predictBtn.innerHTML;
    predictBtn.innerHTML = 'Analyzing Data...';

    // Show loading overlay
    document.body.appendChild(loadingOverlay);
    
    // Animate loading overlay
    anime({
        targets: loadingOverlay,
        opacity: [0, 1],
        duration: 300,
        easing: 'easeOutCubic'
    });

    // Animate loading spinner
    anime({
        targets: loadingOverlay.querySelector('.loading-spinner'),
        rotate: '1turn',
        duration: 2000,
        loop: true,
        easing: 'linear'
    });

    // Store original state for cleanup
    predictBtn.dataset.originalText = originalText;
}

function createLoadingOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay active';
    
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    
    const text = document.createElement('div');
    text.style.cssText = 'color: white; font-size: 1.2rem; font-weight: 600; margin-top: 2rem;';
    text.textContent = 'Predicting avocado prices...';
    
    const container = document.createElement('div');
    container.style.textAlign = 'center';
    container.appendChild(spinner);
    container.appendChild(text);
    
    overlay.appendChild(container);
    return overlay;
}

function hideLoadingState() {
    const predictBtn = document.querySelector('.predict-btn');
    const loadingOverlay = document.querySelector('.loading-overlay');
    
    // Remove button loading state
    predictBtn.classList.remove('loading');
    predictBtn.disabled = false;
    predictBtn.innerHTML = predictBtn.dataset.originalText || 'Predict Price';
    
    // Hide loading overlay
    if (loadingOverlay) {
        anime({
            targets: loadingOverlay,
            opacity: 0,
            duration: 300,
            complete: () => loadingOverlay.remove()
        });
    }
}

function submitForm() {
    const form = document.querySelector('form');
    const formData = new FormData(form);
    
    // Simulate API call with fetch
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(html => {
        // Parse the response to get the prediction result
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const resultElement = doc.querySelector('.result-text');
        const predictionText = resultElement ? resultElement.textContent : 'Error occurred while predicting price. Please check your input.';
        
        hideLoadingState();
        showResult(predictionText);
    })
    .catch(error => {
        console.error('Error:', error);
        hideLoadingState();
        showResult('Error occurred while predicting price. Please try again.');
    });
}

function showResult(predictionText) {
    // Remove existing result
    const existingResult = document.querySelector('.result-container');
    if (existingResult) {
        existingResult.remove();
    }

    // Create result container
    const resultContainer = document.createElement('div');
    resultContainer.className = 'result-container';
    
    const isError = predictionText.includes('Error') || predictionText.includes('not recognized');
    if (isError) {
        resultContainer.classList.add('result-error');
    } else {
        resultContainer.classList.add('result-success');
    }

    const resultContent = document.createElement('div');
    resultContent.className = 'result-content';
    
    const resultText = document.createElement('div');
    resultText.className = 'result-text';
    resultText.textContent = predictionText;
    
    const resultSubtitle = document.createElement('div');
    resultSubtitle.className = 'result-subtitle';
    resultSubtitle.textContent = isError ? 'Please check your input and try again' : 'Based on your input parameters';
    
    resultContent.appendChild(resultText);
    resultContent.appendChild(resultSubtitle);
    resultContainer.appendChild(resultContent);
    
    // Add to form
    document.querySelector('.prediction-form').appendChild(resultContainer);
    
    // Animate result appearance
    setTimeout(() => {
        resultContainer.classList.add('show');
        
        // Additional entrance animation
        anime({
            targets: resultContainer,
            scale: [0.8, 1],
            duration: 600,
            easing: 'easeOutBack'
        });

        // Animate result text
        anime({
            targets: resultText,
            opacity: [0, 1],
            translateY: [20, 0],
            duration: 800,
            delay: 200,
            easing: 'easeOutCubic'
        });

        anime({
            targets: resultSubtitle,
            opacity: [0, 1],
            translateY: [20, 0],
            duration: 600,
            delay: 400,
            easing: 'easeOutCubic'
        });

        // Celebration animation for success
        if (!isError) {
            createCelebrationEffect();
        }
    }, 100);
}

function createCelebrationEffect() {
    // Create floating avocado emojis
    const avocados = ['🥑', '🌿', '💚', '✨'];
    
    for (let i = 0; i < 8; i++) {
        setTimeout(() => {
            const emoji = document.createElement('div');
            emoji.textContent = avocados[Math.floor(Math.random() * avocados.length)];
            emoji.style.cssText = `
                position: fixed;
                font-size: 2rem;
                pointer-events: none;
                z-index: 1000;
                left: ${Math.random() * window.innerWidth}px;
                top: ${window.innerHeight}px;
            `;
            document.body.appendChild(emoji);
            
            // Animate emoji floating up
            anime({
                targets: emoji,
                translateY: -window.innerHeight - 100,
                translateX: (Math.random() - 0.5) * 200,
                rotate: Math.random() * 360,
                opacity: [1, 0],
                duration: 3000,
                easing: 'easeOutCubic',
                complete: () => emoji.remove()
            });
        }, i * 200);
    }
}

// Add some interactive hover effects to info cards if they exist
function initializeInfoCardAnimations() {
    const infoCards = document.querySelectorAll('.info-card');
    
    infoCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            anime({
                targets: this,
                scale: 1.05,
                rotateY: 5,
                duration: 400,
                easing: 'easeOutCubic'
            });
        });
        
        card.addEventListener('mouseleave', function() {
            anime({
                targets: this,
                scale: 1,
                rotateY: 0,
                duration: 400,
                easing: 'easeOutCubic'
            });
        });
    });
}

// Initialize info card animations when they're added
setTimeout(initializeInfoCardAnimations, 2000);

// Add some background animation effects
function createBackgroundAnimations() {
    // Create floating elements
    const numberOfElements = 5;
    
    for (let i = 0; i < numberOfElements; i++) {
        const element = document.createElement('div');
        element.style.cssText = `
            position: fixed;
            width: ${20 + Math.random() * 40}px;
            height: ${20 + Math.random() * 40}px;
            background: radial-gradient(circle, rgba(139, 195, 74, 0.1) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
            z-index: -1;
        `;
        
        document.body.appendChild(element);
        
        // Animate floating elements
        anime({
            targets: element,
            translateX: [
                Math.random() * window.innerWidth,
                Math.random() * window.innerWidth
            ],
            translateY: [
                Math.random() * window.innerHeight,
                Math.random() * window.innerHeight
            ],
            duration: Math.random() * 20000 + 10000,
            loop: true,
            direction: 'alternate',
            easing: 'easeInOutSine'
        });
    }
}

// Initialize background animations
setTimeout(createBackgroundAnimations, 1000);

// Add scroll-triggered animations for better UX
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '50px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                
                if (element.classList.contains('fade-in')) {
                    anime({
                        targets: element,
                        opacity: [0, 1],
                        translateY: [30, 0],
                        duration: 800,
                        easing: 'easeOutCubic'
                    });
                }
                
                observer.unobserve(element);
            }
        });
    }, observerOptions);

    // Observe elements that should animate on scroll
    document.querySelectorAll('.fade-in').forEach(el => {
        observer.observe(el);
    });
}

// Initialize scroll animations
setTimeout(initializeScrollAnimations, 500);

console.log('🎉 All animations and interactions initialized successfully!');
