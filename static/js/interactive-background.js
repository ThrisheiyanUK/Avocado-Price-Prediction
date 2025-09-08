// Interactive Background for Avocado Price Predictor
class InteractiveBackground {
    constructor() {
        this.mouseX = 0;
        this.mouseY = 0;
        this.particles = [];
        this.mouseTrails = [];
        this.maxParticles = 50;
        this.maxTrails = 10;
        this.animationId = null;
        
        this.init();
    }
    
    init() {
        this.createBackgroundContainer();
        this.createCursorGlow();
        this.createParticles();
        this.createFloatingEmojis();
        this.createWaves();
        this.bindEvents();
        this.startAnimation();
    }
    
    createBackgroundContainer() {
        // Create main background container
        const bgContainer = document.createElement('div');
        bgContainer.id = 'interactive-background';
        document.body.insertBefore(bgContainer, document.body.firstChild);
        this.bgContainer = bgContainer;
    }
    
    createCursorGlow() {
        // Create cursor glow effect
        this.cursorGlow = document.createElement('div');
        this.cursorGlow.className = 'cursor-glow';
        document.body.appendChild(this.cursorGlow);
    }
    
    createParticles() {
        // Create floating particles
        const particleTypes = ['avocado', 'leaf', 'sparkle'];
        
        for (let i = 0; i < this.maxParticles; i++) {
            const particle = document.createElement('div');
            const type = particleTypes[Math.floor(Math.random() * particleTypes.length)];
            const size = Math.random() * 30 + 10;
            
            particle.className = `particle ${type}`;
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.left = `${Math.random() * window.innerWidth}px`;
            particle.style.top = `${Math.random() * window.innerHeight}px`;
            particle.style.animationDelay = `${Math.random() * 8}s`;
            particle.style.animationDuration = `${8 + Math.random() * 4}s`;
            
            this.bgContainer.appendChild(particle);
            this.particles.push({
                element: particle,
                baseX: Math.random() * window.innerWidth,
                baseY: Math.random() * window.innerHeight,
                speedX: (Math.random() - 0.5) * 0.5,
                speedY: (Math.random() - 0.5) * 0.5,
                size: size
            });
        }
    }
    
    createFloatingEmojis() {
        // Create floating avocado emojis
        const emojis = ['🥑', '🌿', '💚', '✨', '🍃'];
        
        setInterval(() => {
            if (Math.random() > 0.3) return; // 30% chance to create emoji
            
            const emoji = document.createElement('div');
            emoji.className = 'floating-emoji';
            emoji.textContent = emojis[Math.floor(Math.random() * emojis.length)];
            emoji.style.left = `${Math.random() * window.innerWidth}px`;
            emoji.style.animationDuration = `${10 + Math.random() * 10}s`;
            
            this.bgContainer.appendChild(emoji);
            
            // Remove emoji after animation
            setTimeout(() => {
                if (emoji.parentNode) {
                    emoji.parentNode.removeChild(emoji);
                }
            }, 20000);
        }, 3000);
    }
    
    createWaves() {
        // Create animated wave effect
        const waveContainer = document.createElement('div');
        waveContainer.className = 'wave-container';
        
        for (let i = 0; i < 3; i++) {
            const wave = document.createElement('div');
            wave.className = 'wave';
            wave.style.animationDelay = `${i * 2}s`;
            waveContainer.appendChild(wave);
        }
        
        this.bgContainer.appendChild(waveContainer);
    }
    
    bindEvents() {
        // Mouse move events
        document.addEventListener('mousemove', (e) => {
            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
            this.updateCursorGlow();
            this.createMouseTrail();
            this.attractParticles();
        });
        
        // Mouse enter/leave for cursor glow
        document.addEventListener('mouseenter', () => {
            this.cursorGlow.style.opacity = '1';
        });
        
        document.addEventListener('mouseleave', () => {
            this.cursorGlow.style.opacity = '0';
        });
        
        // Click effect
        document.addEventListener('click', (e) => {
            this.createClickEffect(e.clientX, e.clientY);
        });
        
        // Resize handler
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Form interactions
        const formInputs = document.querySelectorAll('.form-control, .form-select');
        formInputs.forEach(input => {
            input.addEventListener('focus', () => {
                this.createFocusEffect(input);
            });
        });
    }
    
    updateCursorGlow() {
        // Update cursor glow position
        this.cursorGlow.style.left = `${this.mouseX - 20}px`;
        this.cursorGlow.style.top = `${this.mouseY - 20}px`;
    }
    
    createMouseTrail() {
        // Create trailing effect behind mouse
        const trail = document.createElement('div');
        trail.className = 'mouse-trail';
        trail.style.left = `${this.mouseX - 10}px`;
        trail.style.top = `${this.mouseY - 10}px`;
        
        document.body.appendChild(trail);
        
        // Animate trail
        requestAnimationFrame(() => {
            trail.classList.add('active');
        });
        
        // Remove trail after animation
        setTimeout(() => {
            if (trail.parentNode) {
                trail.parentNode.removeChild(trail);
            }
        }, 1000);
        
        // Limit number of trails
        this.mouseTrails.push(trail);
        if (this.mouseTrails.length > this.maxTrails) {
            const oldTrail = this.mouseTrails.shift();
            if (oldTrail.parentNode) {
                oldTrail.parentNode.removeChild(oldTrail);
            }
        }
    }
    
    attractParticles() {
        // Make particles attracted to mouse cursor
        this.particles.forEach(particle => {
            const dx = this.mouseX - parseInt(particle.element.style.left);
            const dy = this.mouseY - parseInt(particle.element.style.top);
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 150) {
                const force = (150 - distance) / 150;
                const attractionX = dx * force * 0.01;
                const attractionY = dy * force * 0.01;
                
                particle.speedX += attractionX;
                particle.speedY += attractionY;
                
                // Add glow effect when near mouse
                particle.element.style.filter = `brightness(${1 + force})`;
            } else {
                particle.element.style.filter = 'brightness(1)';
            }
        });
    }
    
    createClickEffect(x, y) {
        // Create ripple effect on click
        const ripple = document.createElement('div');
        ripple.style.position = 'fixed';
        ripple.style.left = `${x - 25}px`;
        ripple.style.top = `${y - 25}px`;
        ripple.style.width = '50px';
        ripple.style.height = '50px';
        ripple.style.border = '2px solid rgba(139, 195, 74, 0.6)';
        ripple.style.borderRadius = '50%';
        ripple.style.pointerEvents = 'none';
        ripple.style.zIndex = '9999';
        
        document.body.appendChild(ripple);
        
        // Animate ripple
        anime({
            targets: ripple,
            scale: [0, 3],
            opacity: [1, 0],
            duration: 600,
            easing: 'easeOutQuad',
            complete: () => {
                if (ripple.parentNode) {
                    ripple.parentNode.removeChild(ripple);
                }
            }
        });
        
        // Create burst of mini particles
        this.createClickBurst(x, y);
    }
    
    createClickBurst(x, y) {
        // Create burst of small particles on click
        const colors = ['rgba(139, 195, 74, 0.8)', 'rgba(76, 175, 80, 0.8)', 'rgba(104, 159, 56, 0.8)'];
        
        for (let i = 0; i < 8; i++) {
            const particle = document.createElement('div');
            particle.style.position = 'fixed';
            particle.style.left = `${x - 2}px`;
            particle.style.top = `${y - 2}px`;
            particle.style.width = '4px';
            particle.style.height = '4px';
            particle.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            particle.style.borderRadius = '50%';
            particle.style.pointerEvents = 'none';
            particle.style.zIndex = '9998';
            
            document.body.appendChild(particle);
            
            const angle = (i * 45) * (Math.PI / 180);
            const velocity = 100 + Math.random() * 50;
            
            anime({
                targets: particle,
                translateX: Math.cos(angle) * velocity,
                translateY: Math.sin(angle) * velocity,
                opacity: [1, 0],
                scale: [1, 0],
                duration: 800,
                easing: 'easeOutQuad',
                complete: () => {
                    if (particle.parentNode) {
                        particle.parentNode.removeChild(particle);
                    }
                }
            });
        }
    }
    
    createFocusEffect(input) {
        // Create effect when focusing on form inputs
        const rect = input.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        // Create glowing ring around input
        const ring = document.createElement('div');
        ring.style.position = 'fixed';
        ring.style.left = `${centerX - 40}px`;
        ring.style.top = `${centerY - 40}px`;
        ring.style.width = '80px';
        ring.style.height = '80px';
        ring.style.border = '2px solid rgba(139, 195, 74, 0.8)';
        ring.style.borderRadius = '50%';
        ring.style.pointerEvents = 'none';
        ring.style.zIndex = '9999';
        
        document.body.appendChild(ring);
        
        anime({
            targets: ring,
            scale: [0.5, 1.5],
            opacity: [1, 0],
            duration: 1000,
            easing: 'easeOutQuad',
            complete: () => {
                if (ring.parentNode) {
                    ring.parentNode.removeChild(ring);
                }
            }
        });
    }
    
    animateParticles() {
        // Animate floating particles
        this.particles.forEach(particle => {
            // Update position
            particle.baseX += particle.speedX;
            particle.baseY += particle.speedY;
            
            // Apply speed decay
            particle.speedX *= 0.99;
            particle.speedY *= 0.99;
            
            // Boundary check
            if (particle.baseX < -particle.size) {
                particle.baseX = window.innerWidth + particle.size;
            } else if (particle.baseX > window.innerWidth + particle.size) {
                particle.baseX = -particle.size;
            }
            
            if (particle.baseY < -particle.size) {
                particle.baseY = window.innerHeight + particle.size;
            } else if (particle.baseY > window.innerHeight + particle.size) {
                particle.baseY = -particle.size;
            }
            
            // Update DOM element position
            particle.element.style.left = `${particle.baseX}px`;
            particle.element.style.top = `${particle.baseY}px`;
        });
    }
    
    handleResize() {
        // Handle window resize
        this.particles.forEach(particle => {
            if (particle.baseX > window.innerWidth) {
                particle.baseX = Math.random() * window.innerWidth;
            }
            if (particle.baseY > window.innerHeight) {
                particle.baseY = Math.random() * window.innerHeight;
            }
        });
    }
    
    startAnimation() {
        // Main animation loop
        const animate = () => {
            this.animateParticles();
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }
    
    destroy() {
        // Clean up
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.bgContainer && this.bgContainer.parentNode) {
            this.bgContainer.parentNode.removeChild(this.bgContainer);
        }
        
        if (this.cursorGlow && this.cursorGlow.parentNode) {
            this.cursorGlow.parentNode.removeChild(this.cursorGlow);
        }
    }
}

// Constellation effect for special interactions
class ConstellationEffect {
    constructor(container) {
        this.container = container;
        this.stars = [];
        this.maxStars = 20;
        this.connections = [];
        
        this.createStars();
        this.animate();
    }
    
    createStars() {
        for (let i = 0; i < this.maxStars; i++) {
            const star = {
                x: Math.random() * window.innerWidth,
                y: Math.random() * window.innerHeight,
                vx: (Math.random() - 0.5) * 0.2,
                vy: (Math.random() - 0.5) * 0.2,
                alpha: Math.random() * 0.5 + 0.3
            };
            this.stars.push(star);
        }
    }
    
    drawConnections(ctx) {
        this.connections = [];
        
        for (let i = 0; i < this.stars.length; i++) {
            for (let j = i + 1; j < this.stars.length; j++) {
                const dx = this.stars[i].x - this.stars[j].x;
                const dy = this.stars[i].y - this.stars[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 150) {
                    const opacity = (150 - distance) / 150 * 0.3;
                    ctx.strokeStyle = `rgba(139, 195, 74, ${opacity})`;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(this.stars[i].x, this.stars[i].y);
                    ctx.lineTo(this.stars[j].x, this.stars[j].y);
                    ctx.stroke();
                }
            }
        }
    }
    
    animate() {
        // This would be used with canvas, but for simplicity we'll stick to DOM elements
    }
}

// Initialize interactive background when DOM is loaded
let interactiveBackground;

function initInteractiveBackground() {
    if (typeof anime === 'undefined') {
        console.warn('Anime.js not loaded, waiting...');
        setTimeout(initInteractiveBackground, 500);
        return;
    }
    
    interactiveBackground = new InteractiveBackground();
    console.log('🎨 Interactive background initialized!');
}

// Auto-initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initInteractiveBackground);
} else {
    initInteractiveBackground();
}

// Export for manual control
window.InteractiveBackground = InteractiveBackground;
