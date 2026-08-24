/**
 * FoldNetLiveLogo - Protein Structure Intelligence Animation
 * ============================================================
 * Converts the static FoldNet logo into a continuous, living scientific visualization
 * representing the end-to-end FoldNet prediction pipeline:
 * Protein Sequence -> Learned ESM-2 Representation -> Secondary Structure (Alpha Helix) -> Contact Map Network -> Convergence.
 */

class FoldNetLiveLogo {
    /**
     * @param {HTMLElement|string} container - Target DOM element or selector
     * @param {Object} options - Configuration parameters
     * @param {string} options.variant - 'hero' (larger, expressive) or 'dashboard' (restrained, compact)
     * @param {boolean} options.interactive - Enable cursor magnetic & glow interactions
     * @param {boolean} options.autoplay - Start animation immediately upon mount
     * @param {number} options.speed - Animation speed multiplier (default: 1.0)
     */
    constructor(container, options = {}) {
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) return;

        // Default Configuration
        this.options = Object.assign({
            variant: 'dashboard',
            interactive: true,
            autoplay: true,
            speed: 1.0
        }, options);

        // State variables
        this.animFrameId = null;
        this.startTime = null;
        this.mousePos = { x: 50, y: 50, active: false };
        this.targetMouse = { x: 50, y: 50 };
        this.currentMouse = { x: 50, y: 50 };

        // Check reduced motion accessibility
        this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        // Build SVG Elements and Bind Events
        this._initSVG();
        if (this.options.interactive) {
            this._bindEvents();
        }

        // Start render loop
        if (this.options.autoplay && !this.reducedMotion) {
            this.start();
        } else {
            this.renderStatic();
        }
    }

    /**
     * Constructs the responsive, high-DPI SVG structure for the FoldNet logo
     */
    _initSVG() {
        const isHero = this.options.variant === 'hero';
        const sizeClass = isHero ? 'foldnet-logo-hero' : 'foldnet-logo-dashboard';

        // Clear existing container content
        this.container.innerHTML = '';
        this.container.classList.add('foldnet-live-logo-container', sizeClass);

        // Unique ID prefix to avoid SVG gradient collisions if multiple logos exist
        const uid = 'fnl_' + Math.random().toString(36).substr(2, 6);
        this.uid = uid;

        // Construct SVG markup containing definitions, gradients, paths, and nodes
        this.container.innerHTML = `
            <svg class="foldnet-live-logo-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <!-- Background Glow Gradient -->
                    <radialGradient id="${uid}_bgGlow" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stop-color="#38bdf8" stop-opacity="${isHero ? 0.6 : 0.45}"/>
                        <stop offset="100%" stop-color="#818cf8" stop-opacity="0"/>
                    </radialGradient>

                    <!-- Interactive Cursor Glow Gradient -->
                    <radialGradient id="${uid}_cursorGlow" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.8"/>
                        <stop offset="100%" stop-color="#06b6d4" stop-opacity="0"/>
                    </radialGradient>

                    <!-- Sequence & Backbone Gradient -->
                    <linearGradient id="${uid}_lineGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#818cf8"/>
                        <stop offset="50%" stop-color="#38bdf8"/>
                        <stop offset="100%" stop-color="#10b981"/>
                    </linearGradient>

                    <!-- Contact Connection Gradient -->
                    <linearGradient id="${uid}_contactGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.9"/>
                        <stop offset="100%" stop-color="#f43f5e" stop-opacity="0.9"/>
                    </linearGradient>
                </defs>

                <!-- Background Ambient Glow -->
                <circle cx="50" cy="50" r="36" class="fn-glow-bg" fill="url(#${uid}_bgGlow)" />

                <!-- Dynamic Cursor Glow Layer -->
                <circle cx="50" cy="50" r="25" class="fn-cursor-glow" fill="url(#${uid}_cursorGlow)" opacity="0" />

                <!-- Outer Neural Ring -->
                <circle cx="50" cy="50" r="38" class="fn-outer-ring" stroke="#38bdf8" stroke-width="1.8" stroke-dasharray="12 8" />

                <!-- Phase 2: ESM-2 Representation Grid (4x4 Matrix) -->
                <g class="fn-matrix-grid" opacity="0">
                    <rect x="36" y="36" width="5" height="5" rx="1" fill="#38bdf8" class="fn-matrix-cell" />
                    <rect x="43" y="36" width="5" height="5" rx="1" fill="#818cf8" class="fn-matrix-cell" />
                    <rect x="50" y="36" width="5" height="5" rx="1" fill="#38bdf8" class="fn-matrix-cell" />
                    <rect x="57" y="36" width="5" height="5" rx="1" fill="#10b981" class="fn-matrix-cell" />

                    <rect x="36" y="43" width="5" height="5" rx="1" fill="#818cf8" class="fn-matrix-cell" />
                    <rect x="43" y="43" width="5" height="5" rx="1" fill="#38bdf8" class="fn-matrix-cell" />
                    <rect x="50" y="43" width="5" height="5" rx="1" fill="#10b981" class="fn-matrix-cell" />
                    <rect x="57" y="43" width="5" height="5" rx="1" fill="#818cf8" class="fn-matrix-cell" />

                    <rect x="36" y="50" width="5" height="5" rx="1" fill="#38bdf8" class="fn-matrix-cell" />
                    <rect x="43" y="50" width="5" height="5" rx="1" fill="#10b981" class="fn-matrix-cell" />
                    <rect x="50" y="50" width="5" height="5" rx="1" fill="#818cf8" class="fn-matrix-cell" />
                    <rect x="57" y="50" width="5" height="5" rx="1" fill="#38bdf8" class="fn-matrix-cell" />

                    <rect x="36" y="57" width="5" height="5" rx="1" fill="#10b981" class="fn-matrix-cell" />
                    <rect x="43" y="57" width="5" height="5" rx="1" fill="#38bdf8" class="fn-matrix-cell" />
                    <rect x="50" y="57" width="5" height="5" rx="1" fill="#818cf8" class="fn-matrix-cell" />
                    <rect x="57" y="57" width="5" height="5" rx="1" fill="#38bdf8" class="fn-matrix-cell" />
                </g>

                <!-- Phase 3: Secondary Structure Alpha Helix Ribbon Path -->
                <path d="M 25,65 C 20,40 35,20 50,20 C 65,20 80,40 75,65 C 70,80 50,85 50,65 C 50,45 35,35 50,35 C 65,35 60,65 50,65"
                      class="fn-helix-ribbon"
                      stroke="url(#${uid}_lineGrad)"
                      stroke-width="3"
                      stroke-linecap="round"
                      stroke-linejoin="round" />

                <!-- Phase 3: Subtle Beta-Sheet Indicator Ribbon (Secondary Element) -->
                <path d="M 30,72 L 45,78 L 60,72" class="fn-beta-sheet" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" opacity="0" />

                <!-- Phase 4: Contact Map Network Lines -->
                <g class="fn-contact-network" opacity="0">
                    <line x1="25" y1="65" x2="75" y2="65" class="fn-contact-line" stroke="url(#${uid}_contactGrad)" stroke-width="1.2" stroke-dasharray="3 3" />
                    <line x1="50" y1="20" x2="50" y2="65" class="fn-contact-line" stroke="url(#${uid}_contactGrad)" stroke-width="1.2" stroke-dasharray="3 3" />
                    <line x1="32" y1="38" x2="68" y2="38" class="fn-contact-line" stroke="url(#${uid}_contactGrad)" stroke-width="1.2" stroke-dasharray="3 3" />
                </g>

                <!-- Dynamic Sequence / Core Amino-Acid Nodes -->
                <g class="fn-nodes-group">
                    <circle cx="50" cy="20" r="4.5" class="fn-node node-top" fill="#818cf8" />
                    <circle cx="25" cy="65" r="4.5" class="fn-node node-left" fill="#38bdf8" />
                    <circle cx="75" cy="65" r="4.5" class="fn-node node-right" fill="#10b981" />
                    <circle cx="50" cy="65" r="5.5" class="fn-node node-center" fill="#ffffff" />

                    <!-- Moving Sequence Residue Dots (Phase 1) -->
                    <circle cx="20" cy="50" r="2.5" class="fn-seq-dot dot-1" fill="#38bdf8" opacity="0" />
                    <circle cx="35" cy="30" r="2.5" class="fn-seq-dot dot-2" fill="#818cf8" opacity="0" />
                    <circle cx="65" cy="30" r="2.5" class="fn-seq-dot dot-3" fill="#10b981" opacity="0" />
                    <circle cx="80" cy="50" r="2.5" class="fn-seq-dot dot-4" fill="#38bdf8" opacity="0" />
                </g>
            </svg>
        `;

        // Cache element references for fast frame updating
        this.svgEl = this.container.querySelector('.foldnet-live-logo-svg');
        this.glowBg = this.container.querySelector('.fn-glow-bg');
        this.cursorGlow = this.container.querySelector('.fn-cursor-glow');
        this.outerRing = this.container.querySelector('.fn-outer-ring');
        this.matrixGrid = this.container.querySelector('.fn-matrix-grid');
        this.helixRibbon = this.container.querySelector('.fn-helix-ribbon');
        this.betaSheet = this.container.querySelector('.fn-beta-sheet');
        this.contactNetwork = this.container.querySelector('.fn-contact-network');

        this.nodeTop = this.container.querySelector('.node-top');
        this.nodeLeft = this.container.querySelector('.node-left');
        this.nodeRight = this.container.querySelector('.node-right');
        this.nodeCenter = this.container.querySelector('.node-center');

        this.seqDots = Array.from(this.container.querySelectorAll('.fn-seq-dot'));
        this.contactLines = Array.from(this.container.querySelectorAll('.fn-contact-line'));

        // Measure path length for smooth ribbon flow stroke-dashoffset animation
        if (this.helixRibbon) {
            this.ribbonLength = this.helixRibbon.getTotalLength();
            this.helixRibbon.style.strokeDasharray = this.ribbonLength;
            this.helixRibbon.style.strokeDashoffset = '0';
        }
    }

    /**
     * Binds mouse interaction events for magnetic effect and cursor radial glow
     */
    _bindEvents() {
        this.container.addEventListener('mousemove', (e) => {
            const rect = this.container.getBoundingClientRect();
            // Calculate normalized cursor position relative to 100x100 SVG viewbox
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            const y = ((e.clientY - rect.top) / rect.height) * 100;
            this.targetMouse = { x, y };
            this.mousePos.active = true;
        });

        this.container.addEventListener('mouseleave', () => {
            this.targetMouse = { x: 50, y: 50 };
            this.mousePos.active = false;
        });
    }

    /**
     * Main animation update loop (runs target at 60 FPS)
     * Cycle duration: 6.0 seconds
     */
    _animate(timestamp) {
        if (!this.startTime) this.startTime = timestamp;
        const elapsed = (timestamp - this.startTime) * 0.001 * this.options.speed;
        const cycleDuration = 6.0; // 6 seconds total pipeline visualization cycle
        const progress = (elapsed % cycleDuration) / cycleDuration; // Normalized [0, 1]

        // --- Smooth Interpolation for Mouse Cursor Magnetism (Lerp) ---
        this.currentMouse.x += (this.targetMouse.x - this.currentMouse.x) * 0.1;
        this.currentMouse.y += (this.targetMouse.y - this.currentMouse.y) * 0.1;

        // Position cursor glow layer
        if (this.cursorGlow) {
            this.cursorGlow.setAttribute('cx', this.currentMouse.x.toFixed(2));
            this.cursorGlow.setAttribute('cy', this.currentMouse.y.toFixed(2));
            const glowOpacity = this.mousePos.active ? 0.8 : 0.0;
            this.cursorGlow.setAttribute('opacity', glowOpacity.toFixed(2));
        }

        // Magnetic displacement calculation for nodes
        const magDx = (this.currentMouse.x - 50) * 0.12;
        const magDy = (this.currentMouse.y - 50) * 0.12;

        // --- Rotate Outer Neural Engine Ring ---
        if (this.outerRing) {
            const ringAngle = (elapsed * 30) % 360;
            this.outerRing.setAttribute('transform', `rotate(${ringAngle.toFixed(2)}, 50, 50)`);
        }

        // =========================================================================
        // PIPELINE PHASES ANIMATION TIMING
        // =========================================================================
        // Phase 1: Protein Sequence        (t = 0.00 -> 0.25)
        // Phase 2: Learned Representation   (t = 0.25 -> 0.45)
        // Phase 3: Secondary Structure     (t = 0.45 -> 0.70)
        // Phase 4: Contact Map Network     (t = 0.65 -> 0.85)
        // Phase 5: Convergence to Logo     (t = 0.85 -> 1.00)

        // --- Phase 1: Protein Sequence Nodes ---
        if (progress >= 0.0 && progress < 0.35) {
            const p1 = progress / 0.35; // 0 to 1
            this.seqDots.forEach((dot, idx) => {
                const offset = (idx * 0.15);
                const dotP = Math.max(0, Math.min(1, (p1 - offset) / 0.55));
                dot.setAttribute('opacity', (Math.sin(dotP * Math.PI) * 0.9).toFixed(2));
                // Move dot along curving trajectory
                const angle = dotP * Math.PI * 1.2 - Math.PI * 0.6;
                const radius = 28 + Math.sin(dotP * Math.PI * 2) * 4;
                const dx = 50 + Math.cos(angle) * radius + magDx;
                const dy = 50 + Math.sin(angle) * radius + magDy;
                dot.setAttribute('cx', dx.toFixed(2));
                dot.setAttribute('cy', dy.toFixed(2));
            });
        } else {
            this.seqDots.forEach(dot => dot.setAttribute('opacity', '0'));
        }

        // --- Phase 2: ESM-2 Representation Grid ---
        if (progress >= 0.20 && progress < 0.50) {
            const p2 = (progress - 0.20) / 0.30; // 0 to 1
            const opacity = Math.sin(p2 * Math.PI);
            this.matrixGrid.setAttribute('opacity', (opacity * 0.85).toFixed(2));
            // Subtle breathing scale for matrix
            const scale = 1.0 + Math.sin(p2 * Math.PI * 3) * 0.05;
            this.matrixGrid.setAttribute('transform', `matrix(${scale.toFixed(3)}, 0, 0, ${scale.toFixed(3)}, ${(50 * (1 - scale)).toFixed(2)}, ${(50 * (1 - scale)).toFixed(2)})`);
        } else {
            this.matrixGrid.setAttribute('opacity', '0');
        }

        // --- Phase 3: Secondary Structure Alpha Helix Ribbon ---
        if (this.helixRibbon) {
            let helixOpacity = 1.0;
            if (progress < 0.20) {
                helixOpacity = 0.5 + (progress / 0.20) * 0.5;
            } else if (progress >= 0.20 && progress < 0.45) {
                // Fade out slightly during matrix representation phase
                helixOpacity = 0.4 + (1 - (progress - 0.20) / 0.25) * 0.4;
            } else if (progress >= 0.45 && progress < 0.85) {
                // Fully glowing during structure phase
                helixOpacity = 0.8 + Math.sin((progress - 0.45) / 0.40 * Math.PI) * 0.2;
            } else {
                helixOpacity = 1.0; // Converged
            }

            // Magnetic response & glow intensity
            if (this.mousePos.active) helixOpacity = Math.min(1.0, helixOpacity + 0.2);
            this.helixRibbon.setAttribute('opacity', helixOpacity.toFixed(2));

            // Flowing stroke animation along ribbon
            const dashOffset = (elapsed * 45) % this.ribbonLength;
            this.helixRibbon.style.strokeDashoffset = dashOffset.toFixed(2);
        }

        // Secondary Beta-Sheet Ribbon Indicator
        if (this.betaSheet) {
            if (progress >= 0.50 && progress < 0.75) {
                const p3 = (progress - 0.50) / 0.25;
                this.betaSheet.setAttribute('opacity', (Math.sin(p3 * Math.PI) * 0.75).toFixed(2));
            } else {
                this.betaSheet.setAttribute('opacity', '0');
            }
        }

        // --- Phase 4: Contact Map Network ---
        if (progress >= 0.55 && progress < 0.85) {
            const p4 = (progress - 0.55) / 0.30;
            const netOpacity = Math.sin(p4 * Math.PI);
            this.contactNetwork.setAttribute('opacity', (netOpacity * 0.9).toFixed(2));

            // Progressive dash shift for contact connections
            this.contactLines.forEach((line, idx) => {
                const dashShift = (elapsed * 20 + idx * 5) % 20;
                line.style.strokeDashoffset = dashShift.toFixed(2);
            });
        } else if (this.mousePos.active) {
            // Hover interaction reveals contact lines even outside Phase 4
            this.contactNetwork.setAttribute('opacity', '0.65');
        } else {
            this.contactNetwork.setAttribute('opacity', '0');
        }

        // --- Core Node Positions (with cursor magnetic displacement) ---
        if (this.nodeTop) {
            this.nodeTop.setAttribute('cx', (50 + magDx * 0.8).toFixed(2));
            this.nodeTop.setAttribute('cy', (20 + magDy * 0.8).toFixed(2));
        }
        if (this.nodeLeft) {
            this.nodeLeft.setAttribute('cx', (25 + magDx * 0.9).toFixed(2));
            this.nodeLeft.setAttribute('cy', (65 + magDy * 0.9).toFixed(2));
        }
        if (this.nodeRight) {
            this.nodeRight.setAttribute('cx', (75 + magDx * 0.9).toFixed(2));
            this.nodeRight.setAttribute('cy', (65 + magDy * 0.9).toFixed(2));
        }
        if (this.nodeCenter) {
            this.nodeCenter.setAttribute('cx', (50 + magDx * 1.2).toFixed(2));
            this.nodeCenter.setAttribute('cy', (65 + magDy * 1.2).toFixed(2));
        }

        // Request next frame
        this.animFrameId = requestAnimationFrame(this._animate.bind(this));
    }

    /**
     * Start animation loop
     */
    start() {
        if (!this.animFrameId) {
            this.animFrameId = requestAnimationFrame(this._animate.bind(this));
        }
    }

    /**
     * Stop animation loop
     */
    stop() {
        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }
    }

    /**
     * Render static logo (for prefers-reduced-motion compliance)
     */
    renderStatic() {
        this.stop();
        if (this.helixRibbon) this.helixRibbon.setAttribute('opacity', '1.0');
        if (this.matrixGrid) this.matrixGrid.setAttribute('opacity', '0');
        if (this.contactNetwork) this.contactNetwork.setAttribute('opacity', '0.2');
        if (this.outerRing) this.outerRing.setAttribute('transform', 'rotate(0, 50, 50)');
    }
}

// Export module for global browser consumption
if (typeof window !== 'undefined') {
    window.FoldNetLiveLogo = FoldNetLiveLogo;
}
