/* ============================================
   San Diego Eye Plastic Surgeons
   Main JavaScript
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
  initHeader();
  initMobileNav();
  initScrollAnimations();
  initParallax();
  initTestimonialSlider();
  initFaqAccordion();
  initGalleryFilter();
  initContactForm();
});

/* --- Header Scroll Behavior --- */
function initHeader() {
  const header = document.querySelector('.header');
  if (!header) return;

  const handleScroll = () => {
    if (window.scrollY > 60) {
      header.classList.add('header--scrolled');
      header.classList.remove('header--transparent');
    } else {
      header.classList.remove('header--scrolled');
      if (header.dataset.transparent === 'true') {
        header.classList.add('header--transparent');
      }
    }
  };

  handleScroll();
  window.addEventListener('scroll', handleScroll, { passive: true });
}

/* --- Mobile Navigation --- */
function initMobileNav() {
  const toggle = document.querySelector('.nav-toggle');
  const nav = document.querySelector('.nav');
  if (!toggle || !nav) return;

  toggle.addEventListener('click', () => {
    toggle.classList.toggle('active');
    nav.classList.toggle('open');
    document.body.style.overflow = nav.classList.contains('open') ? 'hidden' : '';
  });

  nav.querySelectorAll('a:not(.nav__dropdown-trigger)').forEach(link => {
    link.addEventListener('click', () => {
      toggle.classList.remove('active');
      nav.classList.remove('open');
      document.body.style.overflow = '';
    });
  });

  nav.querySelectorAll('.nav__dropdown-trigger').forEach(trigger => {
    trigger.addEventListener('click', (e) => {
      if (window.innerWidth <= 768) {
        e.preventDefault();
        trigger.closest('.nav__dropdown').classList.toggle('open');
      }
    });
  });
}

/* --- Scroll Animations (IntersectionObserver) --- */
function initScrollAnimations() {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // All animated elements
  const selectors = '.fade-in, .fade-in-left, .fade-in-right, .stagger-children, .text-reveal, .img-reveal, .credentials';
  const elements = document.querySelectorAll(selectors);
  if (!elements.length) return;

  if (prefersReducedMotion) {
    elements.forEach(el => el.classList.add('visible'));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -60px 0px' }
  );

  elements.forEach((el) => observer.observe(el));
}

/* --- Parallax Effect on Hero Backgrounds --- */
function initParallax() {
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReducedMotion) return;

  const heroes = document.querySelectorAll('.hero__bg');
  if (!heroes.length) return;

  let ticking = false;

  function updateParallax() {
    const scrollY = window.scrollY;
    heroes.forEach((bg) => {
      const hero = bg.parentElement;
      const heroBottom = hero.offsetTop + hero.offsetHeight;

      // Only animate while the hero is in view
      if (scrollY < heroBottom) {
        const offset = scrollY * 0.35;
        bg.style.transform = `translate3d(0, ${offset}px, 0) scale(1.1)`;
      }
    });
    ticking = false;
  }

  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(updateParallax);
      ticking = true;
    }
  }, { passive: true });

  // Set initial scale
  heroes.forEach((bg) => {
    bg.style.transform = 'translate3d(0, 0, 0) scale(1.1)';
  });
}

/* --- Testimonial Slider with Crossfade --- */
function initTestimonialSlider() {
  const testimonials = document.querySelectorAll('.testimonial');
  const dots = document.querySelectorAll('.testimonial-dot');
  if (!testimonials.length) return;

  let current = 0;
  let interval;

  function showTestimonial(index) {
    testimonials.forEach((t) => t.classList.remove('active'));
    dots.forEach((d) => d.classList.remove('active'));
    testimonials[index].classList.add('active');
    dots[index].classList.add('active');
    current = index;
  }

  function next() {
    showTestimonial((current + 1) % testimonials.length);
  }

  function startAutoplay() {
    interval = setInterval(next, 5000);
  }

  function stopAutoplay() {
    clearInterval(interval);
  }

  dots.forEach((dot, i) => {
    dot.addEventListener('click', () => {
      stopAutoplay();
      showTestimonial(i);
      startAutoplay();
    });
  });

  showTestimonial(0);
  startAutoplay();
}

/* --- FAQ Accordion --- */
function initFaqAccordion() {
  const items = document.querySelectorAll('.faq-item');
  if (!items.length) return;

  items.forEach((item) => {
    const question = item.querySelector('.faq-item__question');
    const answer = item.querySelector('.faq-item__answer');

    question.addEventListener('click', () => {
      const isOpen = item.classList.contains('active');

      items.forEach((i) => {
        i.classList.remove('active');
        i.querySelector('.faq-item__answer').style.maxHeight = null;
      });

      if (!isOpen) {
        item.classList.add('active');
        answer.style.maxHeight = answer.scrollHeight + 'px';
      }
    });
  });
}

/* --- Gallery Filter --- */
function initGalleryFilter() {
  const buttons = document.querySelectorAll('.gallery-filter__btn');
  const items = document.querySelectorAll('.gallery-grid .gallery-pair');
  if (!buttons.length || !items.length) return;

  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const filter = btn.dataset.filter;

      buttons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      items.forEach((item) => {
        if (filter === 'all' || item.dataset.category === filter) {
          item.style.opacity = '0';
          item.style.display = '';
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              item.style.transition = 'opacity 0.4s ease';
              item.style.opacity = '1';
            });
          });
        } else {
          item.style.transition = 'opacity 0.3s ease';
          item.style.opacity = '0';
          setTimeout(() => { item.style.display = 'none'; }, 300);
        }
      });
    });
  });
}

/* --- Contact Form --- */
function initContactForm() {
  const form = document.querySelector('#contact-form');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = Object.fromEntries(formData);

    const required = form.querySelectorAll('[required]');
    let valid = true;

    required.forEach((field) => {
      if (!field.value.trim()) {
        field.style.borderColor = '#c0392b';
        valid = false;
      } else {
        field.style.borderColor = '';
      }
    });

    if (!valid) return;

    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Sending...';
    submitBtn.disabled = true;

    // Placeholder for Weave integration
    setTimeout(() => {
      submitBtn.textContent = 'Message Sent';
      form.reset();
      setTimeout(() => {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
      }, 3000);
    }, 1000);
  });
}
