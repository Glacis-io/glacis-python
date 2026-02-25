/**
 * GLACIS Lead Capture Script
 *
 * Embeddable script for capturing waitlist signups on glacis.io
 * Posts to the compliance platform API at app.glacis.io
 *
 * Usage:
 *   <script src="/glacis-convert.js" data-source="bbc-radio4"></script>
 *   <form id="glacis-waitlist">
 *     <input type="email" name="email" placeholder="Enter your email" required>
 *     <button type="submit">Join Waitlist</button>
 *   </form>
 */

(function() {
  'use strict';

  // Configuration
  const API_URL = 'https://app.glacis.io/api/v1/public/waitlist';
  const SCRIPT_TAG = document.currentScript;
  const SOURCE = SCRIPT_TAG?.getAttribute('data-source') || 'website';

  // Find or create the waitlist form
  function initForm() {
    const form = document.getElementById('glacis-waitlist');
    if (!form) {
      console.warn('[GLACIS] No form found with id="glacis-waitlist"');
      return;
    }

    const emailInput = form.querySelector('input[type="email"], input[name="email"]');
    const submitButton = form.querySelector('button[type="submit"], input[type="submit"]');

    if (!emailInput) {
      console.warn('[GLACIS] No email input found in form');
      return;
    }

    // Create or find message container
    let messageContainer = form.querySelector('.glacis-message');
    if (!messageContainer) {
      messageContainer = document.createElement('div');
      messageContainer.className = 'glacis-message';
      messageContainer.style.cssText = 'margin-top: 12px; padding: 12px 16px; border-radius: 6px; font-size: 14px; display: none;';
      form.appendChild(messageContainer);
    }

    // Handle form submission
    form.addEventListener('submit', async function(e) {
      e.preventDefault();

      const email = emailInput.value.trim();
      if (!email) {
        showMessage(messageContainer, 'Please enter your email address.', 'error');
        return;
      }

      // Disable form while submitting
      emailInput.disabled = true;
      if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = submitButton.getAttribute('data-loading') || 'Securing your spot...';
      }

      try {
        const response = await fetch(API_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: email,
            source: SOURCE
          })
        });

        const data = await response.json();

        if (response.ok && data.success) {
          // Success!
          showMessage(
            messageContainer,
            'Spot secured. Your Compliance Risk Score assessment is being prepared.',
            'success'
          );

          // Optional: redirect to wizard if URL provided and configured
          if (data.redirectUrl && SCRIPT_TAG?.getAttribute('data-auto-redirect') === 'true') {
            setTimeout(() => {
              window.location.href = data.redirectUrl;
            }, 2000);
          }

          // Track conversion event if analytics available
          if (typeof gtag === 'function') {
            gtag('event', 'waitlist_signup', {
              event_category: 'conversion',
              event_label: SOURCE
            });
          }

          // Clear form
          emailInput.value = '';
        } else if (response.status === 429) {
          // Rate limited
          showMessage(
            messageContainer,
            'Too many requests. Please try again in a few minutes.',
            'error'
          );
        } else {
          // Other error
          showMessage(
            messageContainer,
            data.error || 'Something went wrong. Please try again.',
            'error'
          );
        }
      } catch (error) {
        console.error('[GLACIS] Waitlist error:', error);
        showMessage(
          messageContainer,
          'Network error. Please check your connection and try again.',
          'error'
        );
      } finally {
        // Re-enable form
        emailInput.disabled = false;
        if (submitButton) {
          submitButton.disabled = false;
          submitButton.textContent = submitButton.getAttribute('data-original') || 'Join Waitlist';
        }
      }
    });

    // Store original button text
    if (submitButton && !submitButton.getAttribute('data-original')) {
      submitButton.setAttribute('data-original', submitButton.textContent);
    }
  }

  // Show message with appropriate styling
  function showMessage(container, message, type) {
    container.textContent = message;
    container.style.display = 'block';

    if (type === 'success') {
      container.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
      container.style.border = '1px solid rgba(16, 185, 129, 0.3)';
      container.style.color = '#059669';
    } else {
      container.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
      container.style.border = '1px solid rgba(239, 68, 68, 0.3)';
      container.style.color = '#dc2626';
    }

    // Auto-hide errors after 5 seconds
    if (type === 'error') {
      setTimeout(() => {
        container.style.display = 'none';
      }, 5000);
    }
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initForm);
  } else {
    initForm();
  }

  // Expose for manual initialization if needed
  window.GlacisWaitlist = {
    init: initForm,
    submit: async function(email, source) {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, source: source || SOURCE })
      });
      return response.json();
    }
  };
})();
