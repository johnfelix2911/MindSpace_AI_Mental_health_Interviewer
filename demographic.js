/* ── Collapsible section toggle ─────────────────────────────── */
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    section.classList.toggle('collapsed');
}

/* ── Stressor chip multi-select ────────────────────────────── */
function initStressorChips() {
    const chips = document.querySelectorAll('.stressor-chip');
    chips.forEach(chip => {
        const checkbox = chip.querySelector('input[type="checkbox"]');
        if (checkbox) {
            checkbox.addEventListener('change', function() {
                chip.classList.toggle('active', this.checked);
            });
        }
    });
}

/* ── Form submission (preserved logic) ─────────────────────── */
async function continueInterview() {
    const statusMsg = document.getElementById('statusMessage');
    const btn = document.querySelector('.primary-btn');

    btn.classList.add('loading');
    btn.innerHTML = '<span class="btn-spinner"></span>Saving your information...';
    statusMsg.textContent = '';

    try {
        const stressors = [];
        document.querySelectorAll('.stressor-chip input[type="checkbox"]:checked').forEach(cb => {
            stressors.push(cb.value);
        });

        const formData = {
            name: document.getElementById('name')?.value || null,
            age: document.getElementById('age')?.value || null,
            gender: document.getElementById('gender')?.value || null,
            country: document.getElementById('country')?.value || null,
            role: document.getElementById('role')?.value || null,
            stage: document.getElementById('stage')?.value || null,
            focus: document.getElementById('focus')?.value || null,
            sleep_duration: document.getElementById('sleep_duration')?.value || null,
            workload: document.getElementById('workload')?.value || null,
            screen_time: document.getElementById('screen_time')?.value || null,
            living_situation: document.getElementById('living_situation')?.value || null,
            support_system: document.getElementById('support_system')?.value || null,
            stressors: stressors,
        };

        const response = await fetch('/submit_demographics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();

        if (response.ok) {
            btn.innerHTML = 'Session ready';
            statusMsg.textContent = 'Redirecting to your interview...';
            statusMsg.style.color = '#41d6a4';

            const transition = document.getElementById('pageTransition');
            if (transition) {
                transition.classList.add('active');
            }

            setTimeout(() => {
                window.location.href = '/interview';
            }, 1200);
        } else {
            btn.classList.remove('loading');
            btn.innerHTML = 'Continue to Interview';
            statusMsg.textContent = 'Error: ' + (result.detail || 'Submission failed');
            statusMsg.style.color = '#ff6f7a';
        }
    } catch (error) {
        btn.classList.remove('loading');
        btn.innerHTML = 'Continue to Interview';
        statusMsg.textContent = 'Error: ' + error.message;
        statusMsg.style.color = '#ff6f7a';
    }
}

/* ── Init ──────────────────────────────────────────────────── */
window.addEventListener('DOMContentLoaded', () => {
    initStressorChips();
});
