// Main JavaScript for Data Quality Rules Recommendation Engine

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // File input validation
    const fileInput = document.getElementById('csvFile');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                validateFile(file);
            }
        });
    }

    // Smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
});

function validateFile(file) {
    const maxSize = 16 * 1024 * 1024; // 16MB
    const allowedTypes = ['text/csv', 'application/csv'];

    if (file.size > maxSize) {
        showAlert('File size exceeds 16MB limit.', 'warning');
        return false;
    }

    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
        showAlert('Please select a CSV file.', 'warning');
        return false;
    }

    return true;
}

function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="fas fa-info-circle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    // Insert at the top of the main container
    const container = document.querySelector('main.container');
    container.insertBefore(alertDiv, container.firstChild);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showAlert('Copied to clipboard!', 'success');
    }, function() {
        showAlert('Failed to copy to clipboard.', 'danger');
    });
}

// Add copy buttons to code blocks
document.addEventListener('DOMContentLoaded', function() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(function(codeBlock) {
        const pre = codeBlock.parentElement;
        pre.style.position = 'relative';

        const copyButton = document.createElement('button');
        copyButton.className = 'btn btn-sm btn-outline-secondary position-absolute';
        copyButton.style.top = '8px';
        copyButton.style.right = '8px';
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.title = 'Copy to clipboard';

        copyButton.addEventListener('click', function() {
            copyToClipboard(codeBlock.textContent);
        });

        pre.appendChild(copyButton);
    });
});

// Export functionality
function exportRecommendations(format = 'json') {
    const recommendations = window.currentRecommendations;
    if (!recommendations) {
        showAlert('No recommendations to export.', 'warning');
        return;
    }

    let content, filename, mimeType;

    if (format === 'json') {
        content = JSON.stringify(recommendations, null, 2);
        filename = 'data_quality_recommendations.json';
        mimeType = 'application/json';
    } else if (format === 'csv') {
        const csvRows = [
            ['Rule Name', 'Column', 'Rule Type', 'Severity', 'Description', 'Violations', 'Business Impact']
        ];

        recommendations.forEach(rec => {
            csvRows.push([
                rec.rule_name,
                rec.column,
                rec.rule_type,
                rec.severity,
                rec.description,
                rec.current_violation_count,
                rec.business_impact
            ]);
        });

        content = csvRows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
        filename = 'data_quality_recommendations.csv';
        mimeType = 'text/csv';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showAlert(`Recommendations exported as ${format.toUpperCase()}`, 'success');
}

// Theme toggle functionality
function toggleTheme() {
    const body = document.body;
    const isDark = body.classList.contains('dark-theme');

    if (isDark) {
        body.classList.remove('dark-theme');
        localStorage.setItem('theme', 'light');
    } else {
        body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
    }
}

// Load saved theme
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
});