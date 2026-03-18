let modelData = [];
let currentSortColumn = 'overall';
let currentSortAscending = false;

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    fetchData();
    setupSortListeners();
});

function initTheme() {
    const toggleBtn = document.getElementById('theme-toggle');
    
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
        document.body.classList.add('dark-mode');
    }
    
    toggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('theme', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
    });
}

async function fetchData() {
    try {
        const response = await fetch('data.json');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        document.getElementById('last-updated').textContent = `Last updated: ${data.last_updated}`;
        modelData = data.models || [];
        
        sortTable(currentSortColumn, currentSortAscending);
    } catch (error) {
        console.error('Failed to fetch data:', error);
        const tbody = document.getElementById('table-body');
        tbody.innerHTML = `<tr><td colspan="11" style="color: red; text-align: center;">Error loading data: ${error.message}</td></tr>`;
    }
}

function setupSortListeners() {
    const headers = document.querySelectorAll('th.sortable');
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const columnKey = header.getAttribute('data-sort');
            
            if (columnKey === currentSortColumn) {
                currentSortAscending = !currentSortAscending;
            } else {
                currentSortColumn = columnKey;
                currentSortAscending = ['name', 'rank'].includes(columnKey);
            }
            
            updateSortIndicators(headers);
            sortTable(currentSortColumn, currentSortAscending);
        });
    });
}

function updateSortIndicators(headers) {
    headers.forEach(header => {
        const text = header.textContent.replace(/[↑↓↕]/g, '').trim();
        if (header.getAttribute('data-sort') === currentSortColumn) {
            header.textContent = `${text} ${currentSortAscending ? '↑' : '↓'}`;
        } else {
            header.textContent = `${text} ↕`;
        }
    });
}

function sortTable(columnKey, ascending) {
    if (modelData.length === 0) return;
    
    const rankSorted = [...modelData].sort((a, b) => (b.overall || 0) - (a.overall || 0));
    const modelToRank = new Map();
    rankSorted.forEach((model, index) => {
        modelToRank.set(model.name, index + 1);
    });

    const sortedData = [...modelData].sort((a, b) => {
        let valA, valB;
        
        if (columnKey === 'rank') {
            valA = modelToRank.get(a.name);
            valB = modelToRank.get(b.name);
        } else if (columnKey === 'name') {
            valA = a.name.toLowerCase();
            valB = b.name.toLowerCase();
        } else {
            valA = a[columnKey] !== undefined ? a[columnKey] : -1;
            valB = b[columnKey] !== undefined ? b[columnKey] : -1;
        }

        if (valA < valB) return ascending ? -1 : 1;
        if (valA > valB) return ascending ? 1 : -1;
        return 0;
    });

    buildTable(sortedData, modelToRank);
}

function scoreToColor(score) {
    if (score === undefined || score === null || score < 0) return 'transparent';
    
    const val = Math.max(0, Math.min(1, score));
    
    let r, g, b;
    if (val < 0.5) {
        const ratio = val * 2;
        r = 255;
        g = Math.round(255 * ratio);
        b = 0;
    } else {
        const ratio = (val - 0.5) * 2;
        r = Math.round(255 * (1 - ratio));
        g = 255;
        b = 0;
    }
    
    return `rgba(${r}, ${g}, ${b}, 0.15)`;
}

function formatScore(score) {
    if (score === undefined || score === null || score < 0) return '-';
    return (score * 100).toFixed(1);
}

function buildTable(models, modelToRank) {
    const tbody = document.getElementById('table-body');
    tbody.innerHTML = '';
    
    models.forEach(model => {
        const tr = document.createElement('tr');
        const rank = modelToRank.get(model.name);
        
        const rankClass = rank === 1 ? 'rank-1' : '';
        const rankDisplay = rank === 1 ? '🥇 1' : rank === 2 ? '🥈 2' : rank === 3 ? '🥉 3' : rank;
        
        const metrics = [
            'overall', 
            'sentiment_smsa', 
            'nli_wrete', 
            'qa_facqa', 
            'ner_nergrit', 
            'summarization_indosum', 
            'mt_nusax', 
            'toxicity_id', 
            'cultural_indommu'
        ];
        
        let html = `
            <td class="${rankClass}">${rankDisplay}</td>
            <td style="font-weight: 500;">${model.name}</td>
        `;
        
        metrics.forEach(metric => {
            const score = model[metric];
            const bgColor = scoreToColor(score);
            html += `<td class="score-cell" style="background-color: ${bgColor}" title="${score !== undefined ? score : 'N/A'}">${formatScore(score)}</td>`;
        });
        
        tr.innerHTML = html;
        tbody.appendChild(tr);
    });
}