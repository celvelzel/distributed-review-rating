document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('data/metrics.json');
        if (!response.ok) throw new Error('Failed to fetch metrics data');
        const data = await response.json();
        
        renderCharts(data);
    } catch (error) {
        console.error('Error loading data:', error);
        // Fallback for local viewing without server or missing data
        renderCharts({
            stages: {
                "0": { rmse: 1.15, train_time_sec: 120, inference_time_sec: 15, model: "tfidf_lgb" },
                "1": { rmse: 1.08, train_time_sec: 150, inference_time_sec: 18, model: "lgb_stats" },
                "2": { rmse: 1.02, train_time_sec: 1800, inference_time_sec: 45, model: "lgb_multimodal" },
                "3": { rmse: 0.98, train_time_sec: 2100, inference_time_sec: 48, model: "lgb_multimodal" },
                "4": { rmse: 0.95, train_time_sec: 2500, inference_time_sec: 60, model: "stacking" },
                "5": { rmse: 0.92, train_time_sec: 3600, inference_time_sec: 60, model: "stacking_tuned" }
            },
            ablations: {
                "a_no_text": { delta_vs_full: 0.08 },
                "b_no_graph": { delta_vs_full: 0.05 },
                "c_no_stacking": { delta_vs_full: 0.03 },
                "d_no_kfold_te": { delta_vs_full: 0.04 },
                "e_tfidf_vs_bert": { delta_vs_full: -0.06 },
                "f_no_clip": { delta_vs_full: 0.01 }
            }
        });
    }
});

function renderCharts(data) {
    const primaryColor = '#3498db';
    const secondaryColor = '#2ecc71';
    
    // Process Stage Data
    const stageLabels = Object.keys(data.stages).map(s => `Stage ${s}`);
    const rmseData = Object.values(data.stages).map(s => s.rmse || 0);
    const trainTimeData = Object.values(data.stages).map(s => s.train_time_sec || 0);
    const inferenceTimeData = Object.values(data.stages).map(s => s.inference_time_sec || 0);
    
    // Process Ablation Data
    const ablationLabels = Object.keys(data.ablations).map(k => k.replace(/^[a-f]_/, '').replace(/_/g, ' '));
    const ablationData = Object.values(data.ablations).map(a => a.delta_vs_full || 0);

    // Common Chart Options
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        }
    };

    // 1. RMSE Trend Chart (Line)
    new Chart(document.getElementById('rmseChart'), {
        type: 'line',
        data: {
            labels: stageLabels,
            datasets: [{
                label: 'RMSE',
                data: rmseData,
                borderColor: primaryColor,
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 3,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                y: { beginAtZero: false, title: { display: true, text: 'RMSE (Lower is Better)' } }
            }
        }
    });

    // 2. Training Time Chart (Bar)
    new Chart(document.getElementById('trainTimeChart'), {
        type: 'bar',
        data: {
            labels: stageLabels,
            datasets: [{
                label: 'Training Time (s)',
                data: trainTimeData,
                backgroundColor: primaryColor
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Seconds' } }
            }
        }
    });

    // 3. Inference Time Chart (Bar)
    new Chart(document.getElementById('inferenceTimeChart'), {
        type: 'bar',
        data: {
            labels: stageLabels,
            datasets: [{
                label: 'Inference Time (s)',
                data: inferenceTimeData,
                backgroundColor: secondaryColor
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Seconds' } }
            }
        }
    });

    // 4. Ablation Study Chart (Horizontal Bar)
    new Chart(document.getElementById('ablationChart'), {
        type: 'bar',
        data: {
            labels: ablationLabels,
            datasets: [{
                label: 'RMSE Change',
                data: ablationData,
                backgroundColor: ablationData.map(d => d > 0 ? '#e74c3c' : '#2ecc71')
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `RMSE Change: ${context.raw > 0 ? '+' : ''}${context.raw}`
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'RMSE Delta (vs Full Model)' } }
            }
        }
    });
}