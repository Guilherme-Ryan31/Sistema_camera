// Variável global para o gráfico
let myActivityChart = null;

// NOVA FUNÇÃO - INICIALIZAR O GRÁFICO
function inicializarGrafico() {
    const ctx = document.getElementById('activityChart').getContext('2d');
    
    // Gradiente para o gráfico (combinando com seu estilo)
    const gradient = ctx.createLinearGradient(0, 0, 0, 250);
    gradient.addColorStop(0, 'rgba(244, 67, 54, 0.6)');
    gradient.addColorStop(1, 'rgba(244, 67, 54, 0.1)');

    myActivityChart = new Chart(ctx, {
        type: 'bar', // Gráfico de barras
        data: {
            labels: [], // Ex: ['Violencia', 'Suspeito']
            datasets: [{
                label: 'Contagem de Eventos',
                data: [], // Ex: [5, 2]
                backgroundColor: gradient,
                borderColor: 'rgba(244, 67, 54, 1)',
                borderWidth: 1,
                borderRadius: 5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        color: '#A0A0A0',
                        stepSize: 1 // Forçar contagem inteira (1, 2, 3...)
                    },
                    grid: {
                        color: '#3A3A3A' // Linhas do grid
                    }
                },
                x: {
                    ticks: {
                        color: '#FFFFFF' // Labels do eixo X
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false // Esconde a legenda
                },
                tooltip: {
                    backgroundColor: '#1E1E1E',
                    titleColor: '#FFFFFF',
                    bodyColor: '#FFFFFF'
                }
            }
        }
    });
}

// NOVA FUNÇÃO - ATUALIZAR O GRÁFICO COM DADOS
function atualizarGrafico(videos) {
    // 1. Processar os dados dos vídeos para contar os eventos
    const contagemEventos = {}; // Ex: {'Violencia Detectada': 2, 'Comportamento Suspeito': 1}
    
    videos.forEach(video => {
        const evento = video.evento; // Já vem formatado do Flask!
        if (contagemEventos[evento]) {
            contagemEventos[evento]++;
        } else {
            contagemEventos[evento] = 1;
        }
    });

    // 2. Preparar dados para o Chart.js
    const labels = Object.keys(contagemEventos);
    const data = Object.values(contagemEventos);

    // 3. Atualizar o gráfico
    if (myActivityChart) {
        myActivityChart.data.labels = labels;
        myActivityChart.data.datasets[0].data = data;
        myActivityChart.update(); // Redesenha o gráfico
    }
}