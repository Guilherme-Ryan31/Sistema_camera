// Variável global para o gráfico
let myActivityChart = null;

// NOVA FUNÇÃO - INICIALIZAR O GRÁFICO
function inicializarGrafico() {
    const ctx = document.getElementById('activityChart').getContext('2d');
    
    myActivityChart = new Chart(ctx, {
        type: 'pie', // MUDANÇA: Alterado de 'bar' para 'pie' (Pizza)
        data: {
            labels: [], // Ex: ['Violência', 'Suspeito']
            datasets: [{
                data: [], // Ex: [5, 2]
                backgroundColor: [
                    '#FF4444', // Vermelho para Violência
                    '#FFEB3B', // Amarelo para Suspeito
                    '#FF9800', // Laranja para Ilícito
                    '#2196F3', // Azul para Outros
                    '#4CAF50'  // Verde para Outros
                ],
                borderColor: '#1E2329', // Cor da borda (fundo do site) para separar as fatias
                borderWidth: 2,
                hoverOffset: 10 // Efeito de destaque ao passar o mouse
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true, // MOSTRAR a legenda
                    position: 'right', // Legenda à direita
                    labels: {
                        color: '#FAFAFA', // Cor do texto da legenda
                        font: {
                            size: 12
                        },
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#FFFFFF',
                    bodyColor: '#FFFFFF',
                    bodyFont: {
                        size: 14
                    },
                    padding: 10,
                    callbacks: {
                        label: function(context) {
                            let label = context.label || '';
                            let value = context.raw || 0;
                            // Calcula a porcentagem
                            let total = context.chart._metasets[context.datasetIndex].total;
                            let percentage = Math.round((value / total) * 100) + '%';
                            return ` ${label}: ${value} (${percentage})`;
                        }
                    }
                }
            },
            layout: {
                padding: {
                    left: 20,
                    right: 20,
                    top: 0,
                    bottom: 0
                }
            }
        }
    });
}

// NOVA FUNÇÃO - ATUALIZAR O GRÁFICO COM DADOS
// NOVA FUNÇÃO - ATUALIZAR O GRÁFICO COM DADOS
function atualizarGrafico(videos) {
    // 1. Processar os dados dos vídeos para contar as AÇÕES ESPECÍFICAS
    const contagemAcoes = {}; 
    
    videos.forEach(video => {
        // MUDANÇA CRUCIAL: Tenta ler 'video.acao'. Se não existir, lê 'video.evento'.
        // Isso garante que "Beatboxing" seja contado separado de "Punching".
        let labelGrafico = video.acao || video.evento; 
        
        // Formatação extra para garantir que fique bonito (opcional)
        labelGrafico = labelGrafico.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        if (contagemAcoes[labelGrafico]) {
            contagemAcoes[labelGrafico]++;
        } else {
            contagemAcoes[labelGrafico] = 1;
        }
    });

    // 2. Preparar dados para o Chart.js
    const labels = Object.keys(contagemAcoes);
    const data = Object.values(contagemAcoes);

    // 3. Atualizar o gráfico
    if (myActivityChart) {
        myActivityChart.data.labels = labels;
        myActivityChart.data.datasets[0].data = data;
        
        if (data.length === 0) {
            // Estado vazio
            myActivityChart.data.labels = ['Sem Dados'];
            myActivityChart.data.datasets[0].data = [1];
            myActivityChart.data.datasets[0].backgroundColor = ['#3A3A3A'];
            myActivityChart.options.plugins.legend.display = false;
        } else {
            // Cores variadas para diferenciar as fatias
            myActivityChart.data.datasets[0].backgroundColor = [
                '#FF4444', // Vermelho
                '#2196F3', // Azul
                '#FFEB3B', // Amarelo
                '#4CAF50', // Verde
                '#9C27B0', // Roxo
                '#FF9800'  // Laranja
            ];
            myActivityChart.options.plugins.legend.display = true;
        }

        myActivityChart.update();
    }
}