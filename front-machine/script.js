document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('newsForm');
    const input = document.getElementById('newsInput');
    const loading = document.getElementById('loading');
    const responseElement = document.getElementById('response');

    form.addEventListener('submit', async function(event) {
        event.preventDefault();

        const inputValue = input.value.trim();

        if (inputValue === '') {
            alert('Por favor, digite uma notícia.');
            return;
        }

        loading.style.display = 'block'; // Mostra o indicador de carregamento

        try {
            const apiUrl = 'http://127.0.0.1:5000/predict'; // URL da sua API de detecção de fake news

            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ noticia: inputValue }),
            });

            if (!response.ok) {
                throw new Error('Erro ao enviar notícia.');
            }

            const responseData = await response.json();
            console.log('Resposta:', responseData);
            input.value = ''; // Limpa o campo de entrada
            loading.style.display = 'none'; // Esconde o indicador de carregamento

            // Verifica o resultado da API e exibe mensagem correspondente
            if (responseData.resultado === 'Verdadeira') {
                responseElement.textContent = 'A notícia é VERDADEIRA.';
                responseElement.style.color = 'green';
            } else if (responseData.resultado === 'Falsa') {
                responseElement.textContent = 'A notícia é FALSA.';
                responseElement.style.color = 'red';
            } else {
                responseElement.textContent = 'Não foi possível determinar se a notícia é verdadeira ou falsa.';
                responseElement.style.color = 'black';
            }

            setTimeout(() => {
                responseElement.textContent = '';
                responseElement.style.color = 'black'; // Reseta a cor para a próxima mensagem
            }, 3000); // Tempo em milissegundos para a mensagem desaparecer

        } catch (error) {
            loading.style.display = 'none'; // Esconde o indicador de carregamento em caso de erro
            responseElement.textContent = 'Erro ao enviar notícia. Tente novamente mais tarde.';
            responseElement.style.color = 'black';
            console.error('Erro:', error);
        }
    });
});
