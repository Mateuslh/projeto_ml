document.addEventListener('DOMContentLoaded', function() {
    const form1 = document.getElementById('newsForm1');
    const input1 = document.getElementById('newsInput1');
    const loading1 = document.getElementById('loading1');
    const responseElement1 = document.getElementById('response1');

    form1.addEventListener('submit', async function(event) {
        event.preventDefault();

        const inputValue = input1.value.trim();

        if (inputValue === '') {
            alert('Por favor, digite uma notícia.');
            return;
        }

        loading1.style.display = 'block'; // Mostra o indicador de carregamento

        try {
            const apiUrl = 'http://127.0.0.1:5000/predict_regressao'; // URL da sua API de detecção de fake news regressão

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
            input1.value = ''; // Limpa o campo de entrada
            loading1.style.display = 'none'; // Esconde o indicador de carregamento

            // Verifica o resultado da API e exibe mensagem correspondente
            if ((responseData.resultado === 'Verdadeira') || (responseData.resultado === 'Real')) {
                responseElement1.textContent = 'A notícia é VERDADEIRA.';
                responseElement1.style.color = 'green';
            } else if ((responseData.resultado === 'Falsa') || (responseData.resultado === 'Fake')) {
                responseElement1.textContent = 'A notícia é FALSA.';
                responseElement1.style.color = 'red';
            } else {
                responseElement1.textContent = 'Não foi possível determinar se a notícia é verdadeira ou falsa.';
                responseElement1.style.color = 'black';
            }

            setTimeout(() => {
                responseElement1.textContent = '';
                responseElement1.style.color = 'black'; // Reseta a cor para a próxima mensagem
            }, 3000); // Tempo em milissegundos para a mensagem desaparecer

        } catch (error) {
            loading1.style.display = 'none'; // Esconde o indicador de carregamento em caso de erro
            responseElement1.textContent = 'Erro ao enviar notícia. Tente novamente mais tarde.';
            responseElement1.style.color = 'black';
            console.error('Erro:', error);
        }
    });

    const form2 = document.getElementById('newsForm2');
    const input2 = document.getElementById('newsInput2');
    const loading2 = document.getElementById('loading2');
    const responseElement2 = document.getElementById('response2');

    form2.addEventListener('submit', async function(event) {
        event.preventDefault();

        const inputValue = input2.value.trim();

        if (inputValue === '') {
            alert('Por favor, digite uma notícia.');
            return;
        }

        loading2.style.display = 'block'; // Mostra o indicador de carregamento

        try {
            const apiUrl = 'http://127.0.0.1:5000/predict_floresta'; // URL da sua API de detecção de fake news floresta

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
            input2.value = ''; // Limpa o campo de entrada
            loading2.style.display = 'none'; // Esconde o indicador de carregamento

            // Verifica o resultado da API e exibe mensagem correspondente
            if ((responseData.resultado === 'Verdadeira') || (responseData.resultado === 'Real')) {
                responseElement2.textContent = 'A notícia é VERDADEIRA.';
                responseElement2.style.color = 'green';
            } else if ((responseData.resultado === 'Falsa') || (responseData.resultado === 'Fake')) {
                responseElement2.textContent = 'A notícia é FALSA.';
                responseElement2.style.color = 'red';
            } else {
                responseElement2.textContent = 'Não foi possível determinar se a notícia é verdadeira ou falsa.';
                responseElement2.style.color = 'black';
            }

            setTimeout(() => {
                responseElement2.textContent = '';
                responseElement2.style.color = 'black'; // Reseta a cor para a próxima mensagem
            }, 3000); // Tempo em milissegundos para a mensagem desaparecer

        } catch (error) {
            loading2.style.display = 'none'; // Esconde o indicador de carregamento em caso de erro
            responseElement2.textContent = 'Erro ao enviar notícia. Tente novamente mais tarde.';
            responseElement2.style.color = 'black';
            console.error('Erro:', error);
        }
    });
});
