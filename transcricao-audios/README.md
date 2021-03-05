# Removedor de Silêncio

Transcrição de audios usando os serviços: vosk e wit.

## Uso

Basta colocar os arquivos de audio .wav na pasta input e executar o main.py, as transcrições são escritas uma a uma no arquivo de saída caso ocorra um erro é possivel continuar a partir do ponto interrompido, observe que os arquivos de saída são zerados no inicio do código.
Audios devem ser menores que 20 segundos para o wit e recomenda-se não trabalhar com audios muito longos com vosk, foi usado 1 min de audio no máximo. Uso de multiprocessing para diminuir o tempo de processamento.
OBS: ao transcrever com Vosk ocorre-se o erro "[WinError 121] O tempo limite do semáforo expirou", não sei resolver isso.

## Referência

https://pypi.org/project/SpeechRecognition/
https://wit.ai
https://github.com/alphacep/vosk-api
https://github.com/alphacep/vosk-server
https://docs.python.org/3/library/multiprocessing.html
https://pypi.org/project/websockets/