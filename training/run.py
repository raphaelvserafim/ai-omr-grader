import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Pega o caminho da pasta onde o script está rodando
    caminho_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_data = os.path.join(caminho_atual, "data.yaml")

    # Verifica se o arquivo existe antes de começar
    if not os.path.exists(caminho_data):
        print(f"ERRO: O arquivo {caminho_data} não foi encontrado!")
    else:
        # Carrega o modelo
        model = YOLO('yolov8n.pt')

        # Treina o modelo
        model.train(
            data=caminho_data,
            epochs=300,
            imgsz=640,
            device='mps',   # Usa a GPU do M4
            batch=8,        # Reduzi de 16 (padrão) para 8 para não estourar a RAM
            workers=4,      # Limita o número de processos de carga de dados (evita o "Killed")
            augment=True,
            exist_ok=True,  # Sobrescreve se a pasta já existir
            cache=False     # Se o seu dataset for grande, deixe False para poupar RAM
        )