import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Pega o caminho da pasta onde o script está rodando
    caminho_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_data = os.path.join(caminho_atual, "data.yaml")

    # Verifica se o arquivo existe antes de começar
    if not os.path.exists(caminho_data):
        print(f"ERRO: O arquivo {caminho_data} não foi encontrado!")
        print("Arquivos na pasta:", os.listdir(caminho_atual))
    else:
        # Carrega o modelo
        model = YOLO('yolov8n.pt')

        # Treina o modelo
        model.train(
            data=caminho_data,
            epochs=300,
            imgsz=640,
            device='mps',  # Vai voar no seu M4
            augment=True
        )