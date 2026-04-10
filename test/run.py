import sys
import os
import base64
import time
from datetime import datetime

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
raiz_projeto = os.path.dirname(diretorio_atual)
sys.path.insert(0, raiz_projeto)

from core.omr_service import OMRService

# Função auxiliar para converter imagem em Base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

MODELO_PATH = os.path.join(raiz_projeto, 'models', 'preta.azul.best.pt')
 # 1. Inicializa o serviço
servico = OMRService(MODELO_PATH)

# 2. Converte a imagem para Base64 (como a sua função espera)
try:
    img_b64 = encode_image_to_base64("images/preta.png")
    
    # 3. Processa
    resultado = servico.processar_prova(img_b64)

    if resultado['status'] == 'sucesso':
        print("\n✅ Gabarito extraído com sucesso:", datetime.now().isoformat())
        # Ordena apenas para exibição bonita no terminal
        for q in sorted(resultado['gabarito'].keys()):
            print(f"Questão {q}: {resultado['gabarito'][q]}")
    else:
        print(f"\n❌ Erro no processamento: {resultado['erro']}")

except FileNotFoundError:
    print(f"❌ Erro: Arquivo não encontrado em {IMAGEM_TESTE_PATH}")
except Exception as e:
    print(f"❌ Ocorreu um erro inesperado: {e}")