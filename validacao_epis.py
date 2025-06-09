import cv2
import math
from ultralytics import YOLO
import time # Mantido caso queira adicionar delays ou medições de FPS

# --- CONFIGURAÇÕES ---
WEBCAM_ID = 0  # ID da sua webcam (normalmente 0 pora webcam, 2 para para camera virtual)
MODEL_PATH = "datsetpropriov2.pt"  # Caminho para o seu modelo treinado
MIN_CONFIDENCE = 0.60  # Confiança mínima para considerar uma detecção válida

# Nomes das classes - DEVEM CORRESPONDER EXATAMENTE À ORDEM DO SEU MODELO TREINADO
# Se seu modelo foi treinado com essas classes nessa ordem:
CLASS_NAMES = ['capacete', 'colete', 'luvas', 'mascara', 'oculos']
# Se a ordem ou os nomes forem diferentes no seu modelo, ajuste esta lista!

# --- INICIALIZAÇÃO ---

# Webcam
print(f"Tentando abrir a webcam ID: {WEBCAM_ID}...")
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print(f"ERRO: Não foi possível abrir a webcam ID {WEBCAM_ID}. Verifique se está conectada e não em uso.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Largura do frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Altura do frame
print("Webcam aberta com sucesso.")

# Modelo YOLO
print(f"Carregando modelo YOLO de: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    #model.to('cpu')
    print(f"Modelo YOLO '{MODEL_PATH}' carregado. Dispositivo: {model.device}")
except Exception as e:
    print(f"ERRO ao carregar o modelo YOLO '{MODEL_PATH}': {e}")
    cap.release() # Liberar webcam se o modelo não carregar
    exit()

# --- LOOP PRINCIPAL DE PROCESSAMENTO ---
print("Iniciando processamento da webcam...")
while True:
    success, img = cap.read()
    if not success:
        print("ERRO: Não foi possível ler o frame da webcam. Encerrando.")
        break

    # Realizar a detecção com o modelo YOLO
    # verbose=False para reduzir a quantidade de logs do YOLO durante a inferência
    results = model(img, stream=True, verbose=False)

    # Lista para armazenar os nomes dos EPIs detectados neste frame (opcional, para exibir um resumo)
    detected_ppe_names_in_frame = set()

    for r in results:  # Iterar sobre os resultados da detecção no frame
        boxes = r.boxes
        for box in boxes:
            # Obter confiança da detecção
            confidence = math.ceil(box.conf[0] * 100) / 100
            
            # Pular detecções com confiança abaixo do mínimo
            if confidence < MIN_CONFIDENCE:
                continue

            # Obter índice da classe e nome da classe
            cls_index = int(box.cls[0])
            
            # Validação do índice da classe para evitar erros
            if cls_index < 0 or cls_index >= len(CLASS_NAMES):
                print(f"Aviso: Índice de classe {cls_index} inválido para CLASS_NAMES. Pulando esta detecção.")
                continue
            current_class_name = CLASS_NAMES[cls_index]

            # Adicionar o nome do EPI detectado ao conjunto (para evitar duplicatas no resumo)
            detected_ppe_names_in_frame.add(current_class_name)

            # Obter coordenadas da caixa delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- Desenhar na tela ---
            # Definir uma cor para as detecções (você pode personalizar por classe se quiser)
            color = (0, 255, 0)  # Verde para todas as detecções válidas
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{current_class_name}: {confidence:.2f}"
            text_y_pos = y1 - 10 if y1 > 20 else y1 + 20 # Ajustar posição do texto se estiver perto do topo
            cv2.putText(img, label, (x1, text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- Mostrar um resumo dos EPIs detectados no frame (opcional) ---
    if detected_ppe_names_in_frame:
        summary_text = f"Detectado: {', '.join(sorted(list(detected_ppe_names_in_frame)))}"
    else:
        summary_text = "Nenhum EPI detectado"
    
    # Adicionar um fundo para o texto de status para melhor visibilidade
    (text_width, text_height), baseline = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (5, 5), (10 + text_width, 10 + text_height + baseline), (0,0,0), cv2.FILLED) # Fundo preto
    cv2.putText(img, summary_text, (10, 10 + text_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # Texto branco

    # Exibir a imagem processada
    cv2.imshow('Validacao de TODOS os EPIs', img)

    # Parar o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Tecla 'q' pressionada. Encerrando...")
        break

# --- Finalização ---
print("Liberando recursos...")
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado.")
