import torch
import torchvision

print(f"--- Verificacao do Ambiente PyTorch/Torchvision ---")
print(f"Caminho do executavel Python: {torch.sys.executable}") # Mostra qual Python está sendo usado
print(f"Versao do PyTorch: {torch.__version__}")
print(f"CUDA disponivel para PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Versao do CUDA (compilado com PyTorch): {torch.version.cuda}")
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dispositivos CUDA disponiveis: {torch.cuda.device_count()}")
else:
    print("CUDA NAO ESTA DISPONIVEL para o PyTorch.")

print(f"Versao do Torchvision: {torchvision.__version__}")

# Teste crucial: torchvision.ops.nms com CUDA
if torch.cuda.is_available():
    print("\n--- Testando torchvision.ops.nms na CUDA ---")
    try:
        # Criar tensores de exemplo na GPU
        boxes_exemplo = torch.rand(10, 4, device='cuda') * 100 
        scores_exemplo = torch.rand(10, device='cuda')

        print(f"Boxes de exemplo (shape {boxes_exemplo.shape}, device {boxes_exemplo.device}):\n{boxes_exemplo[:2]}") # Mostrar os 2 primeiros
        print(f"Scores de exemplo (shape {scores_exemplo.shape}, device {scores_exemplo.device}):\n{scores_exemplo[:2]}") # Mostrar os 2 primeiros

        # Executar NMS
        resultado_nms = torchvision.ops.nms(boxes_exemplo, scores_exemplo, iou_threshold=0.5)
        print(f"torchvision.ops.nms executado com sucesso na CUDA. Indices resultantes: {resultado_nms}")
        print(">>> TESTE NMS CUDA: PASSOU")
    except RuntimeError as e_runtime:
        print(f">>> TESTE NMS CUDA: FALHOU (RuntimeError)")
        print(f"    Erro: {e_runtime}")
        if "CUDA" in str(e_runtime) or "backend" in str(e_runtime):
             print("    Esta e a mesma familia de erro que voce esta vendo no seu script principal.")
    except Exception as e_geral:
        print(f">>> TESTE NMS CUDA: FALHOU (Outra Excecao)")
        print(f"    Erro: {e_geral}")
else:
    print("\nCUDA nao disponivel, pulando teste NMS na GPU.")
print("--- Fim da Verificacao ---")