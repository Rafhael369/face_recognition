from detector.yunet import YuNet
from recognition.sface import SFace
import sys
import argparse
import numpy as np
import cv2 as cv

# Verifique a versão do OpenCV
assert cv.__version__ >= "4.8.0", \
    "Instale o opencv-python mais recente para experimentar esta demonstração: python3 -m pip install --upgrade opencv-python"

# Combinações válidas de back-ends e destinos
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="SFace: Perda de hiperesfera restrita por sigmóide para reconhecimento facial robusto (https://ieeexplore.ieee.org/document/9318547)")
parser.add_argument('--input1', '-i1', type=str,
                    help='Uso: Defina o caminho para a imagem de entrada 1 (face original).')
parser.add_argument('--input2', '-i2', type=str,
                    help='Uso: Defina o caminho para a imagem de entrada 2 (face de comparação).')
parser.add_argument('--model', '-m', type=str, default='models/sface.onnx',
                    help='Defina o caminho do modelo, o padrão é models/sface.onnx')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Escolha um dos pares de back-end-alvo para executar esta demonstração:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0,
                    help='Uso: Tipo de distância. \'0\': Cosseno, \'1\': norm_l1. O padrão é \'0\'')
args = parser.parse_args()

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instanciar Sface para reconhecimento facial
    recognizer = SFace(modelPath=args.model,
                       disType=args.dis_type,
                       backendId=backend_id,
                       targetId=target_id)

    # Instancie o YuNet para detecção de rosto
    detector = YuNet(modelPath='models/yunet.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.9,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=backend_id,
                     targetId=target_id)

    img1 = cv.imread(args.input1)
    img2 = cv.imread(args.input2)

    # Detectar rostos nas imagens de entrada
    # detector.setInputSize([img1.shape[1], img1.shape[0]])
    # face1 = detector.infer(img1)
    # if face1 is None or face1.shape[0] == 0:
    #     print('Nenhuma face na imagem: ', args.input1)

    # detector.setInputSize([img2.shape[1], img2.shape[0]])
    # face2 = detector.infer(img2)
    # if face2 is None or face2.shape[0] == 0:
    #     print('Nenhuma face na imagem: ', args.input2)

    # Verifique se os rostos foram detectados antes de tentar combinar
    # if face1 is not None and face1.shape[0] > 0 and face2 is not None and face2.shape[0] > 0:
    result = recognizer.match(img1, img1, img2, img2)
    print('Resultado: {}.'.format(
        'Identidade iguais' if result else 'Identidades diferentes'))
