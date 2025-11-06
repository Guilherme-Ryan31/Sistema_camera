from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
from detector_com_boxes import DetectorComBoxes
import os
import numpy as np
import time

app = Flask(__name__)

detector = None
sistema_ativo = False


def generate_frames():
    """Gera frames - OTIMIZADO como PROJETO_CAMERA"""
    global detector

    while True:
        if detector and detector.rodando:
            frame = detector.processar_frame()

            if frame is not None:
                # OTIMIZAÇÃO: Qualidade reduzida para mais velocidade
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.03)  # Reduzido de 0.05 para 0.03
        else:
            # Frame parado - mais simples
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            blank.fill(40)
            cv2.putText(blank, "SISTEMA PARADO", (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(0.2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/iniciar_sistema', methods=['POST'])
def iniciar_sistema():
    global detector, sistema_ativo

    try:
        if detector is None:
            detector = DetectorComBoxes(video_source=0)

        detector.iniciar()
        sistema_ativo = True
        return jsonify({'status': 'success', 'message': 'Sistema iniciado!'})
    except Exception as e:
        print(f"❌ Erro: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/parar_sistema', methods=['POST'])
def parar_sistema():
    global detector, sistema_ativo

    try:
        if detector:
            detector.parar()
        sistema_ativo = False
        return jsonify({'status': 'success', 'message': 'Sistema parado!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/status')
def get_status():
    global detector, sistema_ativo

    if detector and sistema_ativo:
        ultima_det = None

        if detector.ultima_deteccao:
            ultima_det = {
                'acao': detector.ultima_deteccao['acao'],
                'evento': detector.ultima_deteccao['evento'],

                'confianca': detector.ultima_deteccao.get('confianca', 'N/D'),
                'timestamp': detector.ultima_deteccao['timestamp'].strftime('%H:%M:%S')
            }

        return jsonify({
            'ativo': detector.rodando,
            'gravando': detector.gravando,
            'analisando': detector.analisando,
            'num_videos': len(detector.historico_videos),
            'ultima_deteccao': ultima_det
        })
    else:
        return jsonify({
            'ativo': False,
            'gravando': False,
            'analisando': False,
            'num_videos': 0,
            'ultima_deteccao': None
        })


@app.route('/videos')
def get_videos():
    global detector

    if detector:
        videos = detector.get_historico_videos()
        return jsonify([{
            'nome': v['nome'],
            'caminho': v['caminho'],
            'evento': v['evento'].replace('_', ' ').title(),
            'timestamp': v['timestamp']
        } for v in videos[:8]])

    return jsonify([])


@app.route('/iniciar_gravacao', methods=['POST'])
def iniciar_gravacao():
    global detector

    try:
        if detector and detector.rodando and detector.ultimo_frame is not None:
            detector.iniciar_gravacao(detector.ultimo_frame, "gravacao_manual")
            return jsonify({'status': 'success', 'message': 'Gravação iniciada!'})
        else:
            return jsonify({'status': 'error', 'message': 'Sistema inativo'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/videos_anomalias/<filename>')
def serve_video(filename):
    return send_from_directory('videos_anomalias', filename)


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("🚀 Servidor Flask iniciando...")
    print("📱 http://localhost:5000")

    # OTIMIZAÇÃO: use_reloader=False evita carregar modelos 2x
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000, use_reloader=False)