from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
from detector_com_boxes import DetectorComBoxes
import os
import numpy as np
import time
import json

app = Flask(__name__)

detector = None
sistema_ativo = False


def generate_frames():
    """Gera frames - OTIMIZADO"""
    global detector

    while True:
        if detector and detector.rodando:
            frame = detector.processar_frame()

            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.03)
        else:
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
        return jsonify({'status': 'success', 'message': 'Sistema iniciado com gravação contínua!'})
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
        return jsonify({'status': 'success', 'message': 'Sistema parado e sessão salva!'})
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

        # Informações sobre gravação
        info_gravacao = {
            'gravacao_continua': detector.gravacao_continua,
            'gravando_clip': detector.gravando,
            'num_anomalias': len(detector.anomalias_detectadas),
            'duracao_sessao': None
        }

        if detector.gravacao_continua and detector.inicio_sessao:
            from datetime import datetime
            duracao = (datetime.now() - detector.inicio_sessao).total_seconds()
            info_gravacao['duracao_sessao'] = detector.formatar_timestamp(duracao)

        return jsonify({
            'ativo': detector.rodando,
            'gravando': detector.gravando,
            'analisando': detector.analisando,
            'num_videos': len(detector.historico_videos),
            'ultima_deteccao': ultima_det,
            'gravacao': info_gravacao
        })
    else:
        return jsonify({
            'ativo': False,
            'gravando': False,
            'analisando': False,
            'num_videos': 0,
            'ultima_deteccao': None,
            'gravacao': {
                'gravacao_continua': False,
                'gravando_clip': False,
                'num_anomalias': 0,
                'duracao_sessao': None
            }
        })


@app.route('/videos')
def get_videos():
    """Retorna clips de anomalias"""
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


@app.route('/indices')
def get_indices():
    """Retorna lista de índices de sessões disponíveis"""
    global detector

    if detector:
        indices = detector.get_indices_disponiveis()

        indices_info = []
        for indice_path in indices:
            try:
                with open(indice_path, 'r', encoding='utf-8') as f:
                    dados = json.load(f)

                indices_info.append({
                    'arquivo': os.path.basename(indice_path),
                    'video_sessao': dados['video_sessao'],
                    'inicio': dados['inicio_sessao'],
                    'duracao': dados['duracao_total_segundos'],
                    'total_anomalias': dados['total_anomalias'],
                    'anomalias': dados['anomalias']
                })
            except Exception as e:
                print(f"Erro ao ler índice: {e}")

        return jsonify(indices_info)

    return jsonify([])


@app.route('/indice/<filename>')
def get_indice_especifico(filename):
    """Retorna um índice específico"""
    try:
        caminho = os.path.join('videos_sessoes', filename)
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        return jsonify(dados)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/videos_anomalias/<filename>')
def serve_video_anomalia(filename):
    """Serve clips de anomalias"""
    return send_from_directory('videos_anomalias', filename)


@app.route('/videos_sessoes/<filename>')
def serve_video_sessao(filename):
    """Serve vídeos de sessão completa"""
    return send_from_directory('videos_sessoes', filename)


@app.route('/iniciar_gravacao', methods=['POST'])
def iniciar_gravacao():
    """Iniciar gravação manual de clip"""
    global detector

    try:
        if detector and detector.rodando and detector.ultimo_frame is not None:
            detector.iniciar_gravacao(detector.ultimo_frame, "gravacao_manual")
            return jsonify({'status': 'success', 'message': 'Gravação de clip iniciada!'})
        else:
            return jsonify({'status': 'error', 'message': 'Sistema inativo'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('templates'):
        os.makedirs('templates')

    print("🚀 Servidor Flask iniciando...")
    print("📱 Acesse: http://localhost:5000")
    print("✨ Sistema Dual de Gravação ativado!")
    print("   📹 Sessão completa + 🎬 Clips de anomalias")

    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000, use_reloader=False)