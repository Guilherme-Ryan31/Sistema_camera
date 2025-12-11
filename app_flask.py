from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import cv2
from detector_com_boxes import DetectorComBoxes
from config_loader import ConfigLoader
import os
import numpy as np
import time
import json

app = Flask(__name__)

# Sistema multi-câmera
config = ConfigLoader()
detectores = {}  # {camera_id: detector}
sistema_ativo = False


def generate_frames(camera_id=0):
    """Gera frames para uma câmera específica"""
    global detectores

    qualidade_jpeg = config.get('gravacao', 'qualidade_jpeg', default=70)

    while True:
        detector = detectores.get(camera_id)

        if detector and detector.rodando:
            frame = detector.processar_frame()

            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, qualidade_jpeg])

                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.03)
        else:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            blank.fill(40)
            cv2.putText(blank, f"CAMERA {camera_id} PARADA", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(0.2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id=0):
    """Stream de vídeo para uma câmera específica"""
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/iniciar_sistema', methods=['POST'])
def iniciar_sistema():
    """Inicia todas as câmeras ativas"""
    global detectores, sistema_ativo

    try:
        cameras_ativas = config.get_cameras_ativas()

        if not cameras_ativas:
            return jsonify({'status': 'error', 'message': 'Nenhuma câmera configurada!'})

        for camera in cameras_ativas:
            camera_id = camera['id']
            camera_nome = camera['nome']
            camera_source = camera['source']

            if camera_id not in detectores:
                print(f"🎥 Iniciando {camera_nome} (ID: {camera_id})...")
                detector = DetectorComBoxes(
                    video_source=camera_source,
                    camera_id=camera_id,
                    camera_nome=camera_nome,
                    config=config
                )
                detectores[camera_id] = detector

            detectores[camera_id].iniciar()

        sistema_ativo = True
        return jsonify({
            'status': 'success',
            'message': f'{len(cameras_ativas)} câmera(s) iniciada(s) com sucesso!',
            'cameras': [c['nome'] for c in cameras_ativas]
        })
    except Exception as e:
        print(f"❌ Erro: {e}")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/parar_sistema', methods=['POST'])
def parar_sistema():
    """Para todas as câmeras"""
    global detectores, sistema_ativo

    try:
        for camera_id, detector in detectores.items():
            if detector:
                print(f"🛑 Parando câmera {camera_id}...")
                detector.parar()

        sistema_ativo = False
        return jsonify({'status': 'success', 'message': 'Todas as câmeras foram paradas!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/status')
@app.route('/status/<int:camera_id>')
def get_status(camera_id=None):
    """Retorna status de uma câmera específica ou de todas"""
    global detectores, sistema_ativo

    if camera_id is not None:
        # Status de uma câmera específica
        detector = detectores.get(camera_id)

        if detector and detector.rodando:
            ultima_det = None

            if detector.ultima_deteccao:
                # ATENÇÃO: self.ultima_deteccao agora SÓ é atualizado se for anomalia,
                # e a chave 'confianca' foi removida.
                ultima_det = {
                    'acao': detector.ultima_deteccao['acao'],
                    'evento': detector.ultima_deteccao['evento'],
                    # Removido 'confianca' conforme solicitado
                    'timestamp': detector.ultima_deteccao['timestamp'].strftime('%H:%M:%S')
                }

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
                'camera_id': camera_id,
                'camera_nome': detector.camera_nome,
                'ativo': detector.rodando,
                'gravando': detector.gravando,
                'analisando': detector.analisando,
                'num_videos': len(detector.historico_videos),
                'ultima_deteccao': ultima_det,
                'gravacao': info_gravacao
            })
        else:
            return jsonify({
                'camera_id': camera_id,
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
    else:
        # Status de todas as câmeras
        cameras_status = []
        ultima_deteccao_global = None

        for cam_id, detector in detectores.items():
            if detector:
                from datetime import datetime

                # Pegar última detecção desta câmera
                ultima_det = None
                if detector.ultima_deteccao:
                    # ATENÇÃO: self.ultima_deteccao agora SÓ é atualizado se for anomalia,
                    # e a chave 'confianca' foi removida.
                    ultima_det = {
                        'acao': detector.ultima_deteccao['acao'],
                        'evento': detector.ultima_deteccao['evento'],
                        # Removido 'confianca' conforme solicitado
                        'timestamp': detector.ultima_deteccao['timestamp'].strftime('%H:%M:%S'),
                        'camera_id': cam_id,
                        'camera_nome': detector.camera_nome
                    }
                    # Usar como última detecção global (da câmera 0 ou mais recente)
                    if cam_id == 0 or ultima_deteccao_global is None:
                        ultima_deteccao_global = ultima_det

                status = {
                    'camera_id': cam_id,
                    'camera_nome': detector.camera_nome,
                    'ativo': detector.rodando,
                    'gravando': detector.gravando,
                    'analisando': detector.analisando,
                    'num_anomalias': len(detector.anomalias_detectadas),
                    'ultima_deteccao': ultima_det
                }
                cameras_status.append(status)

        return jsonify({
            'sistema_ativo': sistema_ativo,
            'num_cameras': len(detectores),
            'cameras': cameras_status,
            'ultima_deteccao': ultima_deteccao_global  # Adicionar para compatibilidade
        })


@app.route('/videos')
@app.route('/videos/<int:camera_id>')
def get_videos(camera_id=None):
    """Retorna clips de anomalias de uma câmera ou todas"""
    global detectores

    if camera_id is not None:
        # CASO 1: Vídeos de uma câmera específica
        detector = detectores.get(camera_id)
        if detector:
            videos = detector.get_historico_videos()
            return jsonify([{
                'nome': v['nome'],
                'caminho': f'videos_anomalias/{camera_id}/{v["nome"]}',
                'evento': v['evento'].replace('_', ' ').title(),
                # Manda a ação traduzida. Se não tiver, manda o evento.
                'acao': v.get('acao', v['evento']), 
                'timestamp': v['timestamp'],
                'camera_id': camera_id
            } for v in videos[:8]])
    else:
        # CASO 2: Vídeos de todas as câmeras (O MAIS IMPORTANTE PARA O GRÁFICO)
        todos_videos = []
        for cam_id, detector in detectores.items():
            if detector:
                videos = detector.get_historico_videos()
                for v in videos[:8]:
                    v_info = {
                        'nome': v['nome'],
                        'caminho': f'videos_anomalias/{cam_id}/{v["nome"]}',
                        'evento': v['evento'].replace('_', ' ').title(),
                        # Manda a ação traduzida. Se não tiver, manda o evento.
                        'acao': v.get('acao', v['evento']), 
                        'timestamp': v['timestamp'],
                        'camera_id': cam_id,
                        'camera_nome': detector.camera_nome
                    }
                    todos_videos.append(v_info)

        # Ordenar por timestamp
        todos_videos.sort(key=lambda x: x['timestamp'], reverse=True)
        return jsonify(todos_videos[:8])

    return jsonify([])


@app.route('/indices')
@app.route('/indices/<int:camera_id>')
def get_indices(camera_id=None):
    """Retorna lista de índices de sessões disponíveis"""
    global detectores

    if camera_id is not None:
        # Índices de uma câmera específica
        detector = detectores.get(camera_id)
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
                        'anomalias': dados['anomalias'],
                        'camera_id': camera_id
                    })
                except Exception as e:
                    print(f"Erro ao ler índice: {e}")

            return jsonify(indices_info)
    else:
        # Índices de todas as câmeras
        todos_indices = []
        for cam_id, detector in detectores.items():
            if detector:
                indices = detector.get_indices_disponiveis()
                for indice_path in indices:
                    try:
                        with open(indice_path, 'r', encoding='utf-8') as f:
                            dados = json.load(f)

                        todos_indices.append({
                            'arquivo': os.path.basename(indice_path),
                            'video_sessao': dados['video_sessao'],
                            'inicio': dados['inicio_sessao'],
                            'duracao': dados['duracao_total_segundos'],
                            'total_anomalias': dados['total_anomalias'],
                            'anomalias': dados['anomalias'],
                            'camera_id': cam_id,
                            'camera_nome': detector.camera_nome
                        })
                    except Exception as e:
                        print(f"Erro ao ler índice: {e}")

        return jsonify(todos_indices)

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


@app.route('/videos_anomalias/<int:camera_id>/<filename>')
def serve_video_anomalia(camera_id, filename):
    """Serve clips de anomalias de uma câmera específica"""
    pasta = os.path.join('videos_anomalias', f'camera_{camera_id}')
    return send_from_directory(pasta, filename)

@app.route('/videos_anomalias/<filename>')
def serve_video_anomalia_legacy(filename):
    """Serve clips de anomalias (rota legacy - tenta camera 0)"""
    pasta = os.path.join('videos_anomalias', 'camera_0')
    if os.path.exists(os.path.join(pasta, filename)):
        return send_from_directory(pasta, filename)
    # Fallback: procura em todas as pastas de câmeras
    for cam_dir in os.listdir('videos_anomalias'):
        if cam_dir.startswith('camera_'):
            pasta_cam = os.path.join('videos_anomalias', cam_dir)
            if os.path.exists(os.path.join(pasta_cam, filename)):
                return send_from_directory(pasta_cam, filename)
    return jsonify({'error': 'Vídeo não encontrado'}), 404


@app.route('/videos_sessoes/<int:camera_id>/<filename>')
def serve_video_sessao(camera_id, filename):
    """Serve vídeos de sessão completa de uma câmera específica"""
    pasta = os.path.join('videos_sessoes', f'camera_{camera_id}')
    return send_from_directory(pasta, filename)

@app.route('/videos_sessoes/<filename>')
def serve_video_sessao_legacy(filename):
    """Serve vídeos de sessão (rota legacy - tenta camera 0)"""
    pasta = os.path.join('videos_sessoes', 'camera_0')
    if os.path.exists(os.path.join(pasta, filename)):
        return send_from_directory(pasta, filename)
    # Fallback: procura em todas as pastas de câmeras
    if os.path.exists('videos_sessoes'):
        for cam_dir in os.listdir('videos_sessoes'):
            if cam_dir.startswith('camera_'):
                pasta_cam = os.path.join('videos_sessoes', cam_dir)
                if os.path.exists(os.path.join(pasta_cam, filename)):
                    return send_from_directory(pasta_cam, filename)
    return jsonify({'error': 'Vídeo não encontrado'}), 404


@app.route('/iniciar_gravacao', methods=['POST'])
@app.route('/iniciar_gravacao/<int:camera_id>', methods=['POST'])
def iniciar_gravacao(camera_id=0):
    """Iniciar gravação manual de clip"""
    global detectores

    try:
        detector = detectores.get(camera_id)
        if detector and detector.rodando and detector.ultimo_frame is not None:
            detector.iniciar_gravacao(detector.ultimo_frame, "gravacao_manual")
            return jsonify({'status': 'success', 'message': f'Gravação iniciada na câmera {camera_id}!'})
        else:
            return jsonify({'status': 'error', 'message': 'Câmera inativa'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/cameras')
def get_cameras():
    """Retorna lista de câmeras configuradas"""
    cameras = config.get('cameras', default=[])
    return jsonify(cameras)


@app.route('/cameras/add', methods=['POST'])
def add_camera():
    """Adiciona nova câmera"""
    try:
        data = request.get_json()
        nome = data.get('nome', 'Nova Câmera')
        source = data.get('source', 0)
        tipo = data.get('tipo', 'webcam')

        camera_id = config.add_camera(nome, source, tipo)

        return jsonify({
            'status': 'success',
            'message': 'Câmera adicionada com sucesso!',
            'camera_id': camera_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/cameras/<int:camera_id>/toggle', methods=['POST'])
def toggle_camera(camera_id):
    """Ativa/desativa câmera"""
    try:
        data = request.get_json()
        ativa = data.get('ativa', True)

        if config.toggle_camera(camera_id, ativa):
            return jsonify({
                'status': 'success',
                'message': f'Câmera {camera_id} {"ativada" if ativa else "desativada"}!'
            })
        else:
            return jsonify({'status': 'error', 'message': 'Câmera não encontrada'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Configurações do servidor
    host = config.get('servidor', 'host', default='0.0.0.0')
    porta = config.get('servidor', 'porta', default=5000)
    debug = config.get('servidor', 'debug', default=True)

    cameras_config = config.get_cameras_ativas()

    print("🚀 Servidor Flask iniciando...")
    print(f"📱 Acesse: http://localhost:{porta}")
    print("✨ Sistema Multi-Câmera com Gravação Dual!")
    print(f"   📹 {len(cameras_config)} câmera(s) configurada(s)")
    print("   🎬 Sessão completa + Clips de anomalias")
    print("\n💡 Configurações carregadas de config.json")

    app.run(debug=debug, threaded=True, host=host, port=porta, use_reloader=False)