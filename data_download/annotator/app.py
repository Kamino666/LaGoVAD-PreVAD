from flask import Flask
from flask import render_template, abort, jsonify, request, send_file

import json
import zipfile
import logging
from uuid import uuid4
from pathlib import Path
from datetime import datetime

app = Flask(__name__)

try:
    with open('config.json', encoding='utf-8') as f:
        my_config = json.load(f)
except Exception as e:
    print(e)
    input('回车键结束')
    exit(0)


def setup_logger():
    # 第一步，创建一个logger
    logger = logging.getLogger('annotator')
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个FileHandler，用于写入日志文件
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    handler = logging.FileHandler(f"log_{time_str}_{uuid4().hex[:8]}.txt", encoding='utf-8')  # 文件名字
    handler.setLevel(logging.INFO)  # 单独handler的log等级设置
    # 第三步，设置filehandler的格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 第四步，创建StreamHandler，用于输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)  # 单独handler的log等级设置
    # 第六步，设置StreamHandler的格式
    console.setFormatter(formatter)
    # 第五步，添加handler到logger
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


logger = setup_logger()


@app.route('/')
def main():
    video_paths = [i for i in Path(my_config['input_dir']).glob('*.mp4') if not i.stem.startswith('.')]
    video_paths.sort()
    video_path_list = []
    annot_count = 0
    for vp in video_paths:
        vid_dict = {'src': f'/data/{vp.name}', 'title': vp.name}
        if Path(my_config['output_dir'], f"{vp.stem}.txt").exists():
            vid_dict['has_annot'] = True
            annot_count += 1
        else:
            vid_dict['has_annot'] = False
        video_path_list.append(vid_dict)

    labels_list = my_config['labels']
    label2chinese = my_config['label2chinese']
    ch_labels_list = [label2chinese[lbl] for lbl in labels_list]

    logger.info(f"Loading {len(video_paths)} videos from {my_config['input_dir']}.")
    logger.info(f"{annot_count}/{len(video_paths)} videos have annotation.")
    logger.info(f"Loading {len(labels_list)} labels.")

    return render_template('main.html', videos=video_path_list,
                           labels=labels_list, ch_labels=ch_labels_list)


@app.route('/labels')
def get_labels():
    labels_list = my_config['labels']
    label2chinese = my_config['label2chinese']
    ch_labels_list = [label2chinese[lbl] for lbl in labels_list]
    return jsonify(labels=labels_list, ch_labels=ch_labels_list)


@app.route('/data/<path:filename>')
def get_video(filename):
    filepath = Path(my_config['input_dir'], filename)
    if not filepath.exists():
        abort(404)

    # Windows下send_file有bug，一定得用绝对路径，且不能有`\`
    filepath = str(filepath.absolute()).replace('\\', '//')
    logger.info(f"Get video: {filepath}")
    return send_file(filepath)


@app.route('/get_annots/<path:filename>')
def get_annots(filename):
    anno_file = Path(filename).stem + '.txt'
    anno_path = Path(my_config['output_dir'], anno_file)
    if not anno_path.exists():
        return jsonify({'annotations': []})
    # Annotation Format:
    # class1 time1
    # class1 time2
    # ...
    with open(str(anno_path), encoding='utf-8') as fp:
        cls_tm_pairs = fp.read().strip().split('\n')
        cls_tm_pairs = [pair.split(' ') for pair in cls_tm_pairs if pair != '']
        print(cls_tm_pairs)
    return jsonify({'annotations': cls_tm_pairs})


@app.route('/update_annots/<path:filename>', methods=['POST'])
def update_annots(filename):
    anno_file = Path(filename).stem + '.txt'
    anno_path = Path(my_config['output_dir'], anno_file)
    action = request.form.get('action')

    if action == 'add':
        class_name = request.form.get('class_name', '<undefined>')
        timestamp = request.form.get('timestamp', None)
        if timestamp is None:
            abort(400)
        with open(str(Path(my_config['output_dir'], anno_file)), 'a+', encoding='utf-8') as fp:
            fp.write(f'\n{class_name} {timestamp}')
    elif action == 'delete':
        if not anno_path.exists():
            abort(400)
        class_name = request.form.get('class_name', None)
        timestamp = request.form.get('timestamp', None)
        query = f'{class_name} {timestamp}'
        if class_name is None or timestamp is None:
            abort(400)
        # 读取已有的标注文件，删除对应的行
        with open(str(Path(my_config['output_dir'], anno_file)), 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            lines = [line.strip() for line in lines if query != line.strip() and line.strip() != '']
        # 重新写入标注文件
        with open(str(Path(my_config['output_dir'], anno_file)), 'w+', encoding='utf-8') as fp:
            fp.write('\n'.join(lines))

    with open(str(anno_path), encoding='utf-8') as fp:
        cls_tm_pairs = fp.read().strip().split('\n')
        cls_tm_pairs = [pair.split(' ') for pair in cls_tm_pairs if pair != '']
        print(cls_tm_pairs)
    return jsonify({'annotations': cls_tm_pairs})


@app.route('/backup')
def backup():
    annot_files = list(Path(my_config['output_dir']).glob('*.txt'))
    output_file = Path(my_config['output_dir'], datetime.now().strftime(f'%Y-%m-%d_%H-%M_{uuid4().hex[:8]}.zip'))
    zip = zipfile.ZipFile(
        output_file, 'w',
        zipfile.ZIP_DEFLATED
    )
    for file in annot_files:
        zip.write(file)
    zip.close()
    return jsonify({'path': str(output_file)})


if __name__ == '__main__':
    # app.run(host='localhost', port=8001, debug=True)
    app.run(host='localhost', port=my_config['port'])
